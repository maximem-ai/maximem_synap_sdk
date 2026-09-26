/**
 * Disk backing for the stream client's outbound buffers.
 *
 * The send queue and the unacknowledged map already survive a dropped
 * connection. Neither survives the process: both live in memory, so a hard
 * kill between a turn being recorded and the server acknowledging it loses
 * that turn with no trace anywhere.
 *
 * Every change to either buffer appends one line here, and the next start
 * reads the file back before the first event goes out. Mirrors Python's
 * `transport/outbox_journal.py` line for line, including the file format, so
 * a Python process and a Node process on the same storage root read each
 * other's journals.
 *
 * **Why a log and not a snapshot.** Rewriting both buffers on every change is
 * O(depth) per event and leaves a window between the change and the write.
 * Debouncing the write widens that window on purpose, which is the opposite of
 * the point. An append is one small line and happens before the caller is told
 * the event was accepted.
 *
 * **Why synchronous.** `appendFileSync`, not the promise form. An async write
 * still pending when the process dies is a write that did not happen, and this
 * file exists precisely for the case where the process dies.
 *
 * **Where it does not run.** Workers, edge runtimes and the browser have no
 * filesystem. `node:fs` is loaded through a guarded dynamic require and, when
 * it is not there, the journal disables itself and the SDK behaves exactly as
 * it did before: buffered events survive a reconnect and not a restart. That
 * is a real gap against Python and it is declared in SDK_PARITY.md rather than
 * papered over, because those runtimes have no disk to offer.
 */

/** The ops, one character each because they are written once per event. */
const OP_QUEUED = 'q';
const OP_UNACKED = 'u';
const OP_ACKED = 'a';
const OP_DROPPED = 'd';

/** Rewrite the file once it holds this many lines. The buffers are bounded at
 *  100 entries each, so past a few hundred lines it is mostly tombstones. */
const COMPACT_AT_LINES = 500;

interface NodeFs {
  appendFileSync: (p: string, data: string, enc: string) => void;
  readFileSync: (p: string, enc: string) => string;
  writeFileSync: (p: string, data: string, enc: string) => void;
  mkdirSync: (p: string, opts: { recursive: boolean }) => void;
  existsSync: (p: string) => boolean;
  renameSync: (a: string, b: string) => void;
  unlinkSync: (p: string) => void;
}

/**
 * `node:fs`, resolved once for the process.
 *
 * It has to be a dynamic `import()`, because this package ships an ESM build
 * and a CJS one and is also used on edge runtimes: a static import would make
 * the edge bundle fail to load over a module it can never use, and `require`
 * is not defined under ESM, which is what the test runner uses. A dynamic
 * import works in all three and simply rejects where there is no `node:fs`.
 *
 * Resolving it is therefore async, but every write here has to be synchronous
 * (an async write still pending when the process dies is a write that did not
 * happen). So it is resolved ONCE, up front, by `OutboxJournal.init()`, which
 * the stream client awaits inside `connect()` before any event can flow. After
 * that the module object is in hand and the appends are plain sync calls.
 */
let _fs: NodeFs | null = null;
let _fsResolved = false;
let _fsPending: Promise<void> | null = null;

async function resolveFs(): Promise<void> {
  if (_fsResolved) return;
  // Share one in-flight import. Two clients constructed at once would
  // otherwise each start their own, and the second could observe `_fs` still
  // null and disable itself for the life of the process.
  if (_fsPending === null) {
    _fsPending = (async () => {
      try {
        _fs = (await import('node:fs')) as unknown as NodeFs;
      } catch {
        _fs = null; // edge runtime, worker, browser
      }
      _fsResolved = true;
    })();
  }
  await _fsPending;
}

export interface JournalEvent { event_id?: string; [k: string]: unknown }

/**
 * `~/.synap`, the same root Python's SDK uses, so a Node process and a Python
 * process on one machine read each other's journals. Empty string where there
 * is no home directory to ask for, which disables the journal.
 */
export function defaultOutboxRoot(): string {
  try {
    const env = (globalThis as { process?: { env?: Record<string, string | undefined> } })
      .process?.env;
    const home = env?.['SYNAP_STORAGE_PATH'] ?? env?.['HOME'] ?? env?.['USERPROFILE'];
    if (home !== undefined && home !== '') {
      return env?.['SYNAP_STORAGE_PATH'] !== undefined && env['SYNAP_STORAGE_PATH'] !== ''
        ? env['SYNAP_STORAGE_PATH'] : `${home}/.synap`;
    }
  } catch { /* no process object: edge runtime */ }
  return '';
}

export class OutboxJournal {
  readonly path: string;
  #fs: NodeFs | null;
  #lines = 0;
  #disabled = false;

  /**
   * @param root  Directory the journal lives under. One file per instance id,
   *              because two clients on one instance share a stream in the
   *              registry and so the events are the instance's, not the
   *              object's.
   */
  /**
   * Resolve `node:fs` once for the process. The stream client awaits this in
   * `connect()` before any event can flow, so by the time anything is written
   * the module is in hand and every append is a plain synchronous call. An
   * async write still pending when the process dies is a write that did not
   * happen, and this file exists for exactly that case.
   */
  static async init(): Promise<void> {
    await resolveFs();
  }

  constructor(root: string, instanceId: string, fsImpl?: NodeFs | null) {
    // An empty root means there is nowhere to put this: no home directory, or
    // the caller turned it off. Without this guard the path became
    // `/outbox/<id>.jsonl`, which is a write to the filesystem root.
    this.path = root === '' ? '' : `${root}/outbox/${instanceId}.jsonl`;
    this.#fs = fsImpl !== undefined ? fsImpl : _fs;
    if (this.#fs === null || this.path === '') this.#disabled = true;
  }

  get enabled(): boolean { return !this.#disabled; }

  // ------------------------------------------------------------------ writes

  recordQueued(event: JournalEvent): void { this.#append({ op: OP_QUEUED, e: event }); }
  recordUnacked(event: JournalEvent): void { this.#append({ op: OP_UNACKED, e: event }); }
  recordAcked(id: string): void { this.#append({ op: OP_ACKED, id }); }

  /** An event the buffers gave up on. It does not come back on restart either:
   *  replaying what the depth or age bound deliberately evicted would quietly
   *  undo the bound that evicted it. */
  recordDropped(id: string | undefined): void {
    if (id === undefined || id === '') return;
    this.#append({ op: OP_DROPPED, id });
  }

  // ------------------------------------------------------------------- reads

  /**
   * Replay the file into `{queued, unacked}`, oldest first.
   *
   * A partial last line from a kill mid-append stops the read there. What came
   * before it is the prefix that was durably written; a torn line is an event
   * the caller was never told had been accepted.
   */
  load(): { queued: JournalEvent[]; unacked: JournalEvent[] } {
    const empty = { queued: [], unacked: [] };
    if (this.#disabled || this.#fs === null) return empty;
    let text: string;
    try {
      if (!this.#fs.existsSync(this.path)) return empty;
      text = this.#fs.readFileSync(this.path, 'utf8');
    } catch {
      return empty;
    }
    const queued = new Map<string, JournalEvent>();
    const unacked = new Map<string, JournalEvent>();
    let lines = 0;
    for (const raw of text.split('\n')) {
      const line = raw.trim();
      if (line === '') continue;
      let rec: { op?: string; id?: string; e?: JournalEvent };
      try {
        rec = JSON.parse(line) as typeof rec;
      } catch {
        break; // torn last line; everything before it stands
      }
      lines += 1;
      const id = rec.op === OP_QUEUED || rec.op === OP_UNACKED
        ? rec.e?.event_id : rec.id;
      if (id === undefined || id === '') continue;
      if (rec.op === OP_QUEUED) {
        queued.set(id, rec.e as JournalEvent);
      } else if (rec.op === OP_UNACKED) {
        // It left the queue to be written, so it is no longer queued. Without
        // this a reconnect-then-kill replays it from both lists.
        queued.delete(id);
        unacked.set(id, rec.e as JournalEvent);
      } else if (rec.op === OP_ACKED || rec.op === OP_DROPPED) {
        queued.delete(id);
        unacked.delete(id);
      }
    }
    this.#lines = lines;
    return { queued: [...queued.values()], unacked: [...unacked.values()] };
  }

  // ------------------------------------------------------------- maintenance

  /**
   * Rewrite the file to hold exactly what the buffers hold now.
   *
   * Written beside the target and renamed over it, because rename is atomic: a
   * kill during a compaction leaves either the whole old file or the whole new
   * one, never a half-written mix.
   */
  compact(queued: JournalEvent[], unacked: JournalEvent[]): void {
    if (this.#disabled || this.#fs === null) return;
    if (queued.length === 0 && unacked.length === 0) { this.clear(); return; }
    try {
      this.#mkdir();
      const body = [
        ...unacked.map((e) => JSON.stringify({ op: OP_UNACKED, e })),
        ...queued.map((e) => JSON.stringify({ op: OP_QUEUED, e })),
      ].join('\n') + '\n';
      const tmp = `${this.path}.${Date.now()}.tmp`;
      this.#fs.writeFileSync(tmp, body, 'utf8');
      this.#fs.renameSync(tmp, this.path);
      this.#lines = queued.length + unacked.length;
    } catch {
      // Never break a turn or a close over bookkeeping.
    }
  }

  /** Nothing is outstanding, so there is nothing to replay. */
  clear(): void {
    if (this.#disabled || this.#fs === null) return;
    try { this.#fs.unlinkSync(this.path); } catch { /* already gone */ }
    this.#lines = 0;
  }

  get needsCompaction(): boolean { return this.#lines >= COMPACT_AT_LINES; }

  // ---------------------------------------------------------------- internal

  #mkdir(): void {
    const dir = this.path.slice(0, this.path.lastIndexOf('/'));
    this.#fs?.mkdirSync(dir, { recursive: true });
  }

  #append(record: Record<string, unknown>): void {
    if (this.#disabled || this.#fs === null) return;
    try {
      this.#mkdir();
      this.#fs.appendFileSync(this.path, `${JSON.stringify(record)}\n`, 'utf8');
      this.#lines += 1;
    } catch (err) {
      // One warning, then stay quiet: a read-only or full filesystem would
      // otherwise log once per event for the life of the process.
      console.warn(
        `[synap] outbox journal write failed (${String(err)}); buffered events `
        + 'will not survive a restart from here on',
      );
      this.#disabled = true;
    }
  }
}
