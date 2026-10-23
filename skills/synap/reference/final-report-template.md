# Final report template

Fill this in, in the chat, at step 10. Every line gets an answer; "skipped" needs a reason. Paste the verify output unedited after it.

```
Synap setup report
- Instance: <name> (<id>), build <ready>, environment <name>, key stored in <file> (masked <synap_...abcd>)
- Setup key revoked: <yes (auth logout exited 0 and reported revoked) | pending (reason; the key stays active until <expiry>, 7 days at most; open item)>
- Path: <live stream ON by default | human said no because ...: explicit SDK path, and what that leaves out in one sentence | this host cannot run a long-lived process: explicit SDK path>. Reasons: <...>.
- Identity: <what identifies a person: field and where it comes from | per-session id (confirmed by you), so no returning-person memory, no profile, no per-person erasure>
- Write: <call used, where in the code, when it runs, timeout/retries, what is lost if it fails (409 during build, 429, outage)>
- Read: <where recall is added to the model prompt (file, function) | NOT wired, because ...>. Memory informs; <your rules> decide.
- Stream: <wired with all five events, session end and clean shutdown | not wired, because (human said no / host cannot run it)>. Dependency check: <result>.
- Outcome / profile / transcript at completion: <wired | skipped, because ...>
- Sensitive data: never-store list <...>; redaction tested with <samples>; fields that pass through: <...>; instance policy <approved | NOT approved, open item>; consent screen discloses third-party storage: <yes | no, flagged>
- Operational: ON/OFF log line <yes>; variables the deployed environment needs: <list> (integration is silently OFF without them); SDK pinned to <version>; CLI used <version>; .env.example/README updated <yes/no>
- Tests: <names and what they cover>; wiring check: <paste of check_integration_wiring.py output>
- Files changed: <every file>; skill files added (.agents/, .claude/, skills-lock.json): commit or ignore?
- Skipped, and why: <...>
- Decisions waiting on you: <...>
- Bugs found and NOT fixed (with the reason): <...>
- Work started and not finished (exactly where it stopped): <...>
```

Then the verify output, unedited (`RESULT:` and `EXIT CODE:` lines included), and the sign-out sentence from `onboarding-details.md` only if `auth logout` exited 0 and reported the key revoked (exit 9 is an open item, not done).
