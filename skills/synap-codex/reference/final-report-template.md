# Final report template

Fill this in, in the chat, at step 10. Plain words only (see `instances-explained.md`): no raw status words and no mention of Synap "environments". Every line gets an answer; "skipped" needs a reason. Paste the verify output unedited after it.

```
Synap setup report
- Nothing was committed. No commit, no `git add -A`, no push, no stash, no amend: you review and commit.
- Instance: <name> (<id>), memory design <ready, in plain words>, key stored in <file> as <variable name(s), e.g. SYNAP_API_KEY_<NAME> when --env-suffix was used> (masked <synap_...abcd>); other Synap instances in this project and their variable names: <none | list>
- Instance visibility (several instances only): <human wants them to share memories | wants them isolated, policy to be set by the human in the dashboard | not asked, one instance>
- Dashboard shows finished only after the first memory and a useful recall: <yes (verify passed before auth logout) | no, because ...>
- Setup key revoked: <yes (auth logout exited 0 and reported revoked) | pending (reason; the key stays active until <expiry>, 7 days at most; open item)>
- Path: <live stream ON by default | human said no because ...: explicit SDK path, and what that leaves out in one sentence | this host cannot run a long-lived process: explicit SDK path>. Reasons: <...>.
- Identity: <what identifies a person: field and where it comes from | per-session id (confirmed by you), so no returning-person memory, no profile, no per-person erasure>
- Write: <call used, where in the code, when it runs, timeout/retries, what is lost if it fails (409 during build, 429, outage)>
- Read: <where recall is added to the model prompt (file, function) | NOT wired, because ...>. Memory informs; <your rules> decide.
- Stream: <wired with all five events, session end and clean shutdown | not wired, because (human said no / host cannot run it)>. Dependency check: <result>.
- Outcome / profile / transcript at completion: <wired | skipped, because ...>
- Sensitive data: never-store list <...>; redaction tested with <samples>; fields that pass through: <...>; instance policy <approved | NOT approved, open item>; consent screen discloses third-party storage: <yes | no, flagged>
- Operational: ON/OFF log line <yes>; variables the deployed app needs: <list> (integration is silently OFF without them); SDK pinned to <version>; CLI used <version>; .env.example/README updated <yes/no>
- Tests: <names and what they cover>; wiring check: <paste of check_integration_wiring.py output>
- Test data: removed <N> memories for ids <test_user_... list>; could not remove: <the residue list the cleanup script printed | nothing>. Result line: <RESULT: CLEAN | INCOMPLETE>. Cleanup output pasted below the verify output.
- Files created (left untracked): <every new file>. Files changed (left uncommitted): <every modified file, including .gitignore if the CLI added .env to it>. Skill files added (.agents/, .claude/, skills-lock.json): commit or ignore? Your call.
- Skipped, and why: <...>
- Decisions waiting on you: <...>
- Bugs found and NOT fixed (with the reason): <...>
- Work started and not finished (exactly where it stopped): <...>
```

Then the verify output, unedited (`RESULT:` and `EXIT CODE:` lines included), then the cleanup dry run and the real cleanup run, and the sign-out sentence from `onboarding-details.md` only if `auth logout` exited 0 and reported the key revoked (exit 9 is an open item, not done).
