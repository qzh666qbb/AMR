# Debug Session: confirmation-stalled-request

- Status: [OPEN]
- Issue: Confirmation batch stopped advancing after 11 complete paragraphs.
- Debug Server: See .dbg/confirmation-stalled-request.env.
- Log File: .dbg/trae-debug-log-confirmation-stalled-request.ndjson
- Constraint: Preserve frozen experiment code, parameters and completed artifacts.

## Reproduction

1. Inspect runner PID 32970 and its checkpoint journal.
2. Observe ID 261 slot 1 call 012 still in_flight after about four hours.
3. Sample process stack and inspect descriptors without exposing credentials.

## Hypotheses

| ID | Hypothesis | Likelihood | Effort | Expected Signal |
|----|------------|------------|--------|-----------------|
| A | Network request blocked | High | Low | Socket/TLS wait in process stack |
| B | Output pipe blocked | Medium | Low | Write wait on stdout/stderr |
| C | Process suspended or CPU-bound | Low | Low | Stopped state or active parser stack |
| D | Host idle sleep pauses runner and watchdog | High | Low | pmset sleep events overlap heartbeat gaps |

## Evidence

- 11 baselines, 132 candidates, 33 quality groups, 33 evaluated groups.
- ID 261 slot 0 saved; slot 1 has 13 transmitted requests.
- Two earlier uncertain calls already accounted for in prior recoveries.
- Main process state S with zero instantaneous CPU.
- Debug log lines 1-3 capture external process evidence without changing frozen code.
- Native sample /tmp/Python_2026-09-12_220320_ClC2.sample.txt lines 110-111:
  all 880 main-thread samples end in _ssl -> poll.
- A confirmed as the current blocking location; B and C not supported by this
  sample. The exact remote/network cause is not established.
- PID 32970 terminated explicitly after evidence collection; no runner remained.

## Verification

Pending recovery. Preserve the original in_flight journal and count its request
against the sentence budget. Resume ID 261 slot 1 with 37 remaining calls.
An external supervisor will emit 30-second checkpoint heartbeats and stop the
child if a new in_flight request exceeds 300 wall-clock seconds. It will not retry
uncertain requests automatically. Stdout/stderr go to a generated run log.
No candidate, generation, scoring or selection code is changed.

## Follow-up Evidence and Correction

- The resumed batch completed IDs 261 and 262, reaching 13 complete paragraphs.
- ID 263 slot 0 request 018 began at 2026-09-13 13:08:53.996 +0800.
- Native `pmset -g log` records Idle Sleep at 12:48:43 and 13:08:52 that day.
  The first interval ended at 13:06:30; the second lasted 5264 seconds until
  a short DarkWake, with further sleep intervals afterward.
- The watchdog later observed an age of 6269 seconds and terminated the child.
  A wall-clock threshold cannot be enforced while its host is asleep.
- D is confirmed for the long heartbeat gap. The prior sample did not rule out
  host sleep before sampling; A remains a blocking-location observation, not
  a complete explanation of the elapsed time.

## Operational Fix Verification

- Added a separate supervisor without changing the frozen runner. It verifies
  `caffeinate -i -w <supervisor-pid>` before starting any paid workload.
- The real local assertion test verified PID 32082 held
  PreventUserIdleSystemSleep and then released the assertion normally.
- 64 tests passed, including 8 new supervisor tests for guard verification,
  terminal status, wall timeout, no automatic retries and credential isolation.
- Existing 13 paragraphs and 156 candidates passed independent audit.
- No paid requests were sent in this fix-verification turn. No experiment process
  is running. End-to-end uninterrupted batch validation remains pending.
- Idle-sleep protection does not promise operation through lid close or manual
  system sleep. Debug artifacts remain OPEN pending user confirmation.

## End-to-End Post-Fix Evidence

- The supervised run `20260913T141856-122370Z` completed all 30 baselines,
  150 injection sentences, 360 candidates and 90 evaluated groups.
- Runner exit code was 0 and supervisor status was `runner_succeeded`.
- The run lasted 6880.45 seconds with 226 persisted heartbeats; maximum gap was
  31.87 seconds. The run transmitted 1708 new calls with zero new uncertain calls.
- Independent artifact, score, quality and selection audit returned
  `complete_verified`; final analysis returned `complete_independently_verified`.
- 68 tests passed after adding final statistical verification.
- Pre-fix: the host entered Idle Sleep and heartbeat gaps exceeded one hour.
  Post-fix: the complete remaining cohort ran with a maximum 31.87-second gap.

## Current Conclusion

The long apparent stall was caused by host idle sleep pausing both the workload
and its watchdog. A sampled process also showed an SSL read wait, but that alone
did not explain the elapsed wall time. The separate supervisor's verified
idle-sleep assertion resolved the operational failure without changing the
frozen experimental protocol. Status remains [OPEN] only for the required user
confirmation before removing debug artifacts.
