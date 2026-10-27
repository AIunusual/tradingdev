---
name: wait-for-check
description: Wait for an already started repository check without repeated model turns or reading unfinished logs. Use when a long check returns an execution session and the host supports background orchestration with completion events.
---

# Wait for a check

Follow [AGENTS.md](../../../AGENTS.md) for check scope, deadlines and cleanup.
This skill changes how the agent waits; it does not start another check, approve a
result or replace the check command's process supervisor.

## Start once

- Start the required command once, with its existing deadline and cleanup guard.
  Redirect complete stdout and stderr to a task-specific log before launching.
- Keep its command, candidate identity, deadline, log path and execution session.
  If it already exited, handle that result directly.
- For a running session, use the host's completion event mechanism. With
  `functions.exec` and `tools.write_stdin`, run the following loop **inside one
  orchestration call**. Do not turn each `running` response into a model turn.

## Codex orchestration

Set `active_check` to the actual session before the waiting call. Use a distinct
completion key for each check; do not reuse an active session or launch two
watchers for it.

```javascript
store("active_check", {
  session_id: SESSION_ID,
  label: "PR check",
  log_path: "/absolute/task/check.log",
  completion_key: "check-result:UNIQUE_ID",
});
```

Run this as one `functions.exec` call, including the first-line directive:

```javascript
// @exec: {"yield_time_ms": 1000, "max_output_tokens": 1000}
const job = load("active_check");
if (!job || !Number.isInteger(job.session_id) ||
    typeof job.completion_key !== "string" || !job.completion_key ||
    typeof job.log_path !== "string" || !job.log_path) {
  throw new Error("An actual check session and result/log paths are required.");
}
const watcherKey = `check-watcher:${job.session_id}`;
if (load(watcherKey)) throw new Error("This session already has a watcher.");
store(watcherKey, {state: "waiting", completion_key: job.completion_key});
try {
  for (;;) {
    const result = await tools.write_stdin({
      session_id: job.session_id, chars: "", yield_time_ms: 55000,
      max_output_tokens: 1000,
    });
    if (result.exit_code !== undefined) {
      const completed = {...job, state: "exited", exit_code: result.exit_code};
      store(job.completion_key, completed);
      store(watcherKey, completed);
      notify(completed);
      break;
    }
    if (result.session_id !== job.session_id) {
      throw new Error("The waiting tool returned an unexpected session.");
    }
  }
} catch (error) {
  const failed = {...job, state: "monitor_failed", check_status: "unknown",
    error: String(error)};
  store(job.completion_key, failed);
  store(watcherKey, failed);
  notify(failed);
}
```

The orchestration cell can initially yield a cell ID while its code continues.
That cell ID differs from the execution session. Record it; do not poll it with
repeated `functions.wait`, result-store reads, log reads or process scans.
Continue independent authorized work while the program waits. Do not send
repeated “still running” messages. If the host lacks background completion events,
state that limitation; this recipe cannot add that capability to an unrelated host.

## Handle one terminal event

On `exited`, retrieve the completed orchestration result once if needed, then read
the final log summary. An exit code alone does not replace the required verdict,
version checks or cleanup evidence. Continue the original task from that result.

On `monitor_failed`, the check's state is unknown. Report the monitoring failure
and investigate the original session; do not label the check passed, failed or
stopped, restart it, or delete its files without confirming its actual state.
An explicit user cancellation or a concrete failure can require investigation;
normal silence is not such a failure. The check's supervisor owns deadlines and
process cleanup; stopping a watcher does not stop its check.
