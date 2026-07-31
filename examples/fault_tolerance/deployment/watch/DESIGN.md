# nvrx-watch — design

An out-of-job watcher for NVRx deployments. It runs from cron on a login node, or as
a small daemon, and answers one question per pass: **is this run still making
progress, and if not, does something outside the job need to act?**

## Why anything runs outside the job

NVRx handles failures it can see from inside `ft_launcher`: a rank crash, a hang
caught by a section timeout, a node that fails a health check. Two classes of problem
are structurally invisible from there.

**The actor is gone.** The singleton chain's teardown is driven by array task 0's EXIT
trap. That trap does not run on SIGKILL or hard node death, and its `scancel` can fail
against an unreachable controller. Both leave the same state — a terminal task 0 with
cold spares still queued — and in both cases the process that was supposed to fix it no
longer exists. Only an outside observer can reconcile that.

**The failure is a pattern, not an event.** A generation that restarts six times in
twenty minutes, or that restarts cleanly but never advances the checkpoint iteration,
is failing. No single restart in that sequence is anomalous; NVRx's per-cycle view
cannot see the shape. `--ft-min-progress-iterations` covers one narrow case of this
(and stops the job outright); the rest needs an observer with history.

Chain reconciliation alone can be done with a shell script against `squeue`. Anomaly
detection cannot: it needs history, and it should not be Slurm-specific. Hence Python,
and hence the split below.

## Rules the design is built on

**Observation and reporting must not share fate.** Observation goes through `squeue`
and `sacct`; reporting goes over HTTPS to a paging service. If Slurm is unreachable,
the alert path still works, so blindness itself pages.

**A blind watcher must not look healthy.** Every successful pass pings a dead-man
heartbeat URL. A pass that could not observe sends *no* heartbeat and exits non-zero —
the dead-man timer firing is the correct outcome, not a bug to be suppressed.

Two more, added here:

**Idempotent by construction.** Every action is a no-op when there is nothing to do, so
the watcher can run on all three login nodes with dedup handled at the paging service.

**Detect ≠ act.** Only one detector has an action (`cancel_pending`), and it is the one
whose corrective step is already the job's own documented behaviour. Everything else
reports. A watcher that cancels jobs on a heuristic is a new failure mode. The action is
also separable from detection, and separated by default: the watcher is observe-only
unless the owner passes `--act`. Observe-only keeps every detector and every page but
suppresses the action, so an SRE can watch and notify on jobs they do not own (and
cannot `scancel`) — detection and paging are all reads, only the one action needs
ownership. The invocation is just `nvrx-watch <job_id>`: identity and run dirs are read
from Slurm — job name and owner from `scontrol`/`sacct`, and the cycle-info + checkpoint
paths from the job's batch script. The script's path comes from `scontrol show job …
Command=` and the file is read directly (a plain read; no `scontrol write batch_script`,
which needs owner/operator rights — it assumes the SRE can read the owner's sbatch). The
real `--ft-cycle-info-dir` / `--ft-checkpoint-iteration-file` args are resolved from the
script's own variables, so it works for any InJob sbatch. Real production cells are rarely
self-contained: they `source` common libraries that build the ft args and root paths at
the script's own location (`cd "$(dirname "$SCRIPT_PATH")/.." && pwd`), so resolution
follows `source`/`.` includes (read-only) and evaluates that handful of path idioms.
`submit_chain` bakes `NVRX_WORK_DIR` into the demo sbatch it submits so the demo — which
has no common-lib layout — resolves the same way. When the script is unreadable or its
paths can't be resolved, pass `--work-dir` — chain reconciliation still runs from the job
name alone.

## Architecture

```
  sources                    snapshot                detectors            outputs
  ┌──────────────┐                                 ┌──────────────┐    ┌──────────┐
  │ platform.py  │──generations──┐                 │ chain        │───▶│ actions  │
  │ (slurm|k8s)  │               │                 │ reconcilers  │    │ (cancel) │
  ├──────────────┤               ├──▶  Snapshot ──▶├──────────────┤    ├──────────┤
  │ readers.py   │──cycles───────┤                 │ restart      │───▶│ sinks.py │
  │ (cycle infos │               │                 │ anomaly      │    │ (log/PD/ │
  │  +checkpoint)│──iteration────┤                 │ detectors.py │    │ webhook) │
  ├──────────────┤               │                 └──────────────┘    ├──────────┤
  │persistence.py│──deltas───────┘                                     │heartbeat │
  │ (prior state)│                                                     └──────────┘
  └──────────────┘
```

One pass = gather a `Snapshot`, run every detector against it, emit `Finding`s, apply
the actions the enabled findings carry, write state, heartbeat. Detectors are pure
functions of `(Snapshot, Config)` — which is what makes them testable without a
cluster, and is why the state deltas a detector needs are resolved into the snapshot
rather than read from disk inside it.

### Modules

The files are the pipeline: each stage is a module, layered so the graph is a DAG
rooted at `types.py` (the shared vocabulary). Nothing below imports anything above
it.

| Module | Role | Impure? |
|---|---|---|
| `types.py` | the frozen domain types — `Snapshot`, `Finding`, `Action`, `CycleRecord`, `ChainGeneration`, … | no |
| `config.py` | `Config` + env/file/CLI precedence and work-dir→path derivation | no |
| `parsing.py` | pure parsers — ISO timestamps, Slurm nodelist expansion | no |
| `readers.py` | filesystem readers — cycle-info files, checkpoint iteration file → records | **fs** |
| `platform.py` | scheduler adapter (`Platform` protocol, `SlurmPlatform`, `NullPlatform`) | **subprocess** |
| `detectors.py` | the detectors + registry; `run(snapshot, config)` | no |
| `persistence.py` | cross-pass state file: `load` / `advance` / `save`, alert dedup | **fs** |
| `sinks.py` | outputs (`Sink` protocol, log/PagerDuty/webhook) + heartbeat | **network** |
| `runner.py` | wires one pass: gather → detect → act → report → heartbeat | orchestrator |
| `__main__.py` | CLI: argv, logging, the daemon loop | shell |

`parsing.py` is split from `readers.py` on purpose: `detectors.py` needs
`expand_nodelist` but must do no I/O, so the pure helpers live in a module a detector
can import without pulling the filesystem into its import graph. That keeps "detectors
are pure" a structural fact, not a convention. The four impure modules — `readers`,
`platform`, `persistence`, `sinks` — are exactly the ones tests stub; everything else
is exercised as pure functions over in-memory objects.

### Snapshot

| Field | Source | Used by |
|---|---|---|
| `generations` | platform (`squeue`) | chain reconcilers |
| `terminal_state(gen)` | platform (`sacct`) | orphan, no-restart-verdict |
| `cycles` | `cycle_info.<job_id>.<attempt>.<cycle>` JSON | every restart-anomaly detector |
| `checkpoint_iteration` | `--ft-checkpoint-iteration-file` (value + mtime) | progress detectors |
| `prior` | watcher state file | stall timers, deltas |
| `observed_at` | wall clock | every time-based threshold |

Cycle-info files are written by the rendezvous host at cycle start and updated at cycle
end (`fault_tolerance/cycle_info_writer.py`), so a pass sees the full restart history of
every generation that has run under the work dir, including generations that are long
gone. That history is the watcher's memory; the state file only holds what cannot be
recovered from disk (when a value was *first* seen, and what has already been paged).

One property to design around: `cycle_end_time` on the **last** cycle is not guaranteed.
It is written when the cycle ends cleanly, but a SIGKILL, a node death, or an unclean
generation exit — exactly the cases this watcher exists for — leave it empty, and an
empty end time parses to "still open". So `is_open` on the newest cycle means "running,
*or* ended without recording it". Detectors that count *completed* cycles simply exclude
it (a ≤1-cycle detection lag that self-corrects); `cycle_stalled`, which keys on the open
cycle, cross-checks generation liveness against the platform before it fires (below), so
a stale-open cycle from a dead or finished generation is not mistaken for a hung one.

> Naming collision worth knowing: the `generation` field *inside* a cycle-info file is a
> compare-and-swap counter for that file. This document's "generation" always means one
> array in the singleton chain. The code calls the latter `ChainGeneration`.

### Platform abstraction

```python
class Platform(Protocol):
    def list_generations(self, job_name: str) -> list[ChainGeneration]: ...
    def terminal_info(self, gen_id: str, task: int) -> TaskInfo | None: ...
    def cancel_pending(self, gen_id: str) -> bool: ...
```

`SlurmPlatform` implements it with `squeue -h -r -o '%F|%K|%T'` and `sacct -X -n -P`.
`%F` is the ArrayJobID — `%A` returns the element's own JobID, which SLURM reassigns as
tasks start, and grouping by it splits one array into several phantom generations.

A Kubernetes implementation would list worker pods of a JobSet, read the restart count
from the controller, and implement `cancel_pending` as a no-op (the controller owns pod
lifecycle). Every restart-anomaly detector works unchanged, because cycle-info files are
written the same way wherever `ft_launcher` runs. Only the chain reconcilers, which are
tagged `requires_platform`, are skipped when no platform is configured.

The outputs mirror this: `sinks.py` declares a `Sink` protocol (`name`, `emit(finding)`)
with `LogSink` / `PagerDutySink` / `WebhookSink` behind it, so a new alert backend is one
class and touches nothing else — the same open/closed shape as `Platform`.

## Detector catalog

Severity is what reaches the pager: `info` logs only, `warning` pages non-urgently,
`critical` pages. All thresholds are configurable; the defaults below assume a run whose
cycles last tens of minutes and that checkpoints every few hundred iterations.

### Chain reconciliation — needs a platform

| Detector | Fires when | Sev | Action |
|---|---|---|---|
| `orphaned_generation` | Task 0 terminal per `sacct`, spares still PENDING, past `grace` (120s) | warning | `cancel_pending` |
| `chain_exhausted` | No generation running or queued, and the chain marker file exists | critical | — |
| `chain_not_cancelled` | A generation ended `93` (no-restart) but successors are still queued | critical | — |
| `generation_churn` | > `max_generations_per_window` (3) generations ended within `churn_window` (6h) | warning | — |

`orphaned_generation` is the reason the grace period exists: a trap that is mid-flight
must be allowed to finish before an outside observer duplicates its work. `sacct` silence is treated as *unknown*, never as *gone* — a
watcher that cancels a live generation because `sacct` timed out is worse than one that
waits a pass.

`chain_not_cancelled` is new and covers a silent path: `task0_exit` reads `$?`, and if
anything in the launcher→`sh -c`→srun chain rewrote exit 93, `cancel_chain` never fires
and the successor generation restarts a job NVRx said must not restart. Cheap to check, and invisible if it ever regresses, so it is checked.

### Restart anomalies — platform-independent

| Detector | Fires when | Sev |
|---|---|---|
| `restart_storm` | ≥ `storm_cycles` (5) cycles started within `storm_window` (30m) | warning |
| `stalled_progress` | ≥ `stall_cycles` (3) cycles completed with no change in checkpoint iteration | critical |
| `cycle_stalled` | Current cycle open, and neither cycle info nor checkpoint has changed for `stall_seconds` (1h) | critical |
| `restart_budget_low` | Current `cycle_number` ≥ `max_restarts` × `budget_fraction` (0.8) | warning |
| `spares_exhausted` | Current cycle has no standby nodes and no PENDING spares remain | info |
| `suspect_node` | A node is active in ≥ `suspect_cycles` (3) consecutive cycles that ended sooner than `short_cycle_seconds` (10m) | warning |

`stalled_progress` is the one that matters most in practice and the one nothing else
reports. NVRx restarts cleanly, the workload comes back, and every cycle dies before the
next `--save-interval`: the job looks alive from every angle — cycles advance, ranks
report, logs move — and burns nodes indefinitely without advancing training. It is the
`--ft-min-progress-iterations` condition observed rather than enforced, which matters
because enforcement kills the job, and by the time you want the alert you would rather
have had the warning.

`cycle_stalled` is its complement: nothing is restarting because nothing is happening.
Section timeouts should catch this, so firing it means either the timeouts are too
loose for a section that hung or the launcher itself is wedged. Because an unwritten
`cycle_end_time` leaves the last cycle looking open forever, it fires only when the
platform confirms the owning generation is still running — otherwise a run that has
*ended* (its final cycle never marked closed) would false-alarm after `stall_seconds`,
and an ended run is `chain_exhausted`'s to report, not this one's. Under `--platform
none` that cross-check is unavailable, so there the detector is best-effort.

`suspect_node` is deliberately weak. Cycle-info files record which nodes were active,
not why a cycle ended, so recurrence across short cycles is correlation only. It names
a candidate for `--exclude`; it does not act.

## State and dedup

`persistence.py` owns one file, `~/.nvrx_watch/state.json`, rewritten atomically:

```json
{
  "last_pass": "2026-07-30T12:00:00Z",
  "checkpoint_iteration": {"value": 4200, "first_seen": "2026-07-30T11:20:00Z"},
  "latest_cycle": {"key": "job.0.7", "first_seen": "2026-07-30T11:55:00Z"},
  "alerts": {"nvrx-orphan-5615858": "2026-07-30T11:58:00Z"}
}
```

`first_seen` timestamps are what turn a point-in-time snapshot into a stall timer
without keeping a process alive between passes. Alerts re-fire after `alert_cooldown`
(1h) so a condition that persists is not silently forgotten, and a flapping one does not
page every minute.

Losing the state file is not an error: stall timers restart from the current pass, which
delays a detection by at most one threshold. That is the right trade for never having
the watcher itself be the thing that needs recovering.

## Failure handling

Every external call is timeout-wrapped (`timeout -k 5 30`, equivalent in Python via
`subprocess` timeouts). A hung `squeue` is exactly the silent death the dead-man timer
exists to catch, but not hanging is cheaper than being caught. Any source that fails
marks itself unavailable in the snapshot; detectors depending on it are skipped rather
than run against partial data, and the pass reports as degraded — no heartbeat.

Nothing in the watcher writes to the job's directories. It reads cycle infos and the
checkpoint iteration file, and writes only its own state and log.

## Testing

Detectors are pure functions over a `Snapshot`, so the unit tests build snapshots
directly — no cluster, no filesystem, no subprocess. The source layer is tested against
recorded `squeue`/`sacct` output and real cycle-info JSON written by
`CycleInfoWriter`'s format. See `tests/fault_tolerance/unit/test_nvrx_watch.py`.

## Open extensions

Not built, in rough order of expected value:

- **Attribution integration.** `--ft-attribution-endpoint` results carry a cause; a
  detector could distinguish "restarting on the same GPU fault" from "restarting on
  unrelated faults", which `suspect_node` currently only guesses at.
- **Per-cycle log scanning.** `cycle_log_file` in each cycle info points at that cycle's
  application log. Grepping the tail for known-fatal signatures would sharpen every
  restart-anomaly detector.
- **Kubernetes platform.** Interface is in place; see `../README.md#platform-support`.
