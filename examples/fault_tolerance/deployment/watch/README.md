# nvrx-watch

Watches an NVRx run from outside the job. Two jobs:

- **Reconcile the chain** — correct the states the job cannot fix itself, because the
  thing that was supposed to act (array task 0's EXIT trap) is gone.
- **Detect restart anomalies** — find the failures that are a *pattern* rather than an
  event: restart storms, restarts that never advance the checkpoint, a cycle that has
  been open for an hour with nothing moving.

Stdlib only, no install: it runs on a login node outside the training container, where
`nvidia_resiliency_ext` need not be present. Copy the directory and run it.

[DESIGN.md](DESIGN.md) has the architecture, the full detector catalog and the
reasoning behind the thresholds.

## Run it

Give it a **Slurm job id** — any generation of the chain — and it resolves the rest from
Slurm. Nothing else to pass:

```bash
python3 -m nvrx_watch 5728163            # observe-only: detect and page, take no action
python3 -m nvrx_watch 5728163 --act      # owner mode: also release orphaned spares
```

What it reads from the id:

- **job name + owner** — from `scontrol show job` (falling back to `sacct` once the id
  ages out). The watcher then monitors by *name*: `squeue` and the cycle-info glob both
  span every generation, so it **follows the singleton chain automatically** as
  generations turn over — you keep passing the same id, and a cron on it tracks the whole
  chain to the end.
- **cycle-info + checkpoint paths** — parsed from the job's **batch script**: the actual
  `--ft-cycle-info-dir` and `--ft-checkpoint-iteration-file` arguments, resolved from the
  script's own variables (the array-job-id becomes `*` so the glob spans generations).
  This works for **any InJob sbatch**, not just this example. The script's path comes from
  `scontrol show job … Command=` and the file is read directly — a plain read, no
  `scontrol write batch_script` (which needs owner/operator rights); it assumes you can
  read the owner's sbatch. Real production sbatches are rarely self-contained: they
  `source` common libraries that build the ft arguments and hold the path variables, and
  root those paths at the script's own location (`cd "$(dirname "$SCRIPT_PATH")/.." &&
  pwd`). Resolution **follows `source`/`.` includes** (read-only) and evaluates that
  handful of path idioms, so a real restart-matrix cell resolves fully. `submit_chain.sh`
  bakes `NVRX_WORK_DIR` into the demo sbatch it submits so the demo — which has no
  common-lib layout to root against — resolves the same way. If the script can't be read
  or its paths can't be resolved, pass `--work-dir` (or `--cycle-info-glob`) — chain
  reconciliation still runs from the job name alone.

So `--work-dir` is **not** required under `--platform slurm`; it is derived. It is only
needed for `--platform none`, or as an override.

Overrides, for when resolution isn't available (a chain submitted without `submit_chain`,
or no scheduler access):

```bash
# spell out name/dir instead of resolving from a job id
python3 -m nvrx_watch --job-name my_run --work-dir /shared/$USER/nvrx-run --act

# cycle infos only -- no scheduler; needs --work-dir, ignores the job id
python3 -m nvrx_watch --platform none --work-dir /shared/$USER/nvrx-run

python3 -m nvrx_watch --list-detectors
```

`--work-dir` is the sbatch's `NVRX_WORK_DIR`; cycle infos and the checkpoint iteration
file live under it. Other defaults: `--platform` is `slurm`, `--state-dir` is
`~/.nvrx_watch`. `--max-restarts`, `--heartbeat-url` and `--pd-routing-key` are optional
(they enable the restart-budget detector and paging). Every flag also reads from
`NVRX_WATCH_<FLAG>`.

Exit codes: `0` clean, `1` degraded (a source could not be observed — no heartbeat was
sent), `2` at least one critical finding.

## Deploy it

Cron on every login node. Every action is a no-op when there is nothing to do, and the
pager dedups on a deterministic key, so three login nodes collapse into one incident —
and a login node going down does not take the watcher with it.

```cron
# SRE monitoring a team's chain -- just the job id; name, owner and work dir resolve
# from Slurm. Observe-only is the default: detect and page, touch nothing.
*/5 * * * * NVRX_WATCH_PD_ROUTING_KEY=... NVRX_WATCH_HEARTBEAT_URL=https://... \
            python3 -m nvrx_watch 5728163 >> ~/.nvrx_watch/cron.log 2>&1
```

`--interval N` runs it as a daemon instead, though cron is preferred: it restarts the
watcher for free if it dies, which is one less thing to watch.

### Two operator personas

**An SRE** monitoring runs they do not own is the default: `nvrx-watch <job_id>` detects
everything and pages the owner, but never touches a job it lacks permission to cancel.
The one corrective action — `orphaned_generation` releasing a dead generation's stranded
cold spares — is *reported* instead (`[observe-only] owner should run: scancel …`) so the
owner can act. This needs only **read** access: `scontrol`/`squeue`/`sacct` (cluster-wide
read is typical) and the owner's cycle-info + checkpoint files. All detection and paging
is reads; only the single `scancel` is gated, and observe-only removes even the attempt.

**The job owner** adds `--act` to enable that one action — releasing their own run's
stranded spares from outside, when the in-job trap could not.

`--platform none` narrows to cycle-info-only anomaly detection (with `--work-dir`) when
scheduler access is unavailable.

**Set up the dead-man heartbeat.** `NVRX_WATCH_HEARTBEAT_URL` is pinged after each pass
that could actually observe. A pass that went blind deliberately sends nothing, so the
dead-man timer firing is the correct outcome — the alternative is a watcher that looks
healthy while seeing nothing. Works with PagerDuty heartbeats, Healthchecks.io or Dead
Man's Snitch.

`submit_chain.sh` writes `~/.nvrx_watch_expect_chain`. Without that marker, an account
with no chain running stays quiet; with it, `chain_exhausted` pages when the chain runs
out — the failure that otherwise costs a night with nothing in any log.

## What it reports

| Detector | Fires on |
|---|---|
| `orphaned_generation` | Task 0 terminal, cold spares still queued → releases them |
| `chain_exhausted` | Nothing running or queued while a chain is expected |
| `chain_not_cancelled` | A generation exited 93 but successors are still queued |
| `generation_churn` | Too many generations ending per window |
| `restart_storm` | Too many NVRx cycles per window |
| `stalled_progress` | Cycles complete but the checkpoint iteration does not move |
| `cycle_stalled` | Current cycle open with no checkpoint or cycle activity |
| `restart_budget_low` | Generation is near `--max-restarts` |
| `spares_exhausted` | No standby node and no queued spare left |
| `suspect_node` | A node shared by consecutive short-lived cycles |

Only `orphaned_generation` acts, and only by releasing spares the job's own teardown
would have released. Everything else reports. Disable any of them with
`--disable name1,name2`.

## Reporting sinks

Log (always), PagerDuty Events v2 (`--pd-routing-key` / `NVRX_WATCH_PD_ROUTING_KEY`),
and a generic JSON webhook (`NVRX_WATCH_WEBHOOK_URL`) for Slack-compatible endpoints.
Reporting is deliberately independent of the scheduler: if Slurm is unreachable, the
alert path still works, so blindness itself pages.

Findings re-fire after `NVRX_WATCH_ALERT_COOLDOWN` (1h) so a persistent condition is
never silently forgotten and a flapping one does not page every pass.

## Tests

```bash
pytest -s -vvv tests/fault_tolerance/unit/test_nvrx_watch.py
```

Detectors are pure functions over a snapshot, so the tests need no cluster and no
subprocess.
