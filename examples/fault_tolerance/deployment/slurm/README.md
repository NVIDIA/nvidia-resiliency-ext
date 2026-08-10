# Singleton job-array chain on Slurm

Colocated in-job restart with a spare pool, plus a singleton chain that survives the
one failure in-job restart cannot repair: the loss of array task 0, which is both a
training rank and the rendezvous host. See [../README.md](../README.md) for why.

```
nvrx_singleton_array.sbatch   one generation: rendezvous, gate, ft_launcher, teardown
submit_chain.sh               computes the array shape and enqueues K generations
drain_poller.sh               optional: bridges Slurm drain state into an in-job restart
```

## Requirements

- Slurm with job arrays and `--dependency=singleton`
- A filesystem shared by every node (the rendezvous host is published through it)
- `nvidia-resiliency-ext` installed where the job runs (provides `ft_launcher`)
- A [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) checkout — public `main` is
  enough; the example trains on mock data with `NullTokenizer`, so there is no corpus,
  no vocab file and nothing to download

## Run it

Everything below is an environment variable with a working default. Only two have no
safe default, because only two depend on where *you* keep things:

```bash
export NVRX_WORK_DIR=/shared/$USER/nvrx-run   # must be visible from every node
export MEGATRON_PATH=/workspace/megatron-lm   # path to your Megatron-LM checkout

NVRX_DRY_RUN=1 ./submit_chain.sh              # print the sbatch commands and stop
./submit_chain.sh                             # submit
```

`submit_chain.sh` fails immediately if the work dir looks node-local or Megatron is not
where you said — both otherwise surface minutes into a run.

Add a third if you run in a container, in which case `MEGATRON_PATH` is a path *inside*
the image:

```bash
export NVRX_CONTAINER_IMAGE=/path/to/pytorch+nvrx.sqsh
export NVRX_CONTAINER_MOUNTS=/shared:/shared
```

Scheduler settings are sbatch's own, not ours: pass them through
(`./submit_chain.sh -A my_account -p batch`) or use `SBATCH_ACCOUNT`,
`SBATCH_PARTITION` and friends.

> **The defaults run a demo, not real training.** Out of the box a submission runs the
> **no-restart (exit 93) demo**: a tiny model on 1 node + 1 hot spare, a workload fault,
> and a checkpoint-iteration file that never advances make NVRx declare "no progress",
> exit 93, and `cancel_chain` the whole queued chain (~2–3 min). The chain is 4
> generations deep so you can see several cancelled at once. Two other modes:
> ```bash
> # hand-off demo: workload fault -> in-job restart, then GPU failure on task 0 ->
> # singleton hand-off resuming from checkpoint
> NVRX_NO_RESTART_DEMO=0 ./submit_chain.sh
>
> # real training: all injection off, scaled up
> NVRX_NO_RESTART_DEMO=0 NVRX_FAULT_INJECT=0 NVRX_INJECT_GPU_FAILURE= \
> NVRX_MODEL_PROFILE=8b NVRX_TRAIN_TASKS=32 NVRX_HOT_SPARES=1 NVRX_COLD_SPARES=32 \
> NVRX_CHAIN_DEPTH=8 NVRX_TIME_LIMIT=02:00:00 ./submit_chain.sh
> ```

Default shape: 1 training task + 1 hot spare + 3 cold spares (`--array=0-4%2`), one node
per task, 4 GPUs per node (GB200 tray), 4 generations queued — a 4-GPU training world on
2 allocated nodes (cold spares stay queued, so they cost nothing but make the pending-task
cancellation visible). Everything is overridable.

### Knobs

Everything here has a default; set them only when you want to change something.

**Shape** — `NVRX_TRAIN_TASKS` (1), `NVRX_HOT_SPARES` (1), `NVRX_COLD_SPARES` (3),
`NODES_PER_TASK` (1), `GPUS_PER_NODE` (4 — GB200 tray; set 8 for HGX/DGX),
`NVRX_CHAIN_DEPTH` (4), `NVRX_TIME_LIMIT` (00:25:00), `NVRX_JOB_NAME` (nvrx_singleton).

Give each concurrent run its own `NVRX_JOB_NAME`. On a no-restart verdict the teardown
cancels *every* queued job matching the name (`scancel --name`), so two runs sharing a
name would cancel each other's pending generations.

**Workload** — `NVRX_WORK_DIR`, `MEGATRON_PATH`, `NVRX_MODEL_PROFILE` (small|8b),
`NVRX_CONTAINER_IMAGE`, `NVRX_CONTAINER_MOUNTS` (/lustre:/lustre), `MICRO_BATCH_SIZE`,
`GLOBAL_BATCH_SIZE`, `TENSOR_PARALLEL`, `PIPELINE_PARALLEL`,
`SAVE_INTERVAL`, `SAVE_RETAIN_INTERVAL`, `TRAIN_SAMPLES`, `EXIT_DURATION_IN_MINS`.

**NVRx** — `NVRX_MAX_RESTARTS` (7), `NVRX_JOIN_TIMEOUT` (1200),
`NVRX_STORE_CONNECT_WAIT` (300), `NVRX_HEALTH_CHECK_ENDPOINT` (off),
`NVRX_SEGMENT` (1 — NVLink-domain mode; set empty on systems without a GPU
ClusterUUID, e.g. H100), `NVRX_GPUS_PER_NUMA` (2), `NVRX_LOG_DEBUG` (true),
`NVRX_RDZV_HOST_WAIT` (300).

`NVRX_STORE_CONNECT_WAIT` and `NVRX_RDZV_HOST_WAIT` share a default on purpose. A cold
spare that starts in the window between task 0's `srun` exiting and its `EXIT` trap
writing `RDZV_CLOSED` escapes the flag, and holds its node until `cancel_generation`
collects it — or, if that `scancel` failed, until one of these two expires: the gate wait
if it started before task 0 published `rdzv_host`, the store wait if after. That drain is
on the critical path:
`--dependency=singleton` holds the successor generation until the array's last task
exits. Raise `NVRX_STORE_CONNECT_WAIT` where task 0's startup can exceed it (container
pull, cold shared filesystem) — set too tight, no generation ever forms.

The wait for task 0 to start has no timeout and no knob: a timeout there would race
the scheduler and could kill a generation that was about to work.

**Fault injection** — three injectors behind the demo modes. The **default** mode is the
no-restart demo (below); `NVRX_NO_RESTART_DEMO=0` switches to the hand-off demo (both
faults below); adding `NVRX_FAULT_INJECT=0 NVRX_INJECT_GPU_FAILURE=` turns everything off
for real training.

- *Workload fault* (in-job restart on a training rank): `NVRX_FAULT_INJECT` (1),
  `NVRX_FAULT_RANKS` (2), `NVRX_FAULT_DELAY` (60), `NVRX_FAULT_TYPES` (async_exc). Uses
  `megatron.core.fault_injector`. Used by both demos (it forces the restart).
- *Node HW failure* (the rank-0 hand-off path, hand-off demo only): `NVRX_INJECT_GPU_FAILURE`
  (`1:0,2:0`) — `cycle:infra_rank`; `infra_rank 0` is array task 0, the rendezvous host, so
  it fails at cycle 1/2, ending the generation so the singleton chain restarts the next one
  from checkpoint. NVRx's health-check injector, forwarded into the container automatically.
  The default (no-restart) mode turns this off, since a host failure hands off rather than
  exiting 93.

### No-restart (exit 93) demo — the default

The terminal path where NVRx decides the job must **not** restart: `ft_launcher` exits
`93` and task 0's trap runs `cancel_chain` — cancelling the whole queued chain rather
than starting the next generation (which would just reproduce the failure). This is what
stops a doomed run instead of looping on it, so the chain is submitted several
generations deep (`NVRX_CHAIN_DEPTH` 4) to show one running and the rest cancelled.

```bash
./submit_chain.sh                    # default
NVRX_NO_RESTART_DEMO=0 ./submit_chain.sh   # opt out (hand-off demo instead)
```

It points the progress tracker (`--ft-checkpoint-iteration-file`) at a static file that
**never advances**, sets `--ft-max-no-progress-cycles 1`, keeps the workload fault (to
force a restart), and turns the host-failure injector off. The tracker seeds its baseline
from that file at startup, so the very first cycle already reads the same number as the
baseline → "no progress" → with a threshold of 1, NVRx signals `no_progress` and exits 93
after a single fault. Tunables: `NVRX_MAX_NO_PROGRESS_CYCLES` (1 — raise it to watch NVRx
restart a few times before giving up) and `NVRX_FAULT_DELAY` (60).

**Timing:** **~1 fault cycle** — roughly **2–3 min after the job starts running** (model
init + `NVRX_FAULT_DELAY` + the progress check; lower `NVRX_FAULT_DELAY` to speed it up).
Watch for `ft_launcher` exit `93`, then `cancel_chain` in task 0's stdout; the watcher
then reports `chain_exhausted` (and, if `cancel_chain` had failed, `chain_not_cancelled`).
At the demo's `NODES_PER_TASK=1` task 0 records 93 itself: one task per step, no peer to
trigger a teardown. At `NODES_PER_TASK>1` on a site with `KillOnBadExit=1` it will not — a
peer exits 93 first, the step comes down, and task 0 is signalled during its store-host
grace wait, so `srun` returns 143. That is why the trap reads `control/no_restart` rather
than `$?`. `chain_not_cancelled` keys on sacct's code, so it cannot see that case either; a
`cancel_chain` that fails there shows up as `chain_exhausted` or in `squeue`, not in that
detector.

### Hot spares vs cold spares

A **hot spare** is a task that is running and has joined the rendezvous, but holds no
workers. `--nnodes MIN:MAX` sets the world at `MIN`, so extras stay standby; when an
active node dies, NVRx promotes a standby in the next cycle without the scheduler
being involved. That is the fastest repair available.

A **cold spare** is a queued array task. It costs nothing while it waits, and is
admitted when a slot frees. Slower than a hot spare — it must be scheduled and start a
container — but a deep pool is what lets one generation absorb many failures.

`GLOBAL_BATCH_SIZE` must be divisible by the data-parallel size, computed from
`NVRX_TRAIN_TASKS` only. Hot spares never enlarge the world, so the batch size stays
valid whether or not one is in use.

### Watch it

```bash
squeue -u $USER -n nvrx_singleton -r          # generations and their tasks
ls $NVRX_WORK_DIR/nvrx/*/cycle_infos/          # one file per restart cycle
tail -f $NVRX_WORK_DIR/nvrx/*/logs/nvrx_*.log  # launcher log
cat $NVRX_WORK_DIR/checkpoints/latest_checkpointed_iteration.txt
```

Then run [`../watch`](../watch) from a login node — it is what catches the failures
neither the job nor `squeue` will tell you about.

## How one generation works

1. **Task 0 publishes.** It arms its EXIT trap *first*, then writes the rendezvous host
   by atomic rename. Publishing with `echo > file` would let a peer read an existing but
   empty file and launch against `:29400`.
2. **Peers gate.** Every other task waits for `rdzv_host`, checking `RDZV_CLOSED` first:
   when both are present, task 0 is gone and the host it left is stale.
3. **`ft_launcher` runs.** Sections bound each phase — `setup:900,step:60,
   checkpointing:300` — and out-of-section code gets 90s. NVRx restarts in place, up to
   `--max-restarts`, promoting standby nodes as needed.
4. **Teardown.** Task 0's trap writes `RDZV_CLOSED`, cancels the queued chain if NVRx
   returned the no-restart verdict (otherwise the singleton dependency would start the
   next generation and reproduce the same failure), and finally `scancel`s this array
   outright — every task, any state, so a cold spare that escaped the gate does not hold
   a node for the length of `NVRX_STORE_CONNECT_WAIT`.

   The verdict is read from a **file**, not from task 0's exit code. Where the site sets
   `KillOnBadExit=1` and `NODES_PER_TASK>1`, the first peer to exit 93 takes the step down,
   and the store host — last alive by design, since it waits ~3s after closing the
   rendezvous so peers can read the final TCPStore state — is signalled mid-wait, leaving
   `srun` to hand the batch script 143. So each task's `sh` writes
   `control/no_restart` before exiting 93; SLURM's task is that `sh`, so the write always
   precedes the exit that triggers the teardown, and any node that exited cleanly is a
   better witness than task 0's own code. The flag is per-generation and needs no cleanup;
   `cancel_chain` remains the one mechanism that stops the successors.

---

The section below documents a helper that is **independent of the singleton chain**: it
serves any in-job (InJob) restart deployment. It ships here because its two safety
invariants are deployment properties, and `nvrx_singleton_array.sbatch` happens to satisfy
both — see the end of the section.

## `drain_poller.sh` — feed SLURM drained-node info into an in-job (InJob) restart

An InJob restart re-forms the ft_launcher rendezvous **without giving up the SLURM
allocation**, so it can rejoin a node the scheduler has marked `drain`. SLURM drain state is
only visible **outside** the container (needs the slurm client), so this poller bridges it in
via a shared filesystem.

**Producer (this script).** Run it backgrounded on **array task 0's** batch step (outside the
container). Once per interval it queries `sinfo` for DRAIN-flagged nodes, intersects them with
each running array task's nodes, and writes one file per task:

```
# STATE_DIR must be per-generation -- key it by SLURM_ARRAY_JOB_ID (invariant 2 below)
STATE_DIR="<shared-dir>/${SLURM_ARRAY_JOB_ID}/drained" ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}" \
  bash drain_poller.sh &
```

Output `${STATE_DIR}/task_<K>.drained`: non-empty ⇒ a node in task K's segment is drained;
empty ⇒ clean; missing ⇒ not yet polled. One producer serves the whole array (2 slurm queries
per interval regardless of scale). See the header of `drain_poller.sh` for all knobs.

**Consumer (your integration).** Each task's **first node** reads its own `task_<K>.drained`
in-container and folds it into ft_launcher's node-health-check **at rendezvous**: non-empty ⇒
report unhealthy ⇒ ft_launcher exits failure ⇒ SLURM terminates the array task ⇒ a cold spare
replaces it. Exit non-zero, but not your deployment's "no-restart" code. Only the first node
needs to read (the file is segment-scoped and SLURM tears down the rest), so there is no
filesystem read storm.

**Two deployment invariants keep this safe — both load-bearing.** The marker is keyed by task
index alone and carries no allocation identity, so it is only meaningful while an index maps to
at most one allocation. Break either invariant and a replacement inherits the previous
allocation's verdict: it exits unhealthy on a healthy node and loops through spares, or is waved
through onto a node that really is drained.

1. **`--no-requeue`**, set explicitly in the sbatch. Requeue is the only way an index is
   re-allocated within an array job, and `NODE_FAIL` requeues automatically wherever the cluster
   sets `JobRequeue=1` — so it must not be left to inheritance. Requeue preserves both the array
   job id and the index, so per-generation scoping cannot cover it. Cold spares are fine: a spare
   comes up under a *new* index whose marker is missing, which fails open. Enforce this where the
   restart model is known — a launcher that supports both models should refuse to start the poller
   unless the job declares the no-requeue one, rather than leaving it to convention.
2. **`STATE_DIR` keyed by `SLURM_ARRAY_JOB_ID`**, i.e. per generation, alongside the other
   control files. A singleton chain restarts the next generation at index 0, so a `STATE_DIR`
   shared across generations puts the previous generation's `task_0.drained` exactly where the
   new task 0 looks. Same rule, and the same reason, as the control dir.

`nvrx_singleton_array.sbatch` satisfies both already. A deployment that cannot should put the
allocation identity *in* the marker — e.g. a `nodelist=` line the consumer compares against
`$SLURM_JOB_NODELIST`, failing open on a mismatch — rather than loosening the consumer's
single-read contract.

An out-of-container consumer was deliberately avoided: it can't set a controlled exit code and
would fight ft_launcher's restart logic — the decision belongs where the restart is made.
