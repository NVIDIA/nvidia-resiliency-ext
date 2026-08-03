# SLURM deployment helpers

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
