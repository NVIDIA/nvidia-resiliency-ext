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
STATE_DIR="<shared-dir>/drained" ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}" bash drain_poller.sh &
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

An out-of-container consumer was deliberately avoided: it can't set a controlled exit code and
would fight ft_launcher's restart logic — the decision belongs where the restart is made.
