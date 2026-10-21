# InJob deployment examples

Start with the [small GPT recovery walkthrough](../../../docs/source/fault_tolerance/examples/megatron_gpt.rst).
It explains fixed section timeouts and NVRx-backed async checkpoints through
Megatron-LM. The [DDP sections example](../train_ddp_sections_api.py) keeps
synchronous PyTorch checkpoints for teaching the monitoring API.

This directory adds scheduler deployment: hot/cold spare capacity,
independently replaceable allocation units, rendezvous-host loss, and continuation
across jobs.

```text
deployment/
  slurm/          singleton job-array chain and optional scheduler-health producer
  watch/          out-of-job watcher for chain and restart anomalies
```

## Recovery levels

| Level | Mechanism | Requirements |
|---|---|---|
| InJob restart | `ft_launcher` replaces the entire worker group | live store/control service, enough healthy nodes, restart budget, completed checkpoint |
| Cross-job | scheduler starts a successor generation | predecessor fully ended, cleanup complete, resources available, completed checkpoint |

Neither level has a fixed latency guarantee. Measure failure-to-first-committed
optimizer-step time, including detection, cleanup, rendezvous, initialization,
and checkpoint restoration.

The `slurm/` example uses Megatron distributed checkpoints with
`--async-save --use-persistent-ckpt-worker`. Training resumes from the latest
completed checkpoint; worker memory is not preserved by InJob. Submitting an
async save does not make it a recovery point until finalization completes.

A hot spare is already allocated and joined but has no training workers.
It can be promoted at a later rendezvous. A cold spare is a pending array
task: admitting it requires a scheduler allocation. Thus hot-spare promotion
does not need a new allocation, while cold replacement can involve queueing
and container startup. Neither enlarges the active training world.

## Colocated rendezvous-host failure

Array task 0 hosts the rendezvous/TCPStore service and also runs training workers
in this deployment. These are distinct roles; training rank 0 is not a universal
identity for the control host.

An ordinary worker or non-host allocation failure can be repaired from eligible
spares. If task 0's host is lost, this colocated generation cannot retain its
store. The script submits several arrays with `--dependency=singleton`.
Once every task of the failed generation has ended, the next generation becomes
eligible, obtains resources, publishes a new endpoint, and restores a checkpoint.
The dependency does not reserve resources or guarantee immediate scheduling.
An array throttle is a concurrency ceiling, not atomic initial admission.

Task 0's EXIT trap writes a closed-generation marker and cancels **all remaining
tasks of that array**, including running peers. Pending-only cancellation cannot
collect a spare that has already left PENDING. On a no-restart verdict the trap
also cancels queued successors. Use a unique job name for independent chains.

A hard node loss or SIGKILL may prevent the trap from running. The
[watcher](watch/README.md) defaults to observe-only and reports stranded spares.
For their own run, an owner can explicitly enable `--act` to release pending
spares in an orphaned generation. This limited action does not promise cleanup
of every running peer or make a failed control host highly available. Verify
that the generation has fully drained before expecting its successor to run.

Requeue is disabled for this recipe: an old task 0 will not return under the same
array identity. Cold replacement uses other array tasks; host loss uses a new
generation. A normal fixed allocation or allocation shrinking does not provide
the array recipe's independently releasable replacement groups.

## Scope and optional components

The launcher API is scheduler-independent. These scripts implement Slurm
allocation, endpoint publication, and cleanup. They do not implement a Kubernetes
operator; DNS stability alone would not preserve TCPStore state after host loss.

The experimental external `nvrx-control` deployment moves store-host
responsibilities outside compute workers. It changes the colocated failure
boundary but needs its own lifecycle and qualification; see the
[deployment guide](../../../docs/source/fault_tolerance/deployment.rst).

The scheduler segment-health producer is optional. It publishes decisions that
are consumed at the next rendezvous boundary; it does not evict running Slurm
allocations. The external watcher detects and reports anomalies separately from
InJob's section monitoring and checkpoint-progress stop policy.
