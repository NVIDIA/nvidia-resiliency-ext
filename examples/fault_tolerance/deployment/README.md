# NVRx deployment examples

Reference deployments of `ft_launcher` for large-scale training, and the tooling
that watches them from outside the job.

The Python API examples one level up (`basic_ft_example.py`,
`train_ddp_sections_api.py`) show how a *workload* integrates with NVRx. These
show how a *job* is shaped around it: how spare capacity is allocated, what
happens when the rendezvous host itself dies, and how a run survives a failure
that no in-job mechanism can repair.

```
deployment/
  slurm/          singleton job-array chain (sbatch + submit wrapper)
  watch/          nvrx-watch: out-of-job chain and restart-anomaly watcher
```

## Restart models

A failure is repaired at one of two levels, and the cheaper one is worth a lot:

| Level | Mechanism | Cost | Repairs |
|---|---|---|---|
| In-job | `ft_launcher` restart cycle | ~1 min | rank crash, hang, node eviction to a spare |
| Cross-job | scheduler resubmit | full requeue | anything the job itself cannot survive |

**Colocated in-job restart** is the first row and the subject of the `slurm/`
example. A job array is submitted with more tasks than the training run needs.
The active tasks form one rendezvous; the extras are *spares*. When a node
fails, `ft_launcher` re-forms the rendezvous over the remaining healthy nodes
plus a spare, and training resumes from memory or from the last checkpoint — no
new allocation, no queue wait, no scheduler round trip.

### The gap this example closes

Colocated in-job restart has one hole: **array task 0 is special**. It is both a
training rank and the rendezvous host, so its liveness *is* the generation's
liveness. Every other task can be replaced from the spare pool; task 0 cannot,
because the endpoint every other task is dialing is its hostname. A hardware
failure on task 0's node ends the whole array.

The fix is not to make task 0 recoverable — it is to make its loss cheap. The
array is submitted `--dependency=singleton`, K deep. Losing task 0 ends the
current array; the scheduler immediately starts the next one, which rendezvouses
on a *new* task 0 and resumes from the last checkpoint. Each queued array is a
**generation**, and the chain converts an unrecoverable in-job failure into a
bounded cross-job restart with no human in the loop.

```
                  generation N                        generation N+1
   task 0  ── rendezvous host ── HW failure ──X   ⇒   task 0' ── rendezvous host ──▶
   task 1..A-1  active ranks ──────────────── exit      task 1'..A-1'  active
   task A..T-1  cold spares  ──────────────── cancelled task A'..T-1'  cold spares
                                              ▲
                          singleton dependency releases the next array
```

What makes this work without polling or liveness detection is that the three
populations at risk have three different owners:

| Population | Released by |
|---|---|
| Tasks still PENDING (cold spares) | `scancel --state=PENDING` from task 0's EXIT trap |
| Tasks waiting at the rendezvous gate | `RDZV_CLOSED` flag file written by the same trap |
| Tasks already inside `ft_launcher` | NVRx, internally |

That rests on one assumption — *when task 0 exits, its EXIT trap runs* — which
holds for every exit except SIGKILL and hard node death. `watch/` exists to
correct exactly those two cases, from outside the job.

### Why `--no-requeue`

Requeue is off for every cause, `NODE_FAIL` included, so task 0 can never
reappear under its old array index. That is what lets the other tasks treat
`RDZV_CLOSED` as *final* and exit immediately rather than waiting to see whether
task 0 is coming back. The price is that a dead node is no longer repaired in
place: it costs a generation instead of a cold spare.

The alternative model — requeue enabled, `RDZV_CLOSED` meaning "task 0 may
return, keep waiting" — trades faster recovery from task-0 loss for a much
subtler gate. This example ships the singleton model only.

## Platform support

The restart model has two halves, and only one of them is scheduler-specific.

**Platform-agnostic:** everything NVRx itself does — rendezvous, restart cycles,
section timeouts, cycle-info records, the `93` no-restart exit code, checkpoint
progress tracking.

**Platform-specific:** how spare capacity is expressed (`--array=0-N%A` and
`--dependency=singleton` on Slurm), how the rendezvous host is published, and
how the spare pool is released.

`slurm/` implements the Slurm half. A Kubernetes deployment would keep the same
NVRx configuration and replace the scheduler half:

| Concern | Slurm | Kubernetes equivalent |
|---|---|---|
| Spare pool | array tasks beyond the active width | worker replicas beyond `minReplicas` |
| One generation at a time | `--dependency=singleton` | a single Job/JobSet, restarted by the controller |
| Rendezvous host | task 0's hostname, published to a shared file | a headless Service — stable DNS, no publication step |
| Release the pool | `scancel --state=PENDING` | pod deletion by the controller |
| No requeue | `--no-requeue` | `backoffLimit: 0` on the worker Job |

The headless-Service row is worth noting: it removes the rendezvous-host
publication race entirely, and with it most of the `slurm/` script's complexity.
The K8s deployment is *not* a port of the sbatch — it is a smaller thing.

`watch/` was written with this split in mind: its detectors read NVRx cycle-info
files and are scheduler-independent; only the chain-reconciliation half talks to
Slurm, behind a `Platform` interface. See [watch/DESIGN.md](watch/DESIGN.md).
