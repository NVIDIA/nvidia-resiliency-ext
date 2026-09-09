# NVRx Telemetry Contract

Optional [nemo-lens](https://github.com/nvidia-nemo/lens) OTel instrumentation for NVRx fault tolerance and async checkpointing. Enabled when the `otel` extra is installed, and controlled by nemo-lens' environment variables.

## Rules

Design rules for this implementation:

1. **No API changes solely for telemetry information:** The NVRX APIs must not change simply to pass additional data needed only for telemetry.
2. **Spans must be self-contained:** A span must lie entirely within one logical unit of code, wherever possible given the current code architecture.
3. **A span ends promptly:** A span must end promptly and not remain open. If a span covers a long-running task, e.g., a multi-iteration training run, we use a starting zero-duration span and emit a back-dated span at closing time.
4. **Correlation is by attributes:** Spans are correlated using filter or group-by over span record attributes. We do not rely on parentage for correlating spans.

## Scope

| Area                          | Files                                                                                                     |
| ----------------------------- | --------------------------------------------------------------------------------------------------------- |
| Fault tolerance restart cycle | `fault_tolerance/launcher.py`, `fault_tolerance/ft_rendezvous_barrier.py`, `shared_utils/health_check.py` |
| Async checkpointing           | `checkpointing/async_ckpt/core.py`                                                                        |

```mermaid
graph TD
    pyproject["pyproject.toml<br/><code>otel</code> extra"]
    subgraph shared_utils
        shim["shared_utils/telemetry.py<br/>sole owner of the nemo-lens import"]
    end
    subgraph fault_tolerance
        launcher["launcher.py"]
        rdzv["ft_rendezvous_barrier.py"]
        hc["health_check.py"]
    end
    subgraph checkpointing
        core["async_ckpt/core.py"]
    end
    pyproject -->|optional dep| shim
    shim --> launcher
    shim --> rdzv
    shim --> hc
    shim --> core
```

`shared_utils/telemetry.py` imports nemo-lens optionally: if the package is present it delegates to it, and if it is absent every export becomes a no-op. All other files in NVRX import telemetry, never anything from nemo-lens. This avoids any conditional checks for telemetry in NVRX logic.

## `shared_utils/telemetry.py`

Provides (for use in NVRX) three groups of functions:

- **Spans** — `span`, `trace_fn`, `ManualSpan`, `Phase`, `mark`, `backdated_span`, `set_span_attributes`, `record_process_startup`
- **Cross-process context** — `extended_resource_attributes`, `publish_resource_attributes`
- **Lifecycle** — `setup_telemetry`, `shutdown`, `flush`

All three are nops if nemo-lens is not present.

### Where NVRx data lives in OTel

NVRx uses two of OTel's carriers, and which one a value belongs in follows from its lifetime and its cost.

| OTel carrier            | Describes                                | Set through                                                                                               | Serialized            |
| ----------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------- |
| **Resource attributes** | the emitting process, for its whole life | `OTEL_RESOURCE_ATTRIBUTES`, plus `setup_telemetry(resource_attributes=)` for what NVRx knows about itself | once per export batch |
| **Span attributes**     | one span                                 | the dict passed to `span`, `mark`, `backdated_span`, `ManualSpan`/`Phase`, `set_span_attributes`          | once per span         |

**No span reference crosses a process boundary in either direction.**

### Which carrier for which value

A value's carrier follows from whether it is constant for the _emitting process's_ lifetime. OTel's Resource is built once when the TracerProvider is constructed and is immutable after, so anything that changes during a process cannot live there.

| Process           | Constant for its life → Resource                                                                                                                    | Varies during its life → span attribute         |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| ft-launcher agent | `service.name`, `service.instance.id`, `nv.nvrx.ftl.node`                                                                                           | `nv.nvrx.cycle.index`, and every per-span value |
| Trainer worker    | `nv.nvrx.cycle.index`, `nv.nvrx.ftl.membership`, `nv.nvrx.ftl.infra.rank`, the node-budget and segment keys below (see "Placement and node budget") | elastic rank                                    |
| Checkpoint worker | same as the trainer, plus its own `service.name` and `service.instance.id`                                                                          | `nv.nvrx.ckpt.call_idx`                         |

`nv.nvrx.cycle.index` is constant for the lifetime of the training process, but not for the lifetime of the fault-tolerant launcher. Therefore, different mechanisms are used for the two scenarios.

NVRx names follow the shared `nv.` schema: fault-tolerance names under `nv.nvrx.ftl.`, checkpoint names under `nv.nvrx.ckpt.`, and the cycle entity under `nv.nvrx.cycle.`. Eventually, these names will follow a semantic convention defined by nemo-lens.

Values are restricted to OTel's attribute types — string, bool, int, double, or a homogeneous array of those; python types are disallowed.

| Mechanism             | Shape                                                                                  | Used by                                                    |
| --------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `@trace_fn`           | the span _is_ a method                                                                 | `worker_launch`, `teardown`, `completion_sync`             |
| `with span(...)`      | the span is a block                                                                    | `await_round`, `health_check`, most `ckpt` spans           |
| `ManualSpan`          | open and close cross block boundaries, bounded duration                                | `rendezvous`, `attribution`                                |
| `mark(...)`           | an instant; returns its `SpanContext`                                                  | `cycle_start`, `run_start`, `fault`                        |
| `backdated_span(...)` | already elapsed, reconstructed from two timestamps; accepts an explicit parent context | `nv.nvrx.ftl.python.startup`, `nv.nvrx.ftl.python.imports` |
| `Phase`               | a window too long to hold a span open: a start mark now, a backdated span at close     | `cycle`, `run`                                             |

`ManualSpan` owns the `ExitStack` bookkeeping and no-ops while nothing is open, so callers need no guards. Ideally, this mechanism should also be owned by nemo-lens.

`Phase` is used for telemetry with long-running tasks. `open()` creates a zero-duration span `<name>_start` and makes it the active context for all child spans, while `close()` emits `<name>` backdated from the start to the end time. If the process terminates unexpectedly, this will result in a missing backdated span, but will keep all the spans that were emitted until the point of termination.

### Span registry

`SpanRegistry` is nemo-lens's process-global record of what instrumentation a job can turn on. It is keyed by namespace, one per consuming library, and **it registers group names, not span names**:

```python
SpanRegistry.register("nvrx", {"nvrx.job", "nvrx.ft", "nvrx.ckpt", "nvrx.ckpt.phases"}, _PRESETS)
```

A span group is a family of spans. Each call site names its group. Nemo-lens checks if a span group is enabled before deciding to emit telemetry. A disabled span group should have minimal overhead in the nemo-lens architecture.

| Group              | Contents                                                                                                   |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| `nvrx.job`         | `nv.nvrx.ftl.python.startup`, `nv.nvrx.ftl.python.imports`                                                 |
| `nvrx.ft`          | every fault-tolerance span: cycle, run, rendezvous, health check, fault, teardown, attribution             |
| `nvrx.ckpt`        | a checkpoint request from the outside: `save.schedule`, `save.request`, `save.finalize`                    |
| `nvrx.ckpt.phases` | that request broken into stages: `stage_wait`, `shm_drain`, `stage`, `preload`, `write`, `completion_sync` |

| Preset      | Groups                             |
| ----------- | ---------------------------------- |
| `default`   | `nvrx.job`, `nvrx.ft`, `nvrx.ckpt` |
| `per_step`  | the above plus `nvrx.ckpt.phases`  |
| `profiling` | every NVRx group                   |

Nemo-lens uses the union of span groups recorded for the enabled preset to determine which spans to record. This allows NVRX and the training framework to independently register their presets.

## Identity

### Resource attributes

`OTEL_RESOURCE_ATTRIBUTES` carries job context as comma-separated `key=value` pairs, with values percent-encoded. The SDK reads it into the Resource with no code, it is inherited across every process spawn, and it costs nothing per span — OTLP serializes a Resource once per export batch. NVRX reads the variable to propagate its own attributes to a worker process by modifying the environment for the worker, but it does not parse the variable or depend upon it for any information that feeds into NVRX logic.

We do not emit duplicate keys in `OTEL_RESOURCE_ATTRIBUTES`. For any NVRX-specific keys, we always overwrite the values as NVRX is the source of truth (we also set attributes in `setup_telemetry()` which always win over environment variables). Do not encode any information you rely on using keys that NVRX may overwrite.

NVRX sets only what describes itself:

| Attribute             | Value                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| `service.name`        | `nvrx.ft_launcher`, or `nvrx.ckpt_worker` in the checkpoint worker                                 |
| `service.instance.id` | unique per emitting process — the agent, the trainer, and the checkpoint worker must never collide |
| `nv.nvrx.ftl.node`    | this node's identity                                                                               |

`OTEL_SERVICE_NAME` is set by the launching environment, so a service always names itself using `setup_telemetry(service_name, instance_id)`. Rank and instance ID are known by the service that launches this one and are propagated via the `OTEL_RESOURCE_ATTRIBUTES` environment variable. Values published via environment variables are always encoded as string types.

Every process and span group that is enabled will export telemetry. An OTel collector may route these spans to different consumers.

### Published to workers

`_start_workers` extends the worker's environment with the `OTEL_RESOURCE_ATTRIBUTES` variable if it is not already set, or updates it if it is already set. It adds or updates the following keys:

- `nv.nvrx.cycle.index`
- `nv.nvrx.ftl.membership`
- `nv.nvrx.ftl.infra.rank`
- `nv.dl.launch.nnodes.active`
- `nv.dl.launch.nnodes.spare`

and under `--ft-segment` also adds:

- `nv.nvrx.ftl.segment`
- `nv.nvrx.ftl.infra.cluster_uuid`

`nv.nvrx.cycle.index` is a resource attribute for the training process, but is a span attribute for ft-launcher.

## Fault tolerance

### Cycle lifecycle

A cycle opens with a `nv.nvrx.ftl.cycle_start` marker, which exports immediately and whose `SpanContext` is retained. Every span in the cycle is emitted with that context as its parent, so the cycle shares one trace per node without anything being held open. At cycle end, a backdated `nv.nvrx.ftl.cycle` span carries the duration and outcome into the same trace and nests under `nv.nvrx.ftl.cycle_start`. `nv.nvrx.ftl.run` follows the same pattern, and nests under `nv.nvrx.ftl.run_start`. In practice, `nv.nvrx.ftl.run`'s duration subtracted from `nv.nvrx.ftl.cycle`'s duration is the NVRX resiliency overhead, including rendezvous, health check, worker launch and teardown.

```mermaid
sequenceDiagram
    participant L as launcher.py
    participant R as ft_rendezvous_barrier.py

    Note over L: setup_telemetry(nvrx.ft_launcher, nvrx-agent-<host>) once at agent start
    Note over L: nv.nvrx.ftl.python.startup, nv.nvrx.ftl.python.imports (backdated)

    loop each cycle
        L->>L: mark nv.nvrx.ftl.cycle_start, retain its SpanContext
        L->>R: next_rendezvous() [sync]
        loop each rendezvous round
            R->>R: nv.nvrx.ftl.await_round
            R->>R: open nv.nvrx.ftl.rendezvous, closing the previous round's
            R->>R: nv.nvrx.ftl.health_check
        end
        R->>R: close nv.nvrx.ftl.rendezvous {nv.nvrx.ftl.group.rank, nv.nvrx.ftl.membership}
        R-->>L: return
        L->>L: nv.nvrx.ftl.worker_launch, then mark nv.nvrx.ftl.run_start
        Note over L: workers executing
        L->>L: mark nv.nvrx.ftl.fault (on failure)
        L->>L: backdated nv.nvrx.ftl.run
        L->>L: nv.nvrx.ftl.teardown
        L->>L: backdated nv.nvrx.ftl.cycle {nv.nvrx.cycle.outcome}
    end
```

`nv.nvrx.ftl.attribution` runs on the attribution poller's daemon thread. It therefore receives no context and emits its spans as root spans. `nv.nvrx.ftl.attribution.analyzed_cycle` carries the cycle the attribution is about, and may be used to join queries against the cycle.

### Spans

| Span                         | Group      | Source                     | Covers                                                   |
| ---------------------------- | ---------- | -------------------------- | -------------------------------------------------------- |
| `nv.nvrx.ftl.python.startup` | `nvrx.job` | `launcher.py`              | process create time to the entry point's first statement |
| `nv.nvrx.ftl.python.imports` | `nvrx.job` | `launcher.py`              | the entry point's top-level imports                      |
| `nv.nvrx.ftl.cycle_start`    | `nvrx.ft`  | `launcher.py`              | instant: a cycle began                                   |
| `nv.nvrx.ftl.cycle`          | `nvrx.ft`  | `launcher.py`              | one full restart cycle, backdated at close               |
| `nv.nvrx.ftl.await_round`    | `nvrx.ft`  | `ft_rendezvous_barrier.py` | waiting for a round to open                              |
| `nv.nvrx.ftl.rendezvous`     | `nvrx.ft`  | `ft_rendezvous_barrier.py` | one rendezvous round, after it opened                    |
| `nv.nvrx.ftl.health_check`   | `nvrx.ft`  | `ft_rendezvous_barrier.py` | `ensure_node_is_healthy`                                 |
| `nv.nvrx.ftl.worker_launch`  | `nvrx.ft`  | `launcher.py`              | `_start_workers`                                         |
| `nv.nvrx.ftl.run_start`      | `nvrx.ft`  | `launcher.py`              | instant: workers are up                                  |
| `nv.nvrx.ftl.run`            | `nvrx.ft`  | `launcher.py`              | workers executing, backdated at close                    |
| `nv.nvrx.ftl.fault`          | `nvrx.ft`  | `launcher.py`              | instant: a failure was detected                          |
| `nv.nvrx.ftl.teardown`       | `nvrx.ft`  | `launcher.py`              | `_stop_workers`                                          |
| `nv.nvrx.ftl.attribution`    | `nvrx.ft`  | `health_check.py`          | an attribution lookup (root span)                        |

Both `nv.nvrx.ftl.python.startup` and `nv.nvrx.ftl.python.imports` are measured within this process — `psutil.Process().create_time()` and two `time.time()` stamps — and backdated once telemetry is up.

`fault` is an instant because `teardown` only starts once the restart decision is made; without it the interval between detecting a failure and deciding what to do is unmeasured.

A hot spare produces one `await_round` / `rendezvous` pair per round, so volume tracks restart rounds rather than poll frequency.

### Cycle outcomes

`nv.nvrx.cycle.outcome` is set on the backdated `nv.nvrx.ftl.cycle` span.

| Condition                                          | `cycle.outcome`                                    |
| -------------------------------------------------- | -------------------------------------------------- |
| `WorkerState.SUCCEEDED`                            | `completed`                                        |
| Local failure, restart granted or budget exhausted | `failed`                                           |
| Healthy node joins a peer restart                  | `peer_restart`                                     |
| Health check exclusion (`UnhealthyNodeException`)  | `excluded`                                         |
| Standby node, job ends                             | `standby`                                          |
| Attribution stop / peer no-restart                 | `terminated`                                       |
| Signal                                             | _(no cycle span emitted; the marker stands alone)_ |

The exclusion and standby handlers live in `_rendezvous`, so they cover the first rendezvous as well as every restart.

`nv.nvrx.ftl.cycle.state` and `nv.nvrx.ftl.cycle.failures` accompany a `failed` outcome. The `WorkerState` name is on the cycle span rather than on `nv.nvrx.ftl.fault`, because it describes how the cycle ended, and a consumer reading cycle outcomes should not have to join to a second span to get it.

### Placement and node budget

ft-launcher publishes additional keys into the worker's resource attributes, which are constant for the worker's lifetime

- `nv.nvrx.ftl.infra.rank` is the physical node ordinal.
- `nv.dl.launch.nnodes.active` and `nv.dl.launch.nnodes.spare` are the configured node budgets.
- `nv.nvrx.ftl.segment` and `nv.nvrx.ftl.infra.cluster_uuid` when `--ft-segment` is configured. The cluster UUID is parsed from `nvidia-smi` for segment-aware rank assignment.

### Span attributes

Resource attributes are covered under Identity; everything here is per-span.

| Attribute                                | Type | Spans                 | Notes                                             |
| ---------------------------------------- | ---- | --------------------- | ------------------------------------------------- |
| `nv.nvrx.cycle.index`                    | int  | all agent spans       | restart cycle counter                             |
| `nv.nvrx.ftl.node`                       | str  | resource, spans       | node identity                                     |
| `nv.nvrx.ftl.group.rank`                 | int  | `cycle`, `rendezvous` | elastic group rank, once assigned                 |
| `nv.nvrx.ftl.group.world_size`           | int  | `cycle`               | active node count                                 |
| `nv.nvrx.ftl.cycle.failures`             | int  | `cycle`, `fault`      | failed worker count                               |
| `nv.nvrx.ftl.cycle.state`                | str  | `cycle`               | `WorkerState` at detection, on a `failed` outcome |
| `nv.nvrx.cycle.outcome`                  | str  | `cycle`               | see above                                         |
| `nv.nvrx.ftl.rdzv.round`                 | int  | `rendezvous`          | rendezvous round number                           |
| `nv.nvrx.ftl.membership`                 | str  | `cycle`, `rendezvous` | `active`, `standby`, `late_joiner`                |
| `nv.nvrx.ftl.max_restarts`               | int  | `cycle`               | configured budget                                 |
| `nv.nvrx.ftl.remaining_restarts`         | int  | `cycle`               | budget left when the round was joined             |
| `nv.nvrx.ftl.rdzv.run_id`                | str  | `cycle`               | rendezvous run id                                 |
| `nv.nvrx.ftl.infra.rank`                 | int  | `cycle`, resource     | physical node ordinal                             |
| `nv.dl.launch.nnodes.active`             | int  | resource              | configured `min_nodes`                            |
| `nv.dl.launch.nnodes.spare`              | int  | resource              | `max_nodes - min_nodes`                           |
| `nv.nvrx.ftl.segment`                    | int  | resource              | under `--ft-segment` only                         |
| `nv.nvrx.ftl.infra.cluster_uuid`         | str  | resource              | under `--ft-segment` only; NVLink domain          |
| `nv.nvrx.ftl.attribution.analyzed_cycle` | int  | `attribution`         | the cycle the verdict is about                    |
| `nv.nvrx.ckpt.call_idx`                  | int  | `ckpt.*`              | checkpoint call index; joins across ranks         |

Spans do not carry a complete roster of the job's nodes. Each node emits its own `nv.nvrx.ftl.membership` and `nv.nvrx.ftl.group.rank` on each cycle. The complete roster may be obtained using a group-by over `job.uid` (or a similarly named identifier, set outside NVRX) and `nv.nvrx.cycle.index`.

## Checkpointing

Only `PersistentAsyncCaller` is instrumented, `TemporalAsyncCaller` is deprecated and will be removed in a future release, therefore it is not instrumented other than in methods it inherits from `AsyncCaller`.

`save.schedule` instruments the time to create and enqueue an async request to the async worker process. `save.finalize` instruments the time taken to execute finalize functions registered for execution after the checkpoint has been written to disk. `save.completion_sync` records the time spent checking for checkpoint completion across all ranks.

The device-to-host copy is done by either the worker process or the training process, depending on whether `cpu_shm_mode` is set. Depending on this, some spans may appear in one or the other process.

| Span                                | Group              | Process (`cpu_shm_mode=False`) | Process (`cpu_shm_mode=True`) | Covers                                                |
| :---------------------------------- | :----------------- | :----------------------------- | :---------------------------- | :---------------------------------------------------- |
| `nv.nvrx.ckpt.save.schedule`        | `nvrx.ckpt`        | training                       | training                      | `schedule_async_request` end to end                   |
| `nv.nvrx.ckpt.save.stage_wait`      | `nvrx.ckpt.phases` | training                       | training                      | `preload_q.join()` block                              |
| `nv.nvrx.ckpt.save.shm_drain`       | `nvrx.ckpt.phases` | N/A                            | training                      | blocking drain waiting for previous checkpoint        |
| `nv.nvrx.ckpt.save.stage`           | `nvrx.ckpt.phases` | N/A                            | training                      | D2H memory copy                                       |
| `nv.nvrx.ckpt.save.request`         | `nvrx.ckpt`        | worker                         | worker                        | one checkpoint request, end-to-end                    |
| `nv.nvrx.ckpt.save.preload`         | `nvrx.ckpt.phases` | worker                         | worker                        | assemble write buckets, includes D2H copy in IPC mode |
| `nv.nvrx.ckpt.save.write`           | `nvrx.ckpt.phases` | worker                         | worker                        | write to disk                                         |
| `nv.nvrx.ckpt.save.completion_sync` | `nvrx.ckpt.phases` | training                       | training                      | all-reduce between ranks agreeing that work is done   |
| `nv.nvrx.ckpt.save.finalize`        | `nvrx.ckpt`        | training                       | training                      | finalize callbacks, some iterations later             |

NVRX uses the `call_idx` field in `AsyncRequest` and stamps the same on spans in both training and worker processes. This allows us to correlate training and worker processes' spans for each checkpoint without sending any additional data over the IPC queue between the processes.

A single checkpoint save requires three traces to store:

- trace A (iteration N): training process' checkpoint spans
- trace B (worker, call_idx M): worker process' checkpoint spans
- trace C (iteration N+k): training process' checkpoint finalize

We do not use links between these traces as these would require additional data be carried across the IPC boundary. This would have been an API change purely for telemetry, which contravenes the no-API-changes-for-telemetry rule.

The persistent async worker process is launched using a modified environment where `OTEL_RESOURCE_ATTRIBUTES` has been set or modified to carry the worker identity. The environment is reset after launching the worker, ensuring that subsequent process spawns do not pick up resource attributes meant for the worker process.

## What NVRX expects of the training framework

NVRX provides `shared_utils/semconv.py` as a placeholder for shared keys between NVRX and the training framework. At this time, only the `CKPT_CALL_IDX` string is exported – this is what is used to correlate training and worker process' checkpoints.

```python
from nvidia_resiliency_ext.shared_utils.semconv import CKPT_CALL_IDX
```

## Initialization and shutdown

| Process           | Setup                                                         | Shutdown                |
| ----------------- | ------------------------------------------------------------- | ----------------------- |
| Launcher agent    | top of `LocalElasticAgent.run()`, before the first rendezvous | that method's `finally` |
| Checkpoint worker | top of `async_process_target`, after the spawn                | that method's `finally` |

The checkpoint worker reaches its `finally` because `_handle_sigterm` turns the parent's SIGTERM into a `SystemExit` — the same mechanism that releases its CUDA IPC handles.

Only a detected fault warrants an immediate flush, as the process may be killed moments later and the span would be lost.
