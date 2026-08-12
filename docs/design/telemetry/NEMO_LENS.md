# nemo-lens Telemetry Integration

Optional [nemo-lens](https://pypi.org/project/nemo-lens/) OTel instrumentation for NVRx fault tolerance and async checkpointing.

## Scope

Two subsystems emit spans when `nemo-lens` is installed:

1. **Fault tolerance restart cycle** — `fault_tolerance/launcher.py` and `fault_tolerance/ft_rendezvous_barrier.py`
2. **Async checkpoint worker** — `checkpointing/async_ckpt/core.py`

`nemo-lens` is optional. When not installed, all instrumentation is a no-op with no behavioral change.

## Module Structure

```mermaid
graph TD
    pyproject["pyproject.toml<br/><code>otel</code> extra: nemo-lens&gt;=0.2.0"]

    subgraph shared_utils
        shim["shared_utils/telemetry.py<br/>sole owner of the nemo-lens import"]
    end

    subgraph fault_tolerance
        launcher["fault_tolerance/launcher.py"]
        rdzv["fault_tolerance/ft_rendezvous_barrier.py"]
    end

    subgraph checkpointing
        core["checkpointing/async_ckpt/core.py"]
    end

    pyproject -->|optional dep| shim
    shim --> launcher
    shim --> rdzv
    shim --> core
```

`fault_tolerance` and `checkpointing` have no knowledge of nemo-lens. The shim is the only file with a nemo-lens import.

## `shared_utils/telemetry.py`

```python
try:
    from nemo.lens import NemoLensConfig as _NemoLensConfig
    from nemo.lens import managed_span
    from nemo.lens import setup_telemetry as _setup_telemetry

    def setup_telemetry(rank, world_size):
        return _setup_telemetry(_NemoLensConfig.from_env(), rank, world_size)

except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        yield None

    class _NoOpHandle:
        def shutdown(self): pass

    def setup_telemetry(rank, world_size):
        return _NoOpHandle()
```

`NemoLensConfig` is an implementation detail of the shim. Callers never see it.

```mermaid
classDiagram
    class managed_span {
        <<context manager>>
        +__init__(group: str, name: str, tracer=None, **attributes)
        +__enter__() Span | None
        +__exit__(...)
    }

    class setup_telemetry {
        <<function>>
        +__call__(rank: int, world_size: int) TelemetryHandle
    }
```

All call sites import from `shared_utils.telemetry`. Nothing in the codebase imports from `nemo.lens` directly.

The `group` argument to `managed_span` is a string that gates whether the span is emitted. NVRx defines two groups: `"nvrx.ft"` for fault tolerance spans and `"nvrx.ckpt"` for checkpoint spans. Users enable them via the `NEMO_LENS_SPAN_GROUPS` environment variable (or equivalent `NemoLensConfig` field).

## Fault Tolerance Span Lifecycle

The FT restart cycle runs synchronously on the launcher's main thread. OTel propagates span context implicitly via `contextvars`, so spans opened inside synchronously-called functions (including `ft_rendezvous_barrier.py`) automatically nest under the current span with no explicit parent passing.

```mermaid
sequenceDiagram
    participant L as launcher.py
    participant R as ft_rendezvous_barrier.py

    L->>L: handle = setup_telemetry(rank, world_size)
    L->>L: with managed_span("nvrx.ft", "nvrx.ft.cycle")
    L->>L: with managed_span("nvrx.ft", "nvrx.ft.rendezvous")
    L->>R: run_next_rendezvous() [sync]
    R->>R: with managed_span("nvrx.ft", "nvrx.ft.rdzv.await_round")
    R-->>L: return
    L->>L: with managed_span("nvrx.ft", "nvrx.ft.health_check")
    L->>L: with managed_span("nvrx.ft", "nvrx.ft.worker_start")
    Note over L: end nvrx.ft.cycle
    L->>L: handle.shutdown()
```

## Async Checkpoint Worker: Spawn Boundary

The persistent checkpoint worker is launched with `start_method="spawn"`. A spawned process is a fresh interpreter — it inherits environment variables from the parent but no in-memory state – so it must re-initialize telemetry.

Spawned processes inherit environment variables, so `NemoLensConfig.from_env()` inside `setup_telemetry` reads the same configuration in the worker as in the parent. Nothing telemetry-related needs to cross the spawn boundary. The worker already receives `rank` as a normal argument to `async_loop`; `world_size` is not currently passed and needs to be added to `_start_worker` and the `async_loop` signature.

```mermaid
sequenceDiagram
    participant C as async_ckpt/core.py (caller)
    participant W as Worker process (spawned)

    C->>W: spawn(_worker_fn, args=(..., rank, world_size))
    W->>W: handle = setup_telemetry(rank, world_size)
    loop each checkpoint request
        W->>W: with managed_span("nvrx.ckpt", "nvrx.ckpt.save.request")
        W->>W: with managed_span("nvrx.ckpt", "nvrx.ckpt.save.write")
    end
    W->>W: handle.shutdown()
```

`nvrx.ckpt.save.write` tracks write-only duration; `nvrx.ckpt.save.request` is the outer envelope including D2H preload and is the goodput metric.

## Spans

| Span                       | Group       | Source                        |
| -------------------------- | ----------- | ----------------------------- |
| `nvrx.ft.cycle`            | `nvrx.ft`   | `launcher.py`                 |
| `nvrx.ft.rendezvous`       | `nvrx.ft`   | `launcher.py`                 |
| `nvrx.ft.rdzv.await_round` | `nvrx.ft`   | `ft_rendezvous_barrier.py`    |
| `nvrx.ft.health_check`     | `nvrx.ft`   | `launcher.py`                 |
| `nvrx.ft.worker_start`     | `nvrx.ft`   | `launcher.py`                 |
| `nvrx.ckpt.save.request`   | `nvrx.ckpt` | `async_ckpt/core.py` (worker) |
| `nvrx.ckpt.save.write`     | `nvrx.ckpt` | `async_ckpt/core.py` (worker) |

## pyproject.toml

```toml
[tool.poetry.extras]
otel = ["nemo-lens"]

[tool.poetry.dependencies]
nemo-lens = {version = ">=0.2.0", extras = ["sdk"], optional = true}
```

## Future: Cross-Thread Span Parenting

When an attribution poller thread is added, capture the OTel context before thread creation and attach it inside the thread:

```python
import opentelemetry.context as otel_ctx

ctx = otel_ctx.get_current()
def _attribution_poller():
    token = otel_ctx.attach(ctx)
    try:
        with managed_span("nvrx.ft", "nvrx.ft.attribution_poll"):
            ...
    finally:
        otel_ctx.detach(token)

threading.Thread(target=_attribution_poller, daemon=True).start()
```
