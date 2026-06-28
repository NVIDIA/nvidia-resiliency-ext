# nvrx-nodestatesvc

`nvrx-nodestatesvc` is the cluster-resident service that lets NVRx query
scheduler-visible node state at restart boundaries. It runs outside the
training workload, in an environment with access to Slurm commands.

The service/API contract is documented in
[`docs/source/fault_tolerance/integration/node_state_service.rst`](../../docs/source/fault_tolerance/integration/node_state_service.rst).

## Run

```bash
python -m nvidia_resiliency_ext.services.nodestatesvc \
  --host 0.0.0.0 \
  --port 8000
```

Useful options:

```text
--slurm-timeout       Timeout for each Slurm command, default 30s
--slurm-batch-size   Maximum nodes per sinfo command, default 512
--log-level          Python logging level, default INFO
```

Environment variable equivalents:

```text
NVRX_NODESTATESVC_HOST
NVRX_NODESTATESVC_HTTP_PORT
NVRX_NODESTATESVC_SLURM_TIMEOUT
NVRX_NODESTATESVC_SLURM_BATCH_SIZE
NVRX_NODESTATESVC_LOG_LEVEL
```

`/healthz` reports process liveness. `/readyz` verifies that the Slurm backend
is reachable.
