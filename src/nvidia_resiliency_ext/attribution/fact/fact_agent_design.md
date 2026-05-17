# NVRx FACT Agent Design

## Scope

`nvrx-fact-agent` is a node-local client for FACT, Failure Attribution and
Characterization. It exposes a small UDS RPC surface to the FT side, collects
local failure evidence, optionally writes a dmesg artifact, submits the evidence
to FACT, and, on the store-host instance, coordinates the per-cycle FACT GET.
FACT output is node-level attribution evidence; it is not the
application-log restart-recommendation path exposed by `--ft-attribution-endpoint`.

```text
FT failed-cycle UDS RPC -> bounded dmesg window -> FACT Attribution and optional dmesg file
```

The name mirrors what the process is: an NVRx-side agent for FACT, not a FACT
service implementation.

## Roles

| Role | Responsibility |
| --- | --- |
| Leaf FACT agent | ACK local UDS requests, collect local dmesg, optionally write the dmesg file, POST observation, write local status and `observation_id` to TCPStore. |
| Store-host FACT agent | Create FACT `attributor_id`, publish it to TCPStore, wait for per-node statuses until a deadline, perform FACT GET, and write the result artifact. |
| FT side | Start the local FACT agent, send lifecycle RPCs to it, and continue restart handling after ACK. |
| TCPStore | Shared handoff for `attributor_id`, per-node status, and per-node `observation_id`. |

## CLI

FT-side config:

```text
--ft-fact-url http://10.85.104.6:8001/latest
```

The FT side starts a local `nvrx-fact-agent` with this URL and then uses a
private UDS path to notify it. The agent owns FACT POST/GET when enabled. This
is separate from `--ft-attribution-endpoint`, which is the application-log path
for job-level `STOP`/`RESTART` recommendations.

Defaults owned by the agent:

| Setting | Default |
| --- | --- |
| source | `dmesg` |
| dmesg window | 12 minutes |
| prefilter | enabled, using FACT-relevant syslog/dmesg patterns |
| RPC transport | local UDS |

The 12-minute dmesg window is intentional. Most failures are noticed within a
few seconds, but NCCL timeout failures commonly surface around 10 minutes after
the underlying event.

## UDS RPC

On failure, the FT side sends `cycle_failed` to its local FACT agent and waits
only for an accepted ACK.

Leaf payload:

```json
{
  "event": "cycle_failed",
  "run_id": "job-or-rdzv-id",
  "cycle": 3,
  "rdzv_endpoint": "store-host:29400",
  "local_node": "node-a",
  "is_store_host": false,
  "dmesg_path": "/shared/job_health_dmesg_node-a_cycle3.log"
}
```

Store-host payload adds the context needed for FACT and result collection:

```json
{
  "event": "cycle_failed",
  "run_id": "job-or-rdzv-id",
  "cycle": 3,
  "rdzv_endpoint": "store-host:29400",
  "local_node": "node-a",
  "is_store_host": true,
  "job_id": "slurm-or-run-id",
  "role": "trainer",
  "expected_nodes": ["node-a", "node-b"],
  "ranks_per_node": 4,
  "nranks": 8,
  "dmesg_path": "/shared/job_health_dmesg_node-a_cycle3.log",
  "result_path": "/shared/job_health_fact_cycle3.log"
}
```

`dmesg_path` is an optional per-node dmesg text artifact. `result_path` is an
optional artifact path where the store-host FACT agent writes the FACT result
and submission summary. Neither path is input to FACT.

`expected_nodes` is the active node set from the failed cycle, not all possible
standby nodes. It can be large, so the RPC uses framed stream payloads rather
than Unix datagrams.

ACK only means the request was validated and queued. If UDS connect or ACK
fails, the FT side logs and continues; FACT evidence collection is not a
restart barrier.

## Workflow

On the store-host instance:

1. Create the FACT attributor and write `attributor_id` to TCPStore.
2. Start local leaf collection in the same process.
3. Wait for per-node terminal statuses until a deadline.
4. Gather `submitted` observation ids.
5. Perform one authoritative FACT GET.
6. Write the FACT result artifact and per-node submission summary.

On every instance:

1. Collect a bounded recent `dmesg` window.
2. If `dmesg_path` is present, write the raw collected text to that file.
3. Read `attributor_id` from TCPStore.
4. Apply the default prefilter.
5. Shape matching lines as one FACT syslog / `raw_loki_streams` observation.
6. POST that body to the shared `attributor_id`.
7. Write explicit local status to TCPStore.

An empty status is different from a failure: it means the node reported, but no
useful dmesg lines were available for FACT.

## FACT Evidence Shape

FACT does not read log paths. `dmesg_path` and `result_path` are postmortem
artifacts only. The agent POSTs the collected evidence contents with:

```text
source=syslog
format=raw_loki_streams
timestamp_unit=nanoseconds
```

The adapter converts lines like:

```text
gb-nvl-134-compute01: [1247249.751385] NVRM: Xid ...
```

into Loki-style syslog entries where `attributes.hostname` is the node and the
body is the kernel log line.

The current FACT syslog/dmesg match set primarily catches GPU and internal
fabric failures through XID/SXID-style messages, plus a small set of
system/kernel failure strings such as hard lockups, Lustre errors, NFS stalls,
IMEX/Fabric Manager messages, and `NV_ERR_` / `NV_WARN_` markers. Generic
`mlx5*` or HCA port-flap kernel messages do not appear to be shortlisted today.
That is a coverage gap to track because UFM is not always available.

## Store Keys

Use one namespace per run and cycle:

```text
fact_agent:<run_id>:cycle<cycle>:attributor_id
fact_agent:<run_id>:cycle<cycle>:status:<node>
fact_agent:<run_id>:cycle<cycle>:observation:<node>
fact_agent:<run_id>:cycle<cycle>:done_count
fact_agent:<run_id>:cycle<cycle>:done:<index>
fact_agent:<run_id>:cycle<cycle>:result_status
fact_agent:<run_id>:cycle<cycle>:faulty_nodes
fact_agent:<run_id>:cycle<cycle>:result_path
```

Status fields:

```text
node, cycle, source, status, observation_id?, lines_collected, bytes_collected,
dmesg_path, dmesg_write_error, error
```

Statuses:

| Status | Meaning |
| --- | --- |
| `submitted` | FACT returned an `observation_id`; include it in GET. |
| `empty` | Collection worked, but no useful dmesg lines matched. |
| `collect_failed` | Agent could not collect dmesg. |
| `post_failed` | Agent collected evidence but FACT POST failed. |
| `skipped` | Agent skipped because FACT config was unavailable. |
| `timeout` | Store-host synthetic status when no status arrives by deadline. |

## Result Semantics

The store-host FACT agent must not wait at an all-node barrier: a failed node
may be down, wedged, or intentionally absent. Missing status becomes `timeout`
and is liveness evidence, not proof that FACT found nothing.

FACT attribution identifies suspect nodes. It is not a job-level
`STOP`/`RESTART` recommendation; node reuse, avoid, or exclude is a separate
policy layer.

Missing observations are NVRx liveness evidence, not FACT evidence. A node that
kernel-panics, loses power, loses network, or fails a concrete node health check
may be excluded immediately by the orchestrator. A node that only appears in
FACT attribution should enter the repeat-offender policy path unless a
deployment explicitly configures a hard-attribution exception.

The authoritative placement decision remains centralized: the store-host side
collects statuses and FACT attribution, then the FT policy layer decides what to
exclude before the next restart.
