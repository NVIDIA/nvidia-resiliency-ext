# NVRx FACT Integration Design

This is the internal design for the NVRx FACT integration. The user-facing
operator doc is `docs/source/fault_tolerance/fact_node_attribution.rst`.

FACT, Failure Attribution and Characterization, provides node-level attribution
from host evidence. It does not produce the NVRx application-log
`STOP`/`RESTART` decision; that remains the existing application-log
attribution path.

## Scope

The integration has two legs:

1. **Current-cycle attribution:** after a failed FT cycle, live agents collect
   local dmesg, POST observations to FACT Attribution Service, and the
   store-host agent GETs one attribution result.
2. **Repeat-offender avoid policy:** the store-host agent combines current FACT
   suspect nodes with recent FACT node-history records plus an in-memory
   hot cache, then returns `avoid_nodes` for the next rendezvous placement.

Hard health/liveness failures are outside this policy. If a node fails a Node
Health Check, disappears, kernel-panics, loses power, or cannot rejoin, it is a
hard exclusion. FACT attribution is weaker suspect evidence and must fail open.

## Ownership

```text
FT launcher
  parses CLI/YAML
  starts one local nvrx-fact-agent per launcher process
  sends local UDS notifications
  applies avoid_nodes during rendezvous placement

nvrx-fact-agent on every node
  collects local dmesg evidence
  POSTs local observation to FACT Attribution Service
  queues optional dmesg/result artifacts through gRPC

nvrx-fact-agent on the store host
  creates the FACT attributor
  performs the FACT attribution GET
  queries FACT node-history
  maintains the in-memory hot cache
  computes and serves avoid_nodes over local UDS

FACT services
  own current-cycle attribution and durable historical node records
```

NVRx owns placement because it knows quorum, joined participants, and standby
capacity. FACT should eventually own the durable history query interface
because it owns FACT DB schema and ingestion paths.

There is no separate policy daemon. Extend the launcher-managed
`nvrx-fact-agent`.

## Configuration

Current-cycle FACT evidence path:

```text
--ft-fact-url
--ft-fact-agent-socket-path
--ft-fact-agent-rpc-timeout
--ft-fact-agent-store-timeout
--ft-health-log-prefix
--ft-enable-health-log-dmesg
--ft-enable-fact-result-artifact
```

History-based avoid policy:

```text
--ft-fact-history-es-url
--ft-fact-history-es-auth-file
--ft-fact-history-lookback
--ft-fact-history-index
--ft-fact-history-max-candidate-nodes
--ft-fact-history-query-timeout
--ft-fact-min-repeat-count-for-avoid
--ft-fact-max-attribution-avoids-per-cycle
```

Defaults:

```text
dmesg_window = 12m
observation_deadline = 30s

fact_policy_ready_timeout = 60s

fact_history_lookback = 14d
fact_history_index = <deployment FACT node-history index>
fact_history_max_candidate_nodes = 16
fact_history_query_timeout = 30s
min_repeat_count_for_avoid = 2
max_attribution_avoids_per_cycle = 1
```

Repeat-offender policy is available when current-cycle FACT attribution is
enabled:

```text
--ft-fact-url
```

With only `--ft-fact-url`, the policy uses the in-memory hot cache from earlier
cycles in the same NVRx run. If `--ft-fact-history-es-url` and
`--ft-fact-history-es-auth-file` are also set, the store-host agent adds durable
FACT history to the same decision. The auth-file contents must not be logged or
written to artifacts. Most knobs should stay defaulted; if the agent command
line keeps growing, prefer a generated config file over environment
variables.

## Agent Lifecycle

```text
FT parses config
FT computes session context
FT starts nvrx-fact-agent after session context and gRPC log funnel are known
FT sends UDS ping and waits for accepted=true
FT sends UDS cycle_failed after worker failure/stop
FT store-host queries UDS get_avoid_nodes before next placement
FT sends UDS shutdown on normal launcher shutdown
```

Agent startup/session args:

```text
--fact-url
--socket-path
--run-id
--rdzv-endpoint
--store-timeout
--local-node
--is-store-host
--job-id
--ranks-per-node
--username
--cluster
--health-log-prefix
--dmesg-artifact-enabled
--result-artifact-enabled
--grpc-server-address
--grpc-node-id
--fact-history-es-url
--fact-history-es-auth-file
--fact-history-lookback
--fact-history-index
--fact-history-max-candidate-nodes
--fact-history-query-timeout
--fact-min-repeat-count-for-avoid
--fact-max-attribution-avoids-per-cycle
```

Only the store-host agent uses history args. Passing them to all agents is
acceptable because each launcher starts the same binary.

## UDS RPCs

### `ping`

```json
{"event": "ping"}
```

```json
{"accepted": true}
```

### `cycle_failed`

FT sends this after workers for the failed cycle are stopped. The ACK means the
agent accepted work into its local queue; it is not a FACT or policy result.

Store-host payload:

```json
{
  "event": "cycle_failed",
  "cycle_id": "3",
  "cycle_start_time": "2026-05-10T12:00:00+00:00",
  "cycle_end_time": "2026-05-10T12:42:00+00:00",
  "expected_nodes": ["node-a", "node-b"]
}
```

Leaf payload:

```json
{
  "event": "cycle_failed",
  "cycle_id": "3",
  "cycle_start_time": "2026-05-10T12:00:00+00:00",
  "cycle_end_time": "2026-05-10T12:42:00+00:00",
  "expected_nodes": []
}
```

Response:

```json
{"accepted": true}
```

Field contract:

| Field | Scope | Meaning |
| --- | --- | --- |
| `cycle_id` | Per failed cycle | NVRx cycle id. FACT node-history needs an equivalent cycle-distinct episode id. |
| `cycle_start_time` | Per failed cycle | FACT workload start time and upper bound for history lookup. |
| `cycle_end_time` | Per failed cycle | Hot-cache event time and recency tie-breaker. Defaults to agent receive time if omitted. |
| `expected_nodes` | Store-host only | Active nodes for the failed cycle, not every allocated/spare node. Used for FACT workload scope and completion target. |

Rank shape is startup/session state: `ranks_per_node` is passed when the agent
starts, and the store host derives `nranks = ranks_per_node * len(expected_nodes)`.

### `get_avoid_nodes`

Before next placement, the store-host FT side queries its local agent:

```json
{
  "event": "get_avoid_nodes",
  "cycle_id": "3"
}
```

Possible responses:

```text
{"cycle_id": "3", "status": "ready", "avoid_nodes": ["node-a"]}
{"cycle_id": "3", "status": "skipped", "avoid_nodes": []}
{"cycle_id": "3", "status": "pending", "avoid_nodes": []}
```

The current FT side makes a single local query at placement time. Missing,
malformed, skipped, pending, or failed responses are treated as an empty avoid
list. Candidate ranking, reason strings, timestamps, and history details are
internal agent state; FT consumes only `avoid_nodes`.

## TCPStore Contract

TCPStore carries only FACT evidence control-plane state:

```text
fact_agent:<run_id>:cycle<cycle_id>:attributor_id
fact_agent:<run_id>:cycle<cycle_id>:done_count
```

| Key | Writer | Reader | Meaning |
| --- | --- | --- | --- |
| `attributor_id` | Store-host | All agents | FACT attributor id, or a failure sentinel. |
| `done_count` | All agents | Store-host | Atomic count of agents that reached terminal local outcome. |

`done_count` increments for successful submission and terminal local failures:
empty evidence, collection failure, POST failure, or attributor-id failure. It
is not a count of successful FACT submissions. Nodes that never increment
before the deadline are missing/liveness evidence.

No per-node status, observation ids, faulty nodes, result path, result status,
or policy output is written to TCPStore. These are JSONL artifact records when
enabled, or in-memory policy state on the store-host agent.

## FACT Attribution Flow

### 1. Create Attributor

Store-host agent:

```text
POST <fact-url>/attributor
```

Request body:

```json
{
  "workload": {
    "id": "<job_id>:cycle<cycle_id>:<unique>",
    "type": "slurm",
    "job_start_time": "<cycle_start_time>",
    "job_end_time": "<failure notification time>",
    "status": "FAILED",
    "nodes": ["node-a", "node-b"],
    "ranks_per_node": 4,
    "nranks": 8,
    "name": "<SLURM_JOB_NAME or unknown>",
    "username": "<session username>"
  },
  "metadata": {
    "cluster": "<session cluster>",
    "agent": "nvrx-ft-launcher",
    "tenant": "<tenant or unknown>",
    "ruleset": "default"
  }
}
```

The store-host writes the returned `attributor_id` to TCPStore, then runs the
same dmesg observation path locally.

### 2. POST Dmesg Observation

Every live agent:

```text
collect dmesg --since now - 12m
queue raw dmesg artifact when enabled and non-empty
wait for TCPStore[attributor_id]
apply built-in FACT dmesg filter
convert matching lines to raw_loki_streams
POST <fact-url>/attributor/{attributor_id}/observation
queue fact_observation JSONL when enabled
increment done_count
```

The 12-minute window covers NCCL timeout cases where the kernel event may be
roughly 10 minutes old. The FACT workload `job_start_time` remains the actual
cycle start time; the dmesg collection window is independent.

FACT observation body:

```json
{
  "context": {
    "time_interval": {
      "start": "<collection_start>",
      "end": "<collection_end>"
    },
    "resources": "AllJobResources",
    "source": "syslog",
    "format": "raw_loki_streams"
  },
  "body": "[{\"stream\":...,\"values\":...}]"
}
```

The agent emits Loki stream attributes with `hostname`, `appname=kernel`, and
the dmesg line body. Default `dmesg` output has monotonic kernel timestamps, so
the agent assigns synthetic Loki timestamps near collection end time to keep
short-cycle evidence inside the FACT workload window.

An empty or fully prefiltered dmesg window produces no FACT POST, records
`status = "empty"` when result JSONL is enabled, and still increments
`done_count`. Observation POST failures are retried with jitter only while the
observation deadline still has useful time remaining.

### 3. GET Attribution

Store-host waits for `done_count >= len(expected_nodes)` or the observation
deadline, then performs one authoritative GET:

```text
GET <fact-url>/attributor/{attributor_id}/attribution
```

The full FACT response is wrapped as `FactAttributionResult` for audit and for
repeat-offender policy input.

## Artifacts and Durability

FACT submission does not read artifacts; it receives POSTed contents. Artifacts
are optional postmortem evidence and require the launcher gRPC log funnel. There
is no direct shared-file fallback.

For cycle `N`, paths are derived from `health_logging.prefix`:

```text
<prefix> -> <prefix_without_ext>_dmesg_cycle<N>.log
<prefix> -> <prefix_without_ext>_fact_cycle<N>.log
```

### Dmesg Artifact

Enabled by `--ft-enable-health-log-dmesg`. All live agents queue their raw
collected dmesg text before FACT filtering to one shared per-cycle file.
Production collection prefixes each line with the source node name. Empty raw
windows queue no dmesg chunk and do not create a 0-byte file.

### Result JSONL Artifact

Enabled by `--ft-enable-fact-result-artifact`. All agents queue compact JSONL
records to the root writer.

Leaf record:

```json
{
  "record_type": "fact_observation",
  "run_id": "run-1",
  "cycle": 3,
  "job_id": "12345",
  "node": "node-a",
  "source": "dmesg",
  "status": "submitted",
  "attributor_id": "att-1",
  "observation_id": "obs-1",
  "bytes_collected": 2048,
  "lines_collected": 12,
  "dmesg_path": "/lustre/logs/job_health_dmesg_cycle3.log",
  "error": ""
}
```

Leaf status values:

| Status | Meaning |
| --- | --- |
| `submitted` | FACT returned an `observation_id`. |
| `empty` | Collection worked, but filtering left no lines to POST. |
| `collect_failed` | Local dmesg collection failed. |
| `attributor_failed` | No usable `attributor_id` was available. |
| `post_failed` | Dmesg was collected, but FACT observation POST failed. |

Store-host record:

```json
{
  "record_type": "fact_result",
  "status": "complete",
  "run_id": "run-1",
  "cycle": 3,
  "job_id": "12345",
  "expected_node_count": 2,
  "completed_node_count": 1,
  "avoid_nodes": ["node-a"],
  "fact_attribution_result": {
    "attributor_id": "att-1",
    "observation_ids": [],
    "faulty_nodes": ["node-a"],
    "attribution": {}
  }
}
```

Records and dmesg payloads are queued as chunks, so bytes from different chunks
should not interleave. Ordering across nodes is not a correctness contract.

Dmesg collection, FACT POST/GET, TCPStore completion, policy computation, and
gRPC artifact drain are best-effort. On normal launcher shutdown,
`FactAgentManager` asks the agent to exit over UDS so queued JSONL can drain,
but completeness remains best-effort.

## Repeat-Offender Policy

Current FACT attribution returns suspect nodes for this cycle:

```text
current_suspect_nodes = fact_result.faulty_nodes
```

If history config is present, the store-host agent queries FACT node-history
only for those current suspects:

```text
source = configured FACT node-history source
cluster == current cluster
node in current_suspect_nodes
lookback_start <= event_time < cycle_start_time
```

MVP logic uses:

```text
cluster
node
episode_id
event_time
```

`episode_id` is used with `cluster` and `node` to deduplicate historical
episodes. The expected NVRx node-history contract is:

```text
episode_id = <job_id>_<cycle_id>
```

or an equivalent FACT-defined cycle-distinct id. The deployment adapter maps
these generic fields to the concrete FACT history backend. The FACT
node-history source does not currently expose `attributor_id` or
`observation_id`, so MVP dedupe cannot rely on those ids.

The hot cache is in-memory only for MVP:

```text
scope = current NVRx job/process
durability = none
hot_cache_episode_key = (job_id, cycle_id, node_id)
```

FACT history is the durable source once FACT PM catches up. Expected FACT PM
upload lag can be about 30 minutes, so back-to-back failures need the local hot
cache before history catches up. The hot cache is only a current-process
overlay for candidate nodes; MVP does not run a broad bidirectional
reconciliation loop with FACT history.

MVP aggregation:

```text
candidate_nodes = current_fact_result.faulty_nodes
prior_history = FACT node-history rows for candidate_nodes before cycle_start_time,
                or [] when no history source is configured
hot_overlay = earlier NVRx events for candidate_nodes already processed by this
              agent before the current policy computation
repeat_count(node) = 1 current event + prior_history_episodes + hot_overlay_episodes
```

Broad/systemic guard:

```text
if len(current_suspect_nodes) > fact_history_max_candidate_nodes:
  skip history lookup
  do not add these suspects to future repeat counts
  return avoid_nodes=[]
```

The default guard is 16 nodes. This is a broad-event boundary, not a precise
rack classifier.

Decision table:

| Evidence | Action |
| --- | --- |
| One current FACT attribution only | Suspect; no placement action. |
| Two or more total same-node events, including current | `avoid-for-retry` if feasible. |
| More than `fact_history_max_candidate_nodes` current suspects | Audit only; skip history query. |
| Health/liveness failure | Hard exclude outside this policy. |

Ranking:

```text
1. higher repeat_count
2. more recent prior same-node episode
3. lexical node name

prior_last_seen(node) =
  max(event_time from prior FACT history,
      cycle_end_time from prior unreconciled hot-cache episodes)
```

Do not use the current cycle's `cycle_end_time` for this tie-breaker; it is the
same failure event currently being handled. If the top-ranked candidate is not
feasible to avoid, MVP skips all attribution-based avoids for that cycle.

Family-aware escalation is out of MVP because `FactAttributionResult` and FACT
node-history do not currently expose a stable normalized failure family. Until
FACT provides one, the policy is same-node repeat only and stores raw symptoms
for audit. The desired future family taxonomy is subsystem-level:
`accelerator-memory`, `accelerator-gpu`, `accelerator-fabric`,
`network-fabric`, `storage`, `kernel-fatal`, `host-oom`, `UNKNOWN_FAMILY`.

## Placement Handoff

The store-host FT/rendezvous side queries local UDS:

```text
get_avoid_nodes(cycle_id) -> avoid_nodes
```

Then it validates:

```text
node is a joined participant
min_nodes / quorum still holds
rank and accelerator count still holds
standby capacity is available
max attribution avoids per cycle is respected
```

Avoided nodes are treated as standby/spare for the next retry. They are not
hard-excluded and may still join rendezvous. If the policy is absent, late,
malformed, skipped, or infeasible, rendezvous proceeds normally.

## Module Placement

Keep MVP FACT-specific code directly under `attribution/fact`:

```text
nvidia_resiliency_ext/attribution/fact/
  models.py
  history_client.py
  hot_cache.py
  repeat_offender_policy.py
```

| Module | Responsibility |
| --- | --- |
| `models.py` | Dataclasses for history records, hot-cache episodes, candidates, and decisions. |
| `history_client.py` | FACT history query and auth-file handling. |
| `hot_cache.py` | In-memory current-job episode store. |
| `repeat_offender_policy.py` | Count, rank, feasibility precheck, and decision construction. |

Introduce a subpackage only if this grows beyond a few files or needs multiple
policy variants.

## Tests

Minimum coverage:

```text
config parses CLI/YAML and passes settings into FactAgentManager
FactAgentManager passes session/history args to nvrx-fact-agent
cycle_failed ACK means queued, not complete
TCPStore only carries attributor_id and done_count
leaf agents never query FACT history
store-host queries history only for current FACT faulty_nodes
broad current suspect set is audit-only and does not update hot cache
history query failure/timeout makes get_avoid_nodes return empty avoid_nodes
episode_id is required for historical dedupe
repeat_count >= 2 produces one avoid candidate
top candidate infeasible means no avoids for MVP
get_avoid_nodes output is ignored when missing/malformed/late
result JSONL records include terminal local failures before done_count
dmesg artifact is one shared gRPC-written file per cycle
empty raw dmesg queues no dmesg artifact chunk
```

## Open Items

1. Verify production-like FACT node-history rows expose `episode_id =
   <job_id>_<cycle_id>` or an equivalent cycle-distinct id.
2. Define the auth-file format consumed by `history_client.py`.
3. Decide whether the long-term FACT-owned history API is a library or an
   Attribution Service endpoint.
4. Have FACT review the built-in dmesg prefilter, especially missing
   `mlx5*`/HCA port-flap and network/fabric cases.
5. Validate FACT Attribution surge capacity for 5k-20k near-simultaneous POSTs.
6. Decide whether to pre-create cycle-scoped attributors, after confirming FACT
   tolerates idle or long-lived attributors.
7. Revisit `done_count` fan-in at 5k-20k nodes. Alternatives are fixed-deadline
   GET, sharded counters, or hierarchical completion.
8. Confirm whether FACT GET is strongly consistent with completed observation
   POSTs or needs bounded retry/ready signaling.
9. Confirm result JSONL size and gRPC/log-funnel limits for large FACT
    responses.
10. Define production FACT credential delivery, refresh, and redaction.
11. Define a stable physical node id beyond scheduler hostname.
12. Confirm FACT timestamp contract for the node-history `event_time`; avoid
    timestamp-window dedupe until then.
