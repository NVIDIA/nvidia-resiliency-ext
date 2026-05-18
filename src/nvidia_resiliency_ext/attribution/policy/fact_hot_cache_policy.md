# NVRx FACT Hot-Cache Repeat-Offender Policy

## Purpose

NVRx needs a conservative way to avoid reusing nodes that repeatedly appear in
FACT node attribution. FACT history should be the durable source of truth.
NVRx should keep only a small hot cache for evidence and retry feedback that is
needed in the restart hot path but may not yet be visible in FACT history.

```text
decision evidence = FACT historical evidence + NVRx unreconciled hot-cache evidence
```

This is not a replacement for hard health or liveness decisions. A node that
fails a Node Health Check, disappears from rendezvous, kernel-panics, loses
power, or cannot rejoin the runtime is a hard exclusion. FACT attribution is a
weaker node-suspicion signal that can escalate only when repeated, recent,
narrow evidence survives placement checks.

## Policy Boundary

| Input | Meaning | Policy |
| --- | --- | --- |
| Node Health Check / liveness failure | Node cannot safely participate now. | Hard exclude; do not downgrade for quorum. If quorum breaks, the in-job retry is infeasible. |
| Current FACT attribution | This failed cycle implicated one or more nodes. | Add hot-cache evidence; compute suspect/avoid/exclude. |
| FACT historical records | Durable prior node failures produced by FACT PM. | Query as repeat-offender prior. |
| Successful retry feedback | Node participated and the retry succeeded. | Negative evidence for that node/family. |
| Successful retry that avoided the node | Job progressed without observing the node. | Not negative evidence for that node. |

FACT does not produce job-level `STOP` or `RESTART` recommendations here. It
identifies suspect nodes. Placement actions remain owned by the NVRx restart
orchestrator.

## Why A Hot Cache Exists

NVRx makes pseudo-real-time restart decisions. FACT PM / FACT history is
asynchronous. If cycles are long, FACT may already know about the previous
cycle when NVRx needs the next decision. If failures are back-to-back, FACT may
not have caught up.

The hot cache is therefore an overlay, not a competing durable ledger:

1. Store current-cycle FACT attribution locally after the FACT GET.
2. Query FACT history over the configured lookback window.
3. Reconcile hot-cache events that are already visible in FACT.
4. Score FACT history plus unreconciled hot-cache events.
5. Write retry outcome feedback back into the hot cache.

Once FACT catches up, reconciled hot-cache events are retained only for audit
and are no longer counted as local overlay evidence.

## Node Identity And Scope

Repeat evidence should follow the physical node, not just a mutable hostname.
Use the best available stable identity:

```text
node_id = canonical(logical_node_name, physical_node_uid, node_identity_epoch)
```

`logical_node_name` is the scheduler-visible name. `physical_node_uid` should
come from the most stable inventory source available: asset tag, FRU serial,
BMC identity, provider instance id, scheduler inventory record, accelerator
serial, or NIC serial. `node_identity_epoch` changes when an operator cannot
prove that the same physical node remains behind the logical name.

Initial implementation may use logical node name only, but then promotion to
cross-job exclusion should be conservative. If physical identity is unknown,
limit action to job-scoped suspect/avoid until identity is established.

## FACT History Query

NVRx should query FACT Elasticsearch directly. FACT PM is the producer of the
historical records; NVRx does not need FACT PM as a synchronous service in the
restart path.

Default history source:

```text
index = df-aidot-fact-node-*
lookback = 2 weeks
cluster = current cluster
s_node in allocated nodes
```

Useful fields:

```text
s_node
s_id
s_symptoms
s_xid
s_outcome
s_cluster
ts_slurmdb_end_time
```

If the node index is unavailable, fall back to `df-aidot-fact-job-*` and filter
to records with non-empty bad-node fields and infrastructure-style outcomes.
Rows such as CPU `Fail User` jobs with empty bad-node fields are not useful
node-flakiness priors.

FACT ES URL, index pattern, lookback, and auth material must be configurable.
If the query fails, policy should fail open to hot-cache-only decisions.

## Hot Cache Contents

For MVP, use a bounded JSON file in the job log directory:

```text
nvrx-fact-hot-cache.json
```

Current-cycle event fields:

```text
job_id
cycle_index
attempt_id
node
node_id
failure_family
raw_symptoms
attributor_id
observation_ids
fact_visible = false
timestamp
```

Retry feedback fields:

```text
cycle_index
attempt_id
node
node_id
node_in_retry
node_avoided_by_policy
node_excluded_by_policy
attempt_succeeded
useful_runtime_s
same_family_attribution_seen
node_health_status
```

The cache should be durable across launcher restart for the current job, but it
is not the cluster-wide source of truth. Keep it bounded by recent cycles and
recent evidence refs.

## Reconciliation

A hot-cache event is reconciled when FACT history contains a matching record:

```text
same attributor_id or observation_id when available
or same job/cycle id when available
or same node + same failure family/raw symptom + nearby timestamp
```

Reconciled events are not counted twice. If FACT history has the event, FACT
history wins. If not, the hot-cache event remains active so NVRx can make an
immediate retry decision.

## Failure Families

Normalize current FACT attribution and FACT history into one small taxonomy.
Keep raw symptoms and a normalizer version for audit.

| Raw signal | Family |
| --- | --- |
| XID-48, ECC DBE, uncorrectable ECC | `accelerator-memory-ecc-dbe` |
| XID-63, XID-64, row remap, page retirement | `accelerator-memory-ecc-contained` |
| XID-79, XID-95, fallen off bus, uncontained | `accelerator-xid-fatal` |
| Other XID | `accelerator-xid-ambiguous` |
| SXid, NVLink/NVSwitch/fabric GPU signal | `accelerator-fabric` |
| LustreError, ping timeout, detected conn error | `storage-or-fabric-io` |
| kernel lockup, general protection fault | `kernel-fatal` |
| OOM killer | `host-oom` |
| Unmapped node-specific signal | `UNKNOWN_FAMILY` |

`UNKNOWN_FAMILY` may create `suspect` but should not drive
`exclude-attribution` by default. Broad suspect sets usually indicate a
cascade, systemic issue, or weak evidence; preserve them for audit but do not
mass-exclude.

## Decision Policy

Actions, from weakest to strongest:

```text
reuse normally -> suspect -> avoid-for-retry -> exclude-attribution -> hard-exclude
```

Hard exclude is reserved for health/liveness and bypasses this attribution
policy. Attribution actions are feasibility-gated and may be downgraded when
capacity is tight.

Recommended MVP behavior after reconciliation:

| Evidence | Action |
| --- | --- |
| One current NVRx attribution only | `suspect` |
| Current NVRx attribution plus matching FACT history | `avoid-for-retry` if feasible |
| Two merged same-node/same-family events | `avoid-for-retry` if feasible |
| Three merged same-node/same-family events | `exclude-attribution` if feasible |
| FACT history only, no current attribution | No active placement action by default; optionally lower threshold. |
| Broad suspect set | Observe only. |
| Health/liveness failure | Hard exclude. |

Cycle number alone is not policy. A repeat in cycle 5 and cycle 11 is strong
only if it is the same node, same family, inside the suspicion window, and not
offset by successful node-local runtime. Two events separated by month-long
successful cycles should usually decay back toward suspect. Back-to-back short
cycles with the same contained ECC node are the strongest motivating case:
first event suspect, second nearby same-family event avoid if feasible, third
nearby event exclude-attribution if feasible.

Coalesce duplicate reports within an attempt:

```text
key = (attempt_id, node_id, failure_family)
```

Multiple sources for the same key produce one recurrence event. Use the maximum
confidence/source strength for that coalesced event rather than incrementing
repeat count multiple times.

Suspicion should decay with:

1. Wall-clock time since last same-family evidence.
2. Successful useful runtime on the node.
3. Successful retries that included the node.
4. Absence of the same family while the node participated.

A successful retry that avoided the node is not negative evidence for the node.
It only proves the job can progress without it.

## Placement Feasibility

NVRx uses in-job restart. It does not perform scheduler reallocation on every
retry. Replacement capacity is whatever is already inside the rendezvous
envelope:

```text
active quorum = min_nodes
total participants = max_nodes
standby capacity = max_nodes - min_nodes
```

Avoid/exclude must pass:

```text
min_nodes / quorum
rank and accelerator count
topology or rack shape
failure-domain caps
available standby capacity
max attribution exclusions per cycle/rack/domain
```

For fixed-size jobs (`--nnodes=N`), policy can mark suspects but cannot enforce
avoid/exclude without losing quorum. For elastic jobs (`--nnodes=min:max`),
nodes beyond `min_nodes` are already-participating standby candidates that
rendezvous can promote on retry.

## Enforcement Point

Do not put the repeat-offender policy directly in `ft_launcher`. The launcher
should stay a worker lifecycle integration point.

Use a small package, for example:

```text
nvidia_resiliency_ext/fault_tolerance/restart_policy/
  models.py
  symptom_family.py
  fact_history.py
  hot_cache.py
  policy.py
  store_handoff.py
```

The active enforcement hook belongs in rendezvous rank assignment. The policy
owner writes per-round decisions into TCPStore:

```text
avoid_nodes:<round>
policy_ready:<round>
```

The store host waits briefly for `policy_ready:<round>`, reads
`avoid_nodes:<round>`, and assigns active ranks from joined participants not in
the avoid set. Avoided nodes become standby candidates if placement remains
feasible. If policy is absent, late, or fails, rendezvous proceeds without
enforcing avoid nodes. Attribution policy must fail open.

Hard health/liveness exclusions remain separate: those nodes self-exclude
before joining or cannot join at all, and the job may terminate if quorum is
impossible.

## Minimal Configuration

```text
fact_history_enabled = true
fact_history_lookback = 14d
fact_history_index = df-aidot-fact-node-*
fact_history_fail_open = true

hot_cache_enabled = true
hot_cache_path = <job-log-dir>/nvrx-fact-hot-cache.json

suspect_ttl = 24h
clear_after_successful_runtime = 6h
min_repeat_count_for_avoid = 2
min_repeat_count_for_exclude_attribution = 3
unknown_family_can_exclude_attribution = false

cardinality_cap_nodes = 2
cardinality_cap_fraction = 0.05
max_attribution_exclusions_per_cycle = 1
max_attribution_exclusions_per_rack = 1
exclude_attribution_requires_placement_feasible = true
avoid_requires_placement_feasible = true
```

These are structural defaults, not universal tuning. The important properties
are: bound broad attribution, decay with useful runtime, require repeated
same-family evidence, separate health from attribution, and never let
attribution policy strand a feasible restart unless an explicit emergency
override is configured.

## Rollout

1. **Dry-run:** compute merged FACT + hot-cache decisions, log decisions, no
   placement change.
2. **Avoid-only:** avoid nodes when merged evidence crosses the avoid threshold
   and standby capacity makes this feasible.
3. **Job-scoped exclusion:** exclude repeated same-family offenders from this
   job only when placement remains feasible.
4. **FACT feedback upload:** optionally publish NVRx policy feedback so FACT
   history can replace more of the hot cache.

## Open Questions

1. What is the best reconciliation key available in production: job id, cycle
   id, attributor id, observation id, timestamp, or a combination?
2. How quickly does FACT PM normally upload NVRx-cycle attribution into FACT
   Elasticsearch?
3. Can FACT history represent avoided nodes and successful included runtime, or
   do we need a FACT-side schema extension?
4. Should FACT-history-only evidence ever cause avoid, or should it only lower
   the threshold after current NVRx attribution?
5. What duration/risk function should replace raw repeat counts for hour-long
   cycles versus month-long cycles?
