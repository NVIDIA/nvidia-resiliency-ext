# Spike: L1 Category Selection as Policy-Context Input

This document describes a spike branch (`chengsongz/decision-from-category-on-pr400`)
that layers the 38-category L1 taxonomy from PR #390 on top of PR #400's
observability/deterministic-retry-policy work. It is a discussion artifact,
not a merge candidate.

## Motivation

PR #390 introduced a 38-entry L1 category taxonomy and used it directly in L4
via a `category_confirmed_stop` decision basis. On a 79-case labeled corpus,
that approach achieved:

| target   | decision accuracy (PR #390) |
|----------|---------------------------- |
| gpt      | 100.0%                      |
| qwen397b | 97.5%                       |
| gemini   | 100.0%                      |
| nemotron | 94.9%                       |

Running the same 79 cases through PR #400 as-shipped yields:

| target   | decision accuracy (PR #400) |
|----------|---------------------------- |
| gpt      | 72.2%                       |
| qwen397b | 78.5%                       |
| gemini   | 88.6%                       |
| nemotron | 82.3%                       |

The gap is concentrated in two clusters:

1. **STOP-worthy deterministic workload errors** (b1-010 to b1-030 range):
   permission errors, config faults, checkpoint format/shape mismatches,
   missing datasets. All four models under PR #400 call these RESTART.
   The abstract cascade doesn't fire `workload_unrecoverable` because the
   LLM doesn't reliably output `retry_outlook=cannot_recover` with
   `status=established_by_current_log` for these cases.

2. **Product-policy disagreement** (b1-004..b1-009 CUDA OOM cases):
   PR #400's `cuda_oom_no_retry` context calls these STOP; PR #390 labels
   them RESTART.

Cluster (1) is a design-vs-signal issue: PR #400 has a rich policy layer but
insufficient evidence input for the deterministic-workload-error cluster.
Cluster (2) is an intentional product-policy difference and not addressable
by evidence changes.

The spike explores whether adding the L1 category selection as an *evidence
signal* to PR #400's `policy_context` machinery can close cluster (1) without
violating PR #400's design principles.

## Design principle alignment

| Principle | Handling in this spike |
|-----------|-----------------------|
| D2 (L2 non-overriding) | Category selection is a top-level *sibling* of L1 typed claims. L2 grounding still applies to primary/observation; category is not affected. Raw L1 output is preserved verbatim. |
| D8 (score-free deterministic policy) | Category confidence gate is retained (default 80). Category-driven policy context runs only when primary is grounded and cat.decision=STOP - it never causes a RESTART override. RESTART-labeled categories fall through to the existing base rule cascade. |
| D9 (declared policy context) | Category-driven context is subordinate to all deterministic classifier contexts (cuda_oom_no_retry, port_bind_confirmation_retry, rejected_iteration_retry_then_skip). Precedence is explicit and testable. |
| D14 (visible surfaces separate from root) | Category-driven context requires `primary is not None`. When only an observation is available, category has no policy authority. |

## Integration shape

Three additions to PR #400:

1. **L1 schema opt-in**: `category_selection: { category_id, category_confidence, category_rationale }` as an optional top-level field. Existing L1 responses that omit it still validate.

2. **L4 policy_context extension**: new context `l1_category_confirmed_stop`. Matches when:
   - `primary is not None` (grounded)
   - `category_selection.category_confidence >= threshold` (default 80)
   - `category_by_id(category_id).decision == "STOP"`
   Effective policy: rule=`workload_unrecoverable`, allowed_retries=0.
   Runs LAST in `_match_policy_context` - all deterministic classifier
   contexts win over it.

3. **Threshold override**: `NVRX_L1_CATEGORY_CONFIDENCE_THRESHOLD` env var
   read in `decision_pipeline` and passed via `L4PolicyInput`.

Plus ported infrastructure from PR #390:
- `l1/categories.py`: the 38-entry taxonomy
- `l1/prompts.py`: promotes taxonomy into the model prompt (~2.4 KB)
- `l1/validation.py`: 5 in-place contract repair functions
  (biconditional_unknowns, overlong_lists, invalid_evidence_supports,
  overlong_category_rationale, schema_version)

## What is *not* touched in this spike

- Base rule cascade (`_select_base_rule`) is unchanged.
- Immediate-stop gate (`_immediate_stop_qualified`) is unchanged.
- History identity, ledger accounting, job guards - all unchanged.
- Existing policy contexts (cuda_oom_no_retry, etc.) - unchanged.
- L0 assembly - not touched by this spike (a separate cascade-vs-root
  promotion fix from PR #390 was NOT ported; needs discussion whether
  PR #400's L0 rewrite already handles the same case).

## Expected empirical outcome

If this spike works as designed:
- Cluster (1) closes: deterministic-workload STOP cases with high-conf
  category picks (cats 16/18/19/22/24) trigger the new context - decision
  becomes STOP.
- Cluster (2) is unchanged: `cuda_oom_no_retry` still runs first and still
  calls CUDA OOM cases STOP (per PR #400's product policy).
- Cases with RESTART-labeled categories are unchanged: the base rule
  cascade decides, as it does today.

Verification is a rerun of the 79-case suite against this branch.

## Open questions for the design discussion

1. **Should category-driven `policy_context` be config-gated?** Currently it
   is always active. A `PolicyContextConfig.l1_category_confirmed_stop`
   entry with `enabled: true` by default would fit PR #400's shape better.

2. **Should the taxonomy `decision` field be dropped and moved to config?**
   The 38 categories each carry a decision. A more D9-aligned shape would be:
   taxonomy carries {id, name, description}; config declares which cat_ids
   are STOP-labeled and at what threshold. This is a bigger refactor but
   philosophically cleaner.

3. **Should L2 grounding gate category-driven policy_context?** Currently
   only `primary is not None` is required. A stronger form would require
   L2 to have grounded the primary. This tightens the anti-hallucination
   guardrail at the cost of some recoverable cases.

4. **Do we still need the contract-repair layer?** The 5 repairs recover
   valid model output that would otherwise be discarded. If PR #400 has an
   equivalent "L1 contract advisory" path that already handles some of
   these cases without repair, some repairs may be redundant.

5. **What is the L0 story for the b1-014-style cascade-precedes-root case?**
   PR #390 added a `_select_primary_candidate` cascade-promotion rule to L0.
   PR #400's L0 was rewritten; whether the same problem manifests and
   whether the fix should be re-added or is unnecessary needs empirical
   verification.
