# L1 Category Selection as Policy-Context Input

This document describes the L1 category-selection signal and how it participates in the L4 policy layer.

## Motivation

The L4 policy cascade decides RESTART / STOP based on typed claims that come out of the L1 layer (`failure_domain`, `retry_outlook_without_workload_change`) and deterministic classifier signals (`FailureClassifier`). Two problem clusters remained:

1. **STOP-worthy deterministic workload errors.** Permission errors, launch-config faults, checkpoint format/shape mismatches, and missing datasets do not reliably surface as `retry_outlook = cannot_recover` with `status = established_by_current_log` from the LLM, so the abstract cascade does not fire `workload_unrecoverable`. These cases end up RESTART when they should STOP.

2. **Concrete-vs-abstract capability gap.** Empirically, LLMs are ~20 pp more accurate at picking a category from a curated taxonomy than at emitting the abstract `failure_domain` / `retry_outlook` typed claims that L4's base rule cascade consumes.

A concrete signal — "pick one of the N curated categories" — is a more reliable route to a correct L4 decision than the abstract typed claims alone for these cases.

## Design principle alignment

| Principle | Handling |
|-----------|----------|
| **D2 (L2 non-overriding).** L1 category selection is a top-level *sibling* of the typed L1 claims. L2 grounding still applies to the primary/observation; category selection is not affected by L2. Raw L1 output is preserved verbatim. |
| **D8 (score-free deterministic policy).** No confidence threshold gate. The STOP context fires when the primary is grounded and `category.decision == "STOP"`. RESTART-labeled categories fall through to the base rule cascade unless the opt-in RESTART context is enabled. |
| **D9 (declared policy context).** Both category-driven contexts (STOP and RESTART) run *after* every deterministic classifier context (`cuda_oom_no_retry`, `port_bind_confirmation_retry`, `rejected_iteration_retry_then_skip`). Precedence is explicit and testable. |
| **D14 (visible surfaces separate from root).** Category-driven contexts require `primary is not None`. When only an observation is available, the category has no policy authority. |

## Integration shape

Four additions to the L4 layer:

1. **L1 schema.** `category_selection: { category_id, category_confidence, category_rationale }` is a required top-level field on the L1 response. `category_id = 0` and `category_confidence = 0` is the sanctioned "no listed category matches" placeholder.

2. **`l1_category_confirmed_stop` policy context.** Matches when:
   - `primary is not None` (grounded)
   - `category_by_id(category_id).decision == "STOP"`

   Effective policy: `rule = workload_unrecoverable`, `allowed_retries = 0`, `policy_context_id = "l1_category_confirmed_stop"`. Runs after every deterministic classifier context. Enabled by default in `PolicyContextConfig`.

3. **`l1_category_confirmed_restart` policy context (opt-in).** Matches when:
   - `primary is not None`
   - `category_by_id(category_id).decision == "RESTART"`
   - `base_rule == workload_unrecoverable` (only overrides this specific STOP rule)
   - `history.matching_root_attempts == 0` (first-occurrence guard)

   Effective policy: `rule = workload_confirmation_retry`, `history_match_scope = ROOT_ONLY`, `allowed_retries = 1`, `policy_context_id = "l1_category_confirmed_restart"`. Runs *after* the STOP context. Disabled by default (`PolicyContextConfig.l1_category_confirmed_restart.enabled = False`).

4. **Taxonomy.** `l1/categories.json` holds the 38-entry catalog with schema `{id, name, description, decision}`. Loaded and validated at import time via `l1/categories.py`.

## What is *not* touched by this feature

- Base rule cascade (`_select_base_rule`) is unchanged.
- Immediate-stop gate (`_immediate_stop_qualified`) is unchanged.
- History identity, ledger accounting, and job guards are unchanged.
- Existing policy contexts (`cuda_oom_no_retry`, `port_bind_confirmation_retry`, `rejected_iteration_retry_then_skip`) are unchanged.
- L0 assembly is not touched.

## Recurrence and cross-cycle behavior

Both category-driven contexts are designed to be safe under recurrence, and to defer to the L4 history ledger for cross-cycle bookkeeping:

- The **STOP context** already produces `allowed_retries = 0` on first occurrence. History exhaustion cannot override it in the RESTART direction.
- The **RESTART context** fires only on first occurrence via its `history.matching_root_attempts == 0` guard. Any subsequent cycle with the same `root_fingerprint` falls through to the base rule, which combined with `selected_policy_ledger` (scope `ROOT_ONLY`, budget 1) exhausts and STOPs.

## Recovery-assessment fields

The taxonomy's `CategoryDef` intentionally does **not** carry `failure_domain` or `retry_outlook` fields. Those values are already emitted per case by the LLM in `model_recovery_assessment`, and the L4 base rule cascade reads them from there. Duplicating them on the category would introduce a drift risk without any runtime consumer.

## Observability

Both category-driven contexts populate `applied_policy_context.current_signature` with:

- `l1_category_id`
- `l1_category_name`
- `l1_category_confidence_reported`

The RESTART context additionally records `overridden_base_rule` so the audit trail shows which base rule was overridden.
