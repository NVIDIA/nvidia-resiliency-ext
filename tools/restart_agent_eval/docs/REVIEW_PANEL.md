# Eval Document Review Panel

## Panel

Use this panel for eval architecture and technical-document review:

| Reviewer | Model route |
| --- | --- |
| GPT | `us/azure/openai/eccn-gpt-5.5` |
| Claude | `us/azure/anthropic/eccn-claude-opus-4-8` |
| Gemini | `us/gcp/google/eccn-gemini-3.5-flash` |

Qwen and Nemotron are analysis candidates, not default design reviewers.

## Scope

Canonical eval documents are:

- `DESIGN.md`
- `REQUIREMENTS.md`
- `SCHEMA.md`
- `PANEL_SUMMARY.md`
- `TUNING.md`
- `STABILITY.md`
- `STATUS.md`

[eval/README.md](../eval/README.md) is navigation, not a normative contract. The
product-side integration boundary is canonical in
[EVALUATION.md](../../../docs/design/attribution/restart_agent/EVALUATION.md).

Product runtime documents may be included as context, but the panel must review
the eval target document rather than reopening product architecture implicitly.

## Workflow

Review one document at a time:

1. Human and Codex review the document first.
2. Update it until the human reviewer approves sending it externally.
3. Send the target and the smallest useful context packet to all panel members.
4. Discuss findings with the human reviewer before editing.
5. Apply accepted findings and commit the resulting document version.

## Context Packets

Always include `DESIGN.md`. Add:

- `REQUIREMENTS.md` for use cases, metrics, or acceptance criteria;
- `SCHEMA.md` for label, run, score, or artifact contracts;
- `PANEL_SUMMARY.md` for human/model comparison presentation;
- the product design or schema only for a cross-repository contract question.

## Prompt

Ask for concise findings covering correctness, missing requirements, ownership
ambiguity, production/eval divergence, and implementability. Separate required
changes from optional rationale. Do not ask the panel to rewrite the document.
