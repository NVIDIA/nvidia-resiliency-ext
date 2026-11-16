# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Everything the model is told about a field must agree with what is required.

Two texts reach the model: the system prompt and the response schema's own
``description`` strings. Both said ``category_selection`` was optional while
``L1ResponseContract`` listed it among the required top-level fields and left
``optional_top_level_fields`` empty. A model that believed either produced a
reply the validator rejected outright, so usable evidence was discarded and the
round spent on a repair request.

Nothing connected the three places, which is why the contradiction survived
review and a release backport. These tests connect them: the contract is the
source of truth, and no model-visible text may describe anything it requires as
omissible.
"""

from __future__ import annotations

import re

import pytest

from nvidia_resiliency_ext.attribution.restart_agent.l1.prompts import SYSTEM_PROMPT
from nvidia_resiliency_ext.attribution.restart_agent.l1.response_contract import (
    L1_RESPONSE_CONTRACT,
    model_response_schema,
)
from nvidia_resiliency_ext.attribution.restart_agent.l1.validation import (
    model_evidence_contract_errors,
)

# Sentences granting permission to leave something out. Deliberately not
# matching "without": a required field is literally named
# retry_outlook_without_workload_change, and matching it produced false
# positives on three unrelated sentences.
_PERMISSIVE_SENTENCE = re.compile(
    r"[^.]*\b(?:omitting|omitted|may omit|MAY include|is optional|are optional|"
    r"block is optional|optional evidence signal)\b[^.]*\.",
    re.IGNORECASE,
)


def _unwrapped(text: str) -> str:
    """Collapse wrapped lines so a sentence is one string.

    Both texts are hand-wrapped source literals, so a permission phrase, the
    field it names, and the closing period routinely land on three different
    physical lines. Matching per-line let exactly that shape through.
    """
    return " ".join(text.split())


def _schema_required_descriptions() -> dict[str, str]:
    """Each required property's own ``description``, keyed by field name.

    A schema description is attached to its property rather than naming the
    field in prose, so the prompt's sentence-and-field-name matching cannot
    reach it. Position supplies the field here instead.
    """
    schema = model_response_schema()
    required = set(schema.get("required", ()))
    return {
        name: node["description"]
        for name, node in schema.get("properties", {}).items()
        if name in required and isinstance(node, dict) and isinstance(node.get("description"), str)
    }


def _permission_offenders(text: str) -> dict[str, str]:
    """Permissive sentences in prose that name a required field."""
    return {
        field: sentence.strip()
        for sentence in _PERMISSIVE_SENTENCE.findall(_unwrapped(text))
        for field in L1_RESPONSE_CONTRACT.top_level_fields
        if field in sentence
    }


# ---------------------------------------------------------------------------
# The contract is the premise
# ---------------------------------------------------------------------------


def test_category_selection_is_required_by_the_contract():
    """If the field is ever made genuinely optional, this fails and both texts
    should be relaxed with it."""

    assert "category_selection" in L1_RESPONSE_CONTRACT.top_level_fields
    assert "category_selection" not in L1_RESPONSE_CONTRACT.optional_top_level_fields
    assert "category_selection" in model_response_schema()["required"]


# ---------------------------------------------------------------------------
# No model-visible text may contradict it
# ---------------------------------------------------------------------------


def test_the_system_prompt_does_not_permit_omitting_a_required_field():
    """Checked in the general form, so a field added later cannot repeat this."""

    offenders = _permission_offenders(SYSTEM_PROMPT)
    assert not offenders, f"system prompt permits omitting required field(s): {offenders}"


@pytest.mark.parametrize("field", sorted(_schema_required_descriptions()))
def test_a_required_property_does_not_describe_itself_as_optional(field):
    """The schema reaches the model too, and carried the same contradiction:
    category_selection sat in ``required`` while its description read
    "This block is optional"."""

    description = _schema_required_descriptions()[field]
    match = _PERMISSIVE_SENTENCE.search(_unwrapped(description))
    assert not match, f"{field} is required but its description says: {match.group(0).strip()!r}"


def test_both_texts_point_at_the_placeholder_instead():
    """A model told to always emit the block needs somewhere to put 'no match'."""

    prompt = _unwrapped(SYSTEM_PROMPT)
    assert "category_id=0" in prompt
    assert re.search(r"never omit category_selection", prompt, re.IGNORECASE)

    schema_description = model_response_schema()["properties"]["category_selection"]["description"]
    assert "category_id=0" in schema_description
    assert re.search(r"always include", schema_description, re.IGNORECASE)


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wording",
    [
        pytest.param(
            "- Omitting the category_selection field entirely is a valid, "
            "contract-compliant response.",
            id="single_line",
        ),
        pytest.param(
            "- After the required fields above, you MAY include one additional "
            "top-level JSON field\n  named category_selection with three keys: "
            "category_id (integer, 0-N where N is the taxonomy size),\n  "
            "category_confidence (integer 0-100).",
            id="wrapped_across_lines",
        ),
    ],
)
def test_the_guard_catches_the_prose_wordings_that_were_removed(wording):
    """A check that silently matches nothing is worse than no check.

    The wrapped case is not hypothetical: it is the real removed sentence, and
    a per-line version of this guard let it through while appearing to pass.
    """

    assert _permission_offenders(wording), f"guard missed: {wording!r}"


def test_the_guard_catches_the_removed_schema_wording():
    """The description form, which names no field and so needs the positional
    check rather than the prose one."""

    removed = (
        "Optional evidence signal: which curated category best fits the primary "
        "failure. category_id=0 means no curated category matches. This block is "
        "optional and does not itself control STOP/RESTART - L4 policy decides."
    )
    assert _PERMISSIVE_SENTENCE.search(_unwrapped(removed))


# ---------------------------------------------------------------------------
# Concrete anchor
# ---------------------------------------------------------------------------


def test_a_reply_without_category_selection_is_rejected():
    """The behaviour the old wording invited, as the anchor for the assertions
    above, which are about text rather than effect."""

    payload = {name: None for name in L1_RESPONSE_CONTRACT.top_level_fields}
    payload.pop("category_selection")

    errors = model_evidence_contract_errors(payload)

    assert any("missing fields: category_selection" in error for error in errors)
