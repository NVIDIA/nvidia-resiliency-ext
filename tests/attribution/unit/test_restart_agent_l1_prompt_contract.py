# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The prompt and the response contract have to agree on what is required.

They drifted once: the prompt told the model that "omitting the
category_selection field entirely is a valid, contract-compliant response"
while ``L1ResponseContract`` listed ``category_selection`` among the required
top-level fields and left ``optional_top_level_fields`` empty. A model that
followed the instruction produced a reply the validator rejected outright, so
otherwise usable evidence was thrown away and the round was spent on a repair
request.

Nothing connected the two files, which is why the contradiction survived. These
tests connect them: the contract is the source of truth, and the prompt may not
describe anything it requires as omissible.
"""

from __future__ import annotations

import re

from nvidia_resiliency_ext.attribution.restart_agent.l1.prompts import SYSTEM_PROMPT
from nvidia_resiliency_ext.attribution.restart_agent.l1.response_contract import (
    L1_RESPONSE_CONTRACT,
)
from nvidia_resiliency_ext.attribution.restart_agent.l1.validation import (
    model_evidence_contract_errors,
)


def test_category_selection_is_required_by_the_contract():
    """Pins the premise the prompt wording depends on. If the field is ever made
    genuinely optional, this fails and the prompt should be relaxed with it."""

    assert "category_selection" in L1_RESPONSE_CONTRACT.top_level_fields
    assert "category_selection" not in L1_RESPONSE_CONTRACT.optional_top_level_fields


def test_the_prompt_never_says_a_required_field_may_be_omitted():
    """The general form of the defect, so a future field cannot repeat it.

    Sentences that grant permission - "omitting X", "you MAY include X", "X is
    optional" - are checked against the contract's required set rather than
    against the one wording that went wrong. Deliberately not matching on
    "without": a required field is literally named
    retry_outlook_without_workload_change.
    """

    permissive = re.findall(
        r"[^.\n]*\b(?:omitting|may omit|MAY include|is optional|are optional)\b[^.\n]*\.",
        SYSTEM_PROMPT,
        flags=re.IGNORECASE,
    )
    offenders = {
        field: " ".join(sentence.split())
        for sentence in permissive
        for field in L1_RESPONSE_CONTRACT.top_level_fields
        if field in sentence
    }
    assert not offenders, f"prompt permits omitting required field(s): {offenders}"


def test_the_prompt_asks_for_the_placeholder_rather_than_an_absent_field():
    """A model told to always emit the block needs somewhere to put 'no match'."""

    assert "category_id=0" in SYSTEM_PROMPT
    assert re.search(r"never omit category_selection", SYSTEM_PROMPT, flags=re.IGNORECASE)


def test_a_reply_without_category_selection_is_rejected():
    """The behaviour the old wording invited. Kept as the concrete anchor for the
    two assertions above, which are about text rather than effect."""

    payload = {name: None for name in L1_RESPONSE_CONTRACT.top_level_fields}
    payload.pop("category_selection")

    errors = model_evidence_contract_errors(payload)

    assert any("missing fields: category_selection" in error for error in errors)
