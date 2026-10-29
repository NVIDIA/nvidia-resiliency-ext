# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The verdict says which model produced it, not just which route."""

import pytest

pytest.importorskip("pydantic")
backend = pytest.importorskip("nvidia_resiliency_ext.services.attrsvc.restart_agent_backend")
_enriched_source = backend._enriched_source


class _Result:
    def __init__(self, route_id, model):
        self.route_id = route_id
        self.model = model


def test_source_names_the_model_behind_a_generic_route():
    # A route built from the environment is always called "nvrx-default",
    # which on its own identifies nothing.
    src = _enriched_source(_Result("nvrx-default", "us/azure/openai/eccn-gpt-5.4"))

    assert src == "l1_enriched:nvrx-default (us/azure/openai/eccn-gpt-5.4)"


def test_source_does_not_repeat_a_route_named_after_its_model():
    src = _enriched_source(_Result("eccn-gpt-5.4", "eccn-gpt-5.4"))

    assert src == "l1_enriched:eccn-gpt-5.4"


def test_source_falls_back_to_the_route_when_no_model_is_reported():
    src = _enriched_source(_Result("nvrx-default", None))

    assert src == "l1_enriched:nvrx-default"
