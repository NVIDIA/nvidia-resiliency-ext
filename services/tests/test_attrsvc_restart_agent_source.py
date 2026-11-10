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


def test_healthz_reports_which_model_answers(tmp_path):
    """A rollout must be identifiable without shell access to the host.

    Asserted against the response ``status()`` actually returns, not against a
    route list rebuilt from configuration. Rebuilding one here would still pass
    if the field were dropped from the payload, which is the only place an
    operator can see it.
    """
    import asyncio

    from nvidia_resiliency_ext.attribution.restart_agent import build_restart_agent_runtime
    from nvidia_resiliency_ext.services.attrsvc.config import Settings
    from nvidia_resiliency_ext.services.attrsvc.restart_agent_backend import (
        RestartAgentServiceBackend,
    )
    from nvidia_resiliency_ext.services.attrsvc.restart_agent_config import (
        restart_agent_config_from_settings,
    )

    key = tmp_path / "api.key"
    key.write_text("dummy")
    settings = Settings(
        ANALYSIS_BACKEND="lib",
        ALLOWED_ROOT=str(tmp_path),
        LLM_MODEL="us/azure/openai/eccn-gpt-5.4",
    )
    config = restart_agent_config_from_settings(settings, environ={"LLM_API_KEY_FILE": str(key)})
    backend = RestartAgentServiceBackend(
        allowed_root=str(tmp_path),
        runtime=build_restart_agent_runtime(config),
        config=config,
    )
    try:
        status = asyncio.run(backend.status())
    finally:
        backend.shutdown()

    routes = status["restart_agent"]["model_routes"]
    assert routes, "an enriched config must advertise at least one route"
    assert routes[0]["model"] == "us/azure/openai/eccn-gpt-5.4"
    # The route id alone would not have identified it.
    assert routes[0]["route_id"] == "nvrx-default"

    # The same endpoint is where alerting destinations are reported, and it is
    # equally unreadable from outside the host.
    slack = status["slack"]
    assert set(slack) >= {"enabled", "channels", "notify_actions", "counters"}
    assert isinstance(slack["channels"], list)


def test_the_public_status_map_only_emits_public_vocabulary():
    """Internal status records how an attempt ran; public status is the request
    lifecycle a client sees. They are not interchangeable even where the strings
    coincide - an analysis that failed is a request that completed - so a map
    that mixed the two constants read like a bug to a reviewer."""
    from nvidia_resiliency_ext.attribution.orchestration.client_response import (
        _ATTRSVC_LIFECYCLE_STATUSES,
    )
    from nvidia_resiliency_ext.services.attrsvc.restart_agent_backend import (
        _PUBLIC_COMPLETED,
        _PUBLIC_IN_FLIGHT,
        _PUBLIC_PENDING,
    )

    emitted = {_PUBLIC_PENDING, _PUBLIC_IN_FLIGHT, _PUBLIC_COMPLETED}

    # What this service emits is exactly what the client validates against.
    # These live in different modules, so nothing but a test keeps them aligned.
    assert emitted == set(_ATTRSVC_LIFECYCLE_STATUSES)
