# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

attrsvc_config = pytest.importorskip("nvidia_resiliency_ext.services.attrsvc.config")
parse_service_endpoint = attrsvc_config.parse_service_endpoint


def test_parse_service_endpoint_defaults_to_loopback():
    endpoint = parse_service_endpoint("")

    assert endpoint.host == "127.0.0.1"
    assert endpoint.port == 8000
    assert endpoint.display_url == "http://127.0.0.1:8000"


def test_parse_service_endpoint_accepts_explicit_bind_all_host():
    endpoint = parse_service_endpoint("", host="0.0.0.0")  # nosec B104

    assert endpoint.host == "0.0.0.0"
    assert endpoint.port == 8000


# ─── identifying what is deployed ───


def test_healthz_build_identity_reports_the_installed_version(monkeypatch):
    """A rollout has to be confirmable without shell access to the host."""
    from nvidia_resiliency_ext.services.attrsvc.app import BUILD_TAG_ENV_VAR, _build_identity

    monkeypatch.delenv(BUILD_TAG_ENV_VAR, raising=False)
    identity = _build_identity()

    assert "version" in identity
    assert identity["version"]
    # Absent rather than empty, so a consumer can test for it plainly.
    assert "tag" not in identity


def test_healthz_build_identity_includes_a_configured_tag(monkeypatch):
    # The image build stamps this; every image otherwise reports the same
    # version, because the container passes a literal to the version bypass.
    from nvidia_resiliency_ext.services.attrsvc.app import BUILD_TAG_ENV_VAR, _build_identity

    monkeypatch.setenv(BUILD_TAG_ENV_VAR, "main-dba35233")

    assert _build_identity()["tag"] == "main-dba35233"


def test_healthz_build_identity_ignores_a_blank_tag(monkeypatch):
    from nvidia_resiliency_ext.services.attrsvc.app import BUILD_TAG_ENV_VAR, _build_identity

    monkeypatch.setenv(BUILD_TAG_ENV_VAR, "   ")

    assert "tag" not in _build_identity()
