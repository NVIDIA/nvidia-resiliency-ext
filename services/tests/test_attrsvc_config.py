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


# ─── writing logs where a deployment can read them ───


@pytest.fixture
def _restore_root_handlers():
    """Keep a test's handlers from leaking into the rest of the session."""
    import logging

    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    yield
    root.handlers[:] = saved
    root.setLevel(saved_level)


def test_configure_logging_writes_to_the_configured_file(
    monkeypatch, tmp_path, _restore_root_handlers
):
    """A container's stderr is not readable from outside; a mounted file is."""
    import logging

    log_file = tmp_path / "nested" / "attrsvc.log"
    monkeypatch.setenv("NVRX_ATTRSVC_LOG_FILE", str(log_file))

    attrsvc_config.configure_logging("INFO")
    logging.getLogger("test.attrsvc").info("hello from the service")

    for handler in logging.getLogger().handlers:
        handler.flush()
    # The parent directory is created rather than required to exist.
    assert "hello from the service" in log_file.read_text()


def test_configure_logging_does_not_duplicate_the_file_handler(
    monkeypatch, tmp_path, _restore_root_handlers
):
    """It runs twice -- for the startup window, then once Settings is built.

    ``addHandler`` is not idempotent the way ``basicConfig`` is, so a second
    call used to attach a second handler and write every record twice.
    """
    import logging

    log_file = tmp_path / "attrsvc.log"
    monkeypatch.setenv("NVRX_ATTRSVC_LOG_FILE", str(log_file))

    attrsvc_config.configure_logging("INFO")
    attrsvc_config.configure_logging("INFO")
    logging.getLogger("test.attrsvc").info("said once")

    for handler in logging.getLogger().handlers:
        handler.flush()
    assert log_file.read_text().count("said once") == 1


def test_configure_logging_survives_an_unwritable_log_file(
    monkeypatch, tmp_path, _restore_root_handlers
):
    """A bad mount must not stop the service from starting."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    bad_path = blocker / "attrsvc.log"
    monkeypatch.setenv("NVRX_ATTRSVC_LOG_FILE", str(bad_path))

    attrsvc_config.configure_logging("INFO")

    # Checked by path rather than by type: pytest attaches a FileHandler of its
    # own on /dev/null, so "no FileHandler at all" would never hold here.
    assert not attrsvc_config._has_file_handler(str(bad_path))
