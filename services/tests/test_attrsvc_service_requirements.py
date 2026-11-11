# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the image's dependency list in step with pyproject.

``deploy/requirements.txt`` exists because the image installs the root package
with ``--no-deps``, which drops every declared dependency including the ones
attrsvc imports at runtime. Maintaining that list by hand disabled alerting
once: ``slack-sdk`` was required in pyproject, the list omitted it, and the
service started, served, analyzed and never alerted -- alerting sits behind an
optional import that is designed to succeed when the dependency is absent.

Adding a dependency to pyproject prompts nobody to think of this file, so the
omission is caught here rather than in production.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS = REPO_ROOT / "services" / "attrsvc" / "deploy" / "requirements.txt"

# Workload-side only: the training stack, GPU bindings and the rank-monitor
# gRPC transport. Everything else pyproject requires must reach the image.
WORKLOAD_PACKAGES = {
    "torch",
    "numpy",
    "nvidia-ml-py",
    "psutil",
    "packaging",
    "pyyaml",
    "defusedxml",
    "grpcio",
    "grpcio-tools",
    "protobuf",
}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _pyproject_required() -> set[str]:
    """Non-optional package names from ``[tool.poetry.dependencies]``."""
    block = PYPROJECT.read_text(encoding="utf-8").split("[tool.poetry.dependencies]", 1)[1]
    required = set()
    for line in block.split("\n[", 1)[0].splitlines():
        match = re.match(r"^(?P<name>[A-Za-z0-9_.-]+)\s*=\s*(?P<spec>.+?)\s*$", line.strip())
        if not match or "optional = true" in match.group("spec"):
            continue
        if (name := _normalize(match.group("name"))) != "python":
            required.add(name)
    return required


def _image_packages() -> set[str]:
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    return {
        _normalize(re.split(r"[\[<>=!~;]", line.strip(), maxsplit=1)[0])
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    }


def test_every_required_dependency_reaches_the_image():
    """The slack-sdk omission, generalized: a new non-optional dependency is
    either listed, denied, or silently missing from the image."""
    missing = _pyproject_required() - WORKLOAD_PACKAGES - _image_packages()
    assert not missing, (
        f"required in pyproject but absent from the image: {sorted(missing)}. "
        f"Add to services/attrsvc/deploy/requirements.txt, or to WORKLOAD_PACKAGES "
        f"here if a CPU service image must not carry them."
    )


def test_the_image_carries_nothing_workload_side():
    """The point of --no-deps. PyTorch above all: this is a CPU image."""
    leaked = _image_packages() & WORKLOAD_PACKAGES
    assert not leaked, f"workload-side packages in a CPU service image: {sorted(leaked)}"


def test_the_parse_sees_a_plausible_dependency_table():
    """A parse that silently matched nothing would make both checks vacuous."""
    required = _pyproject_required()
    assert {"fastapi", "slack-sdk", "torch"} <= required
    assert "mcp" not in required, "optional extras must not be treated as required"
