"""Artifact-root validation and mirrored one-log layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .runtime import SYSTEM_CLOCK, Clock


@dataclass(frozen=True)
class ArtifactLayout:
    """Mirrored source, durable-gold, and disposable-run locations for one log."""

    log_root: Path | None
    gold_root: Path | None
    run_root: Path | None
    relative_log_path: Path
    run_id: str
    run_dir: Path
    gold_path: Path | None


def resolve_artifact_layout(
    *,
    log_path: Path,
    log_root: Path | None,
    gold_root: Path | None,
    run_root: Path | None,
    run_dir: Path | None,
    gold_label: Path | None,
    run_id: str | None = None,
    clock: Clock = SYSTEM_CLOCK,
) -> ArtifactLayout:
    """Resolve one case without allowing source, gold, and runs to overlap."""

    resolved_log = log_path.expanduser().resolve()
    resolved_log_root = log_root.expanduser().resolve() if log_root is not None else None
    resolved_gold_root = gold_root.expanduser().resolve() if gold_root is not None else None
    resolved_run_root = run_root.expanduser().resolve() if run_root is not None else None

    if resolved_log_root is not None:
        try:
            relative_log_path = resolved_log.relative_to(resolved_log_root)
        except ValueError as exc:
            raise SystemExit(
                f"--log must be under --log-root: log={resolved_log} root={resolved_log_root}"
            ) from exc
    elif run_dir is not None:
        relative_log_path = Path(resolved_log.name)
    else:
        raise SystemExit(
            "--log-root is required unless --run-dir is supplied; set "
            "RESTART_AGENT_EVAL_LOG_ROOT for repeated use"
        )

    if run_dir is None and resolved_run_root is None:
        raise SystemExit(
            "--run-root is required unless --run-dir is supplied; set "
            "RESTART_AGENT_EVAL_RUN_ROOT for repeated use"
        )
    if run_dir is None and resolved_gold_root is None:
        raise SystemExit(
            "--gold-root is required unless --run-dir is supplied; set "
            "RESTART_AGENT_EVAL_GOLD_ROOT for repeated use"
        )
    if resolved_gold_root is not None and resolved_log_root is None:
        raise SystemExit("--gold-root requires --log-root so the mirrored case path is stable")

    configured_roots = [
        root
        for root in (resolved_log_root, resolved_gold_root, resolved_run_root)
        if root is not None
    ]
    validate_artifact_roots(configured_roots)

    run_id = run_id or clock.now_utc().strftime("%Y%m%dT%H%M%S%fZ")
    resolved_run_dir = (
        run_dir.expanduser().resolve()
        if run_dir is not None
        else resolved_run_root / relative_log_path / run_id
    )
    resolved_gold_path = (
        gold_label.expanduser().resolve()
        if gold_label is not None
        else (
            resolved_gold_root / relative_log_path / "gold.json"
            if resolved_gold_root is not None
            else None
        )
    )
    return ArtifactLayout(
        log_root=resolved_log_root,
        gold_root=resolved_gold_root,
        run_root=resolved_run_root,
        relative_log_path=relative_log_path,
        run_id=run_id,
        run_dir=resolved_run_dir,
        gold_path=resolved_gold_path,
    )


def validate_artifact_roots(roots: list[Path]) -> None:
    """Require roots that can be managed independently without recursive overlap."""

    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise SystemExit(
                    "--log-root, --gold-root, and --run-root must be distinct, "
                    f"non-overlapping directories: {left} conflicts with {right}"
                )
