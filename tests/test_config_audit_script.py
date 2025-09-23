"""Tests for the Hydra configuration auditor script."""

from pathlib import Path

from scripts.config_audit import CONFIG_ROOT, audit


def _default_targets() -> list[Path]:
    experiment_cfgs = sorted((CONFIG_ROOT / "experiment").glob("*.yaml"))
    return [CONFIG_ROOT / "train.yaml", CONFIG_ROOT / "eval.yaml", *experiment_cfgs]


def test_config_audit_defaults(capsys) -> None:
    """Ensure the auditor reports success for the project configs."""
    targets = _default_targets()
    assert targets, "Expected at least the base train/eval configs to exist"

    exit_code = audit(targets)
    captured = capsys.readouterr()

    assert exit_code == 0, f"Audit reported issues:\n{captured.out.strip()}"
    assert "resolved" in captured.out.lower()
