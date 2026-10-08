"""Runtime paths that may differ between training machines."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_env_path(name: str, default: str | Path) -> Path:
    """Resolve an optional environment path relative to the repository root."""
    value = os.environ.get(name)
    path = Path(value).expanduser() if value else Path(default).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def get_mano_data_dir() -> Path:
    """Return the licensed MANO asset directory for this machine."""
    return resolve_env_path(
        "HAND3R_MANO_DIR",
        REPO_ROOT / "weights" / "mano",
    )


def get_mano_mean_params_path() -> Path:
    return get_mano_data_dir() / "mano_mean_params.npz"
