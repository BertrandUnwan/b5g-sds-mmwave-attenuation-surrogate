from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import joblib
import torch


# -----------------------------
# Naming + path resolution
# -----------------------------

def _artifact_names(freq_ghz: int, tag: str) -> tuple[str, str]:
    """Return (checkpoint_filename, scaler_filename) for a frequency + tag."""
    ckpt = f"mlp_{freq_ghz}ghz_checkpoint_{tag}.pt"
    scaler = f"scaler_{freq_ghz}ghz_checkpoint_{tag}.joblib"
    return ckpt, scaler


def resolve_model_dir(model_dir: Optional[Union[str, Path]]) -> Path:
    """Resolve the directory that holds MLP checkpoints + scalers.

    If `model_dir` is None, we default to `./models/mlp_streaming` (relative to the
    current working directory). For fully packaged assets later, we can upgrade this
    to use `importlib.resources`.
    """
    if model_dir is None:
        return Path("models") / "mlp_streaming"
    return Path(model_dir)


def assert_artifacts_exist(model_dir: Path, freq_ghz: int, tag: str) -> None:
    """Raise a friendly error if required artifacts are missing."""
    ckpt, scaler = _artifact_names(freq_ghz, tag)
    ckpt_fp = model_dir / ckpt
    scaler_fp = model_dir / scaler

    missing: list[str] = []
    if not ckpt_fp.exists():
        missing.append(str(ckpt_fp))
    if not scaler_fp.exists():
        missing.append(str(scaler_fp))

    if missing:
        msg = (
            "Missing required model artifacts. Expected files at:\n"
            + "\n".join(f" - {p}" for p in missing)
            + "\n\n"
            + "Tip: confirm MODEL_DIR points to the folder that contains files like\n"
            + f"  mlp_{freq_ghz}ghz_checkpoint_{tag}.pt\n"
            + f"  scaler_{freq_ghz}ghz_checkpoint_{tag}.joblib\n"
        )
        raise FileNotFoundError(msg)


# -----------------------------
# Loaders (with caching)
# -----------------------------

@lru_cache(maxsize=16)
def load_scaler(model_dir: Path, freq_ghz: int, tag: str):
    """Load (and cache) the sklearn StandardScaler for the given frequency."""
    _, scaler_name = _artifact_names(freq_ghz, tag)
    scaler_fp = model_dir / scaler_name
    return joblib.load(scaler_fp)


@lru_cache(maxsize=16)
def load_checkpoint_state(
    model_dir: Path,
    freq_ghz: int,
    tag: str,
    device: Union[str, torch.device] = "cpu",
):
    """Load (and cache) the torch state_dict for the given frequency."""
    ckpt_name, _ = _artifact_names(freq_ghz, tag)
    ckpt_fp = model_dir / ckpt_name
    return torch.load(ckpt_fp, map_location=device)


def load_artifacts(
    model_dir: Optional[Union[str, Path]],
    freq_ghz: int,
    tag: str,
    device: Union[str, torch.device] = "cpu",
):
    """Convenience helper: resolve dir, verify, then load scaler + state."""
    md = resolve_model_dir(model_dir)
    assert_artifacts_exist(md, freq_ghz=freq_ghz, tag=tag)
    scaler = load_scaler(md, freq_ghz=freq_ghz, tag=tag)
    state = load_checkpoint_state(md, freq_ghz=freq_ghz, tag=tag, device=device)
    return scaler, state