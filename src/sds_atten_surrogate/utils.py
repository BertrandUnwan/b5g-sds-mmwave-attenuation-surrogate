from __future__ import annotations

import numpy as np
import torch


def select_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def expm1_safe(x: np.ndarray) -> np.ndarray:
    # Helps avoid inf if x is huge, though your model should not produce extreme values.
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -50.0, 50.0)
    return np.expm1(x).astype(np.float64)