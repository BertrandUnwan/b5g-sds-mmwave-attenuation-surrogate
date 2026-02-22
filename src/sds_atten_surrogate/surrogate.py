from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Mapping, Sequence, Union

import numpy as np
import torch

from .config import SurrogateConfig
from .features import FEATURE_VARS
from .io import load_artifacts, resolve_model_dir
from .model import ShallowMLP
from .utils import expm1_safe, select_device


Number = Union[int, float]


class Surrogate:
    """Deployment wrapper around the trained streaming MLP attenuation surrogate.

    Public API:
        - predict_one(feature_dict) -> float
        - predict_many(list_of_dicts) -> list[float]
        - predict_matrix(X) -> np.ndarray
        - predict_dataframe(df) -> pandas.Series | np.ndarray

    Notes:
        - Requires all FEATURE_VARS for prediction.
        - Extra keys / columns are ignored (engineer-friendly).
    """

    def __init__(self, config: SurrogateConfig = SurrogateConfig()):
        self.config = config
        self.device = select_device(config.device)

        # Resolve model directory (external or default)
        # Use getattr so older configs without `model_dir` still work.
        model_dir = resolve_model_dir(getattr(self.config, "model_dir", None))

        # Load scaler + state_dict via centralized IO layer
        self.scaler, state = load_artifacts(
            model_dir=model_dir,
            freq_ghz=int(self.config.freq_ghz),
            tag=str(self.config.checkpoint_tag),
            device=self.device,
        )

        self.model = ShallowMLP(input_dim=len(FEATURE_VARS))
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def with_freq(self, freq_ghz: int) -> "Surrogate":
        """Return a new Surrogate instance pointing at the other frequency artifacts."""
        new_cfg = replace(self.config, freq_ghz=int(freq_ghz))  # type: ignore
        return Surrogate(new_cfg)

    def predict_one(self, features: Dict[str, Number]) -> float:
        """Predict attenuation (dB/km) for a single feature dict."""
        preds = self.predict_many([features])
        return float(preds[0])

    def predict_many(self, rows: Sequence[Mapping[str, Number]], batch_size: int = 4096) -> List[float]:
        """Predict attenuation (dB/km) for multiple feature dicts."""
        if len(rows) == 0:
            return []

        X = self._rows_to_matrix(rows)
        Xs = self.scaler.transform(X).astype(np.float32)

        preds_log: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, Xs.shape[0], batch_size):
                xb = torch.tensor(
                    Xs[start : start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                yb = self.model(xb).detach().cpu().numpy().reshape(-1)
                preds_log.append(yb)

        yhat_log = np.concatenate(preds_log, axis=0)
        yhat = expm1_safe(yhat_log)
        return [float(v) for v in yhat]

    def predict_matrix(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Predict attenuation (dB/km) from a feature matrix.

        X must be shape (N, len(FEATURE_VARS)) in correct feature order.
        Returns attenuation (dB/km) as np.ndarray shape (N,)
        """
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2 or X.shape[1] != len(FEATURE_VARS):
            raise ValueError(f"X must be shape (N, {len(FEATURE_VARS)}). Got {X.shape}.")

        Xs = self.scaler.transform(X).astype(np.float32)

        preds_log: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, Xs.shape[0], batch_size):
                xb = torch.tensor(
                    Xs[start : start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                yb = self.model(xb).detach().cpu().numpy().reshape(-1)
                preds_log.append(yb)

        yhat_log = np.concatenate(preds_log, axis=0)
        return expm1_safe(yhat_log)

    def predict_dataframe(
        self,
        df,
        batch_size: int = 4096,
        return_series: bool = True,
        output_col: str | None = None,
    ):
        """Predict attenuation (dB/km) from a pandas DataFrame.

        Requirements:
            - df must contain all FEATURE_VARS columns.
            - extra columns are ignored.

        Returns:
            - If return_series=True (default): pandas.Series aligned to df.index.
            - Else: np.ndarray shape (N,).
        """
        try:
            import pandas as pd  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("predict_dataframe requires pandas. Install it via `pip install pandas`.") from e

        if df is None:
            raise ValueError("df cannot be None")

        missing = [c for c in FEATURE_VARS if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in df: {missing}")

        X = df.loc[:, FEATURE_VARS].to_numpy(dtype=np.float32, copy=False)
        yhat = self.predict_matrix(X, batch_size=batch_size)

        if not return_series:
            return yhat

        name = output_col if output_col is not None else f"ATTEN_{int(self.config.freq_ghz)}_DB_PER_KM"
        return pd.Series(yhat, index=df.index, name=name)

    def _rows_to_matrix(self, rows: Sequence[Mapping[str, Number]]) -> np.ndarray:
        """Convert list of feature dicts to a matrix aligned with FEATURE_VARS.

        Requires all FEATURE_VARS to exist; extra keys are ignored.
        """
        required = set(FEATURE_VARS)
        X = np.zeros((len(rows), len(FEATURE_VARS)), dtype=np.float32)

        for i, r in enumerate(rows):
            keys = set(r.keys())
            missing = required - keys
            if missing:
                raise KeyError(f"Row {i}: missing features {sorted(missing)}")

            X[i, :] = np.array([float(r[name]) for name in FEATURE_VARS], dtype=np.float32)

        return X