from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

FreqGHz = Literal[28, 38]

@dataclass(frozen=True)
class SurrogateConfig:
    """
    Configuration for loading the trained MLP attenuation surrogate.
    """
    freq_ghz: FreqGHz = 38

    artifacts_dir: Path = Path("models/mlp_streaming")

    model_dir: str | None = None

    # IMPORTANT: this must match your filenames exactly: "..._checkpoint_2021-12.pt"
    # So tag should be "2021-12"
    checkpoint_tag: str = "2021-12"

    # Optional override: "cpu", "mps", "cuda"
    device: Optional[str] = None

    def resolved_artifacts_dir(self) -> Path:
        """
        Returns an absolute, expanded path to the artifacts directory.
        This allows both relative and absolute paths to work safely.
        """
        p = Path(self.artifacts_dir).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p).resolve()

    def model_path(self) -> Path:
        base = self.resolved_artifacts_dir()
        return base / f"mlp_{self.freq_ghz}ghz_checkpoint_{self.checkpoint_tag}.pt"

    def scaler_path(self) -> Path:
        base = self.resolved_artifacts_dir()
        return base / f"scaler_{self.freq_ghz}ghz_checkpoint_{self.checkpoint_tag}.joblib"