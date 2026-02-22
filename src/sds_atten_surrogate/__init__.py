"""b5g-sds-atten-surrogate

Physics-consistent surrogate model for dust-induced mmWave specific attenuation
at **28 GHz** and **38 GHz**.

Public API
----------
- `SurrogateConfig`: configuration (freq selection, artifact paths, etc.)
- `Surrogate`: predictor with convenience helpers:
  - `predict_one(feature_dict)`
  - `predict_many(list_of_dicts)`
  - `predict_matrix(matrix)`
  - `predict_dataframe(df)`

Quick start
-----------
```python
from sds_atten_surrogate import Surrogate

# Two independent models (toggle by frequency)
model_28 = Surrogate().with_freq(28)
model_38 = Surrogate().with_freq(38)

y28 = model_28.predict_one(feature_dict)
y38 = model_38.predict_one(feature_dict)

# Pandas batch inference
# df must contain all required FEATURE_VARS columns; extra columns are ignored.
pred_28 = model_28.predict_dataframe(df)
```
"""

from .config import SurrogateConfig
from .surrogate import Surrogate

__all__ = ["Surrogate", "SurrogateConfig"]

__version__ = "0.1.0"