# B5G Sand and Dust Storm (SDS) Attenuation Surrogate Model

## Project Overview

This project develops a **physics-informed, machine-learning surrogate model** to predict **millimeter-wave (mmWave) signal attenuation caused by Sand and Dust Storms (SDS)** in **Beyond-5G (B5G)** terrestrial wireless systems over **Saudi Arabia**.

Instead of relying solely on closed-form analytical models or computationally expensive simulators, this work leverages **NASA MERRA-2 reanalysis datasets** to learn the relationship between:

- Atmospheric state variables
- Aerosol/dust loading indicators
- Meteorological conditions

and the resulting:

> **SDS-induced specific attenuation (dB/km)**

The final system includes:
- A trained surrogate regression model
- A reusable Python package (`sds_atten_surrogate`)
- A FastAPI inference service for deployment-ready predictions

---

## Motivation

Millimeter-wave frequencies (e.g., **28 GHz and 38 GHz**) are highly sensitive to environmental factors such as:

- Dust and sand concentration
- Particle size distribution
- Atmospheric humidity
- Wind-driven particle suspension
- Temperature and pressure conditions

While existing analytical and simulator-based approaches (e.g., ITU models, NYUSIM-based studies) provide valuable insight, they:

- Require simplifying assumptions
- Depend on particle microphysical modeling
- Are computationally expensive for repeated large-scale evaluations
- Are not easily adaptable to climatological reanalysis workflows

This project addresses these gaps by building a **data-driven surrogate model trained on real atmospheric reanalysis data**, enabling:

- Fast inference
- Scalable evaluation
- Deployment-ready API integration

---

## Geographic and Temporal Scope

- **Region:** Saudi Arabia  
- **Time span:** 2012 – 2021  
- **Spatial resolution:** MERRA-2 native lat/lon grid  
- **Temporal resolution:** Hourly  

All datasets are aligned on:

```
(time, latitude, longitude)
```

---

## Data Sources

### 1️⃣ Aerosol Data (MERRA-2)

Used to characterize dust loading and optical behavior.

Key variables include:

- Dust column mass density
- Dust surface mass concentration
- Aerosol optical thickness indicators
- Fine dust fraction

---

### 2️⃣ Meteorological Data (MERRA-2)

Used to capture environmental conditions affecting dust suspension and propagation.

Key variables include:

- 2-meter temperature (T2M)
- Surface pressure (PS)
- Sea-level pressure (SLP)
- 2-meter specific humidity (QV2M)
- 10-meter wind components (U10M, V10M)

Derived features include:

- Wind speed magnitude
- Wind direction
- Cyclical hour encoding (HOUR_SIN / HOUR_COS)
- Cyclical day-of-year encoding (DOY_SIN / DOY_COS)
- Log-transformed dust mass features

---

## Feature Engineering Philosophy

The model is **physics-guided but data-driven**.

Key design principles:

- Preserve mechanistic relationships ("more dust → more attenuation")
- Include humidity and meteorology as modulation variables
- Use log-transformed dust mass for numerical stability
- Treat attenuation as **specific attenuation (dB/km)**

The target is decoupled from path length so users can compute:

```
Total Attenuation (dB) = Specific Attenuation (dB/km) × Distance (km)
```

---

## Target Variable

The supervised learning target is:

> **SDS-induced specific attenuation (dB/km)**

This formulation:

- Aligns with propagation literature
- Enables flexible link-length scaling
- Supports network planning workflows

---

## Model Architecture

- Regression-based neural network surrogate (MLP)
- Frequency-specific models for:
  - 28 GHz
  - 38 GHz
- Trained using standardized atmospheric + aerosol features
- Checkpoints stored for reproducible inference
- Model artifacts excluded due to size. Contact author for full project demo

The surrogate is packaged as a reusable Python module:

```
src/sds_atten_surrogate/
```

---

## Inference Interfaces

The surrogate supports multiple inference APIs:

### Python API

```python
from sds_atten_surrogate import Surrogate, SurrogateConfig

cfg = SurrogateConfig(freq_ghz=28)
s = Surrogate(cfg)

prediction = s.predict_one(feature_dict)
```

Supported methods:

- `predict_one(feature_dict)`
- `predict_many(list_of_dicts)`
- `predict_numpy(X)`
- `predict_dataframe(df)`

---

## FastAPI Deployment Layer

A thin FastAPI wrapper is provided in:

```
api/main.py
```

Run locally:

```bash
uvicorn api.main:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

Available endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /predict_many`

This allows engineers to integrate attenuation prediction into external planning tools.

---

## Validation Strategy

This project does **not rely on proprietary field measurements**.

Validation is performed against:

- Peer-reviewed dust attenuation studies
- Analytical models
- Simulator-based results (e.g., NYUSIM-derived curves)
- Published mmWave attenuation magnitudes

The surrogate is validated for:

- Magnitude consistency
- Trend agreement
- Frequency scaling behavior
- Physical plausibility under extreme dust conditions

---

## System Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                    MERRA-2 Reanalysis Data                 │
│  • Aerosols (dust mass, optical depth, fine fraction)      │
│  • Meteorology (T2M, PS, SLP, QV2M, wind components)       │
│  • Hourly | 2012–2021 | Saudi Arabia                       │
└───────────────────────────────┬────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────┐
│                Data Merge & Grid Alignment                 │
│              (time, latitude, longitude)                   │
└───────────────────────────────┬────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────┐
│                   Feature Engineering                      │
│  • Dust optical indicators                                 │
│  • Humidity & pressure modulation                          │
│  • Wind speed & direction                                  │
│  • Cyclical time encoding (hour, DOY)                      │
│  • Log-transformed dust mass features                      │
└───────────────────────────────┬────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────┐
│              Physics-Informed Label Generation             │
│         SDS Specific Attenuation (dB/km)                   │
│         → 28 GHz model                                     │
│         → 38 GHz model                                     │
└───────────────────────────────┬────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────┐
│                  Labeled Dataset Artifact                  │
│        labeled_full_2012_2021_attn_28_38.nc4               │
└───────────────────────────────┬────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────┐
│                   Model Training Pipeline                  │
│  • Ridge Regression                                        │
│  • Random Forest                                           │
│  • Streaming MLP (frequency-specific)                      │
└───────────────────────────────┬────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────┐
│                       Model Artifacts                      │
│  • MLP checkpoints (.pt)                                   │
│  • Feature scalers (.joblib)                               │
│  • Training logs & metrics (.csv)                          │
└───────────────┬───────────────────────────────┬────────────┘
                │                               │
                ▼                               ▼
┌──────────────────────────────┐     ┌──────────────────────────────┐
│     Python Package Layer     │     │      FastAPI Service Layer   │
│  sds_atten_surrogate         │     │  /health                     │
│  • predict_one()             │     │  /metadata                   │
│  • predict_many()            │     │  /predict                    │
│  • predict_dataframe()       │     │  /predict_many               │
└───────────────┬──────────────┘     └───────────────┬──────────────┘
                │                                    │
                ▼                                    ▼
┌──────────────────────────────┐     ┌──────────────────────────────┐
│   Engineer Workflow Usage    │     │  External System Integration │
│  • Link budget planning      │     │  • Network planning tools    │
│  • Backhaul design           │     │  • Sensitivity analysis      │
│  • Dust event evaluation     │     │  • Deployment automation     │
└──────────────────────────────┘     └──────────────────────────────┘
```

---

## Current Status

- Environment setup: ✅  
- Data ingestion & merging: ✅  
- Feature engineering: ✅  
- Physics-informed label generation: ✅  
- Model training (28 & 38 GHz): ✅  
- Validation against literature: ✅ 
- API deployment layer: ✅  
- Scaling & cloud deployment: 🔜  

---

## Intended Use Case

A network engineer provides:

- Location (lat, lon)
- Atmospheric conditions (or timestamp)
- Frequency (28 or 38 GHz)

The system returns:

> Predicted SDS attenuation (dB/km)

This enables:

- mmWave backhaul planning
- Fixed wireless access evaluation
- Sensitivity analysis under dust events

---

## Academic Context

This project is developed as a **graduate-level capstone research prototype**, emphasizing:

- Reproducibility
- Physical interpretability
- Transparent validation
- Deployable ML architecture

The system is intentionally designed to be scalable for future research extensions.

---

## Future Work (Post-Capstone)

- Extreme-event benchmarking (visibility < 0.1 km)
- Cloud containerization (Docker)
- Batch inference pipelines
- Model versioning & monitoring
- Integration with radio network simulators

---

**Author:** Bertrand Unwan  
**Program:** MEng Software Engineering & Intelligent Systems  
**Institution:** University of Alberta  

---