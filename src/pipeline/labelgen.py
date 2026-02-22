import numpy as np
import xarray as xr

from pathlib import Path


# --- Helper: Robustly convert xarray/numpy object to Python float ---
def _scalar(x) -> float:
    """Robustly convert an xarray/numpy object to a Python float.

    Handles xarray DataArray/Variable (including dask-backed), numpy arrays/scalars,
    and plain Python numerics.

    If `x` is a DataArray, we take its median (over all dims) before converting.
    """
    try:
        # xarray DataArray / Variable
        if hasattr(x, "median") and hasattr(x, "values"):
            m = x.median()
            if hasattr(m, "compute"):
                m = m.compute()
            return float(np.asarray(m.values).item())

        # numpy arrays / scalars
        if hasattr(x, "shape"):
            return float(np.asarray(x).item())

        return float(x)
    except Exception:
        # Last resort: try .item()
        try:
            return float(x.item())
        except Exception as e:
            raise TypeError(f"Could not convert to float: {type(x)}") from e

def _get_var(ds: xr.Dataset, name: str):
    """Safe variable getter with a clear error message."""
    if name not in ds:
        raise KeyError(f"Required variable '{name}' not found. Available vars include: {list(ds.data_vars)[:25]} ...")
    return ds[name]


def _get_mass_proxy(ds: xr.Dataset, base_name: str):
    """
    Returns a mass-like variable in linear space.
    If LOG1P_<base_name> exists, converts back using expm1.
    Otherwise uses <base_name> directly.
    """
    log_name = f"LOG1P_{base_name}"
    if log_name in ds:
        return np.expm1(ds[log_name])
    return _get_var(ds, base_name)


# --- Helper: Estimate visibility from available dust proxies ---
def _estimate_visibility_km(ds: xr.Dataset) -> xr.DataArray:
    """Estimate meteorological visibility V (km) from available dust proxies.

    Priority:
      1) Use VISIBILITY_KM if present.
      2) Use dust optical depth / extinction (DUEXTTAU / DUSCATAU) via Koschmieder.
      3) Fallback: map dust mass proxy to visibility using a monotone inverse relation.

    Notes:
    - This is only for label generation (physics-inspired surrogate labels).
    - Tune the fallback constants to match literature magnitudes if needed.
    """
    # 1) Direct visibility
    if "VISIBILITY_KM" in ds:
        V = ds["VISIBILITY_KM"].astype("float32")
        V = V.clip(min=0.05)
        V.name = "VISIBILITY_KM"
        return V

    # 2) AOD-based visibility (Koschmieder): V ≈ 3.912 / beta_ext, beta_ext ≈ AOD / H
    aod_name = None
    for cand in ("DUEXTTAU", "DUSCATAU", "AOD_DUST", "DUST_AOD", "DU_AOD", "AODDU"):
        if cand in ds:
            aod_name = cand
            break

    if aod_name is not None:
        aod = ds[aod_name].astype("float32").clip(min=1e-6)
        # Effective dust layer height in km (dynamic wind-driven field when U10M/V10M exist)
        H_km = _estimate_dust_layer_height_km(ds)
        # Adaptive Koschmieder constant: size + humidity aware
        K = _estimate_koschmieder_constant(ds)

        # Optional use of scattering vs extinction to adjust effective extinction slightly
        # (kept mild + bounded, purely as an upgrade knob)
        if "DUEXTTAU" in ds and "DUSCATAU" in ds:
            ext = ds["DUEXTTAU"].astype("float32").clip(min=1e-6)
            sca = ds["DUSCATAU"].astype("float32").clip(min=0.0)
            ssa = (sca / ext).clip(min=0.0, max=1.0)
            k_ssa = np.float32(ds.attrs.get("SSA_EXT_SENS", 0.15))  # small adjustment strength
            aod_eff = aod * (1.0 + k_ssa * (ssa - np.float32(0.9)))
            aod_eff = aod_eff.clip(min=1e-6)
        else:
            aod_eff = aod

        V = (K * H_km / aod_eff).astype("float32")
        V = V.clip(min=0.05, max=200.0)
        V.name = "VISIBILITY_KM"
        V.attrs["derived_from"] = aod_name
        V.attrs["method"] = "Koschmieder + layer height approximation"
        V.attrs["layer_height_source"] = "wind_dynamic" if ("U10M" in ds and "V10M" in ds) else "constant"
        V.attrs["layer_height_median_km"] = _scalar(H_km) if hasattr(H_km, "median") else float(H_km)
        V.attrs["koschmieder_base"] = float(ds.attrs.get("K_VIS_BASE", 3.912))
        V.attrs["koschmieder_median"] = _scalar(K) if hasattr(K, "median") else float(K)
        V.attrs["ssa_ext_sens"] = float(ds.attrs.get("SSA_EXT_SENS", 0.15))
        V.attrs["aod_eff_used"] = "yes" if ("DUEXTTAU" in ds and "DUSCATAU" in ds) else "no"
        return V

    # 3) Fallback: dust-mass-proxy -> visibility mapping
    # Prefer DUCMASS (column dust mass proxy); if absent, fallback to DUSMASS.
    mass_name = "DUCMASS" if "DUCMASS" in ds else ("DUSMASS" if "DUSMASS" in ds else None)
    if mass_name is None:
        raise KeyError(
            "No usable dust mass proxy found for visibility fallback. Expected one of: DUCMASS, DUSMASS. "
            f"Available vars include: {list(ds.data_vars)[:25]} ..."
        )

    dust_mass = _get_mass_proxy(ds, mass_name).astype("float32")

    # Estimate fine fraction using PM2.5 column mass if available:
    #   fine_frac ≈ DUCMASS25 / DUCMASS  (clipped to [0, 1])
    if "DUCMASS25" in ds and mass_name == "DUCMASS":
        dust_mass25 = _get_mass_proxy(ds, "DUCMASS25").astype("float32")
        fine_frac = (dust_mass25 / (dust_mass + 1e-12)).clip(min=0.0, max=1.0)
    elif "DUSMASS25" in ds and mass_name == "DUSMASS":
        dust_mass25 = _get_mass_proxy(ds, "DUSMASS25").astype("float32")
        fine_frac = (dust_mass25 / (dust_mass + 1e-12)).clip(min=0.0, max=1.0)
    else:
        # If we don't have a 2.5µm split, assume a moderate fine fraction.
        fine_frac = xr.zeros_like(dust_mass) + np.float32(0.5)

    # Combined dust loading proxy (heavier dust and higher fine fraction -> lower visibility)
    dust_loading = dust_mass * (0.5 + fine_frac)

    # Tunable mapping constants
    V_clear_km = float(ds.attrs.get("V_CLEAR_KM", 20.0))  # clear-air visibility
    k_mass = float(ds.attrs.get("V_K_MASS", 8.0))         # sensitivity to dust_loading

    V = (V_clear_km / (1.0 + k_mass * dust_loading)).astype("float32")
    V = V.clip(min=0.05, max=200.0)
    V.name = "VISIBILITY_KM"
    V.attrs["derived_from"] = f"{mass_name},{mass_name}25" if f"{mass_name}25" in ds else mass_name
    V.attrs["method"] = "Monotone inverse mapping (fallback)"
    return V

def _estimate_particle_radius_m(ds: xr.Dataset) -> xr.DataArray:
    """Estimate equivalent particle radius a_e (m) as a per-sample DataArray.

    Bigger jump: instead of a single dataset-wide radius, we compute a_e(x,y,t,...) using
    DUANGSTR (dust Ångström exponent) at each grid/time point.

      - higher DUANGSTR  -> finer particles -> smaller a_e
      - lower DUANGSTR   -> coarser particles -> larger a_e

    If DUANGSTR is missing, we return a constant radius broadcast to the same shape as
    another dust variable if possible, otherwise a scalar DataArray.

    Tuning knobs (dataset attrs):
      - DUST_PARTICLE_RADIUS_M: fixed fallback value (default 50e-6)
      - AE_MIN_UM / AE_MAX_UM: radius bounds in micrometers (defaults 15 and 80)
      - ANG_MIN / ANG_MAX: DUANGSTR range to map (defaults 0.0 and 1.5)

    Returns:
      xr.DataArray of radius in meters.
    """
    a0 = np.float32(ds.attrs.get("DUST_PARTICLE_RADIUS_M", 50e-6))

    ae_min_um = np.float32(ds.attrs.get("AE_MIN_UM", 15.0))
    ae_max_um = np.float32(ds.attrs.get("AE_MAX_UM", 80.0))
    ang_min = np.float32(ds.attrs.get("ANG_MIN", 0.0))
    ang_max = np.float32(ds.attrs.get("ANG_MAX", 1.5))

    # Helper to build a constant DataArray with a reasonable shape
    def _const_like_any(var_names: tuple[str, ...]) -> xr.DataArray:
        for vn in var_names:
            if vn in ds:
                return xr.full_like(ds[vn].astype("float32"), a0, dtype="float32")
        return xr.DataArray(np.float32(a0))

    if "DUANGSTR" not in ds:
        return _const_like_any(("DUEXTTAU", "DUSCATAU", "DUCMASS", "DUSMASS"))

    ang = ds["DUANGSTR"].astype("float32")

    # Clip DUANGSTR into [ang_min, ang_max]
    ang_clipped = ang.clip(min=float(ang_min), max=float(ang_max))

    # Normalize DUANGSTR to [0, 1]
    t = (ang_clipped - ang_min) / (ang_max - ang_min + np.float32(1e-12))

    # --- Nonlinear mapping + hygroscopic growth ---
    # Nonlinear mapping: make radius more sensitive near extremes.
    # We use a logistic curve on centered t, then map to [AE_MAX, AE_MIN].
    # Larger logistic_slope => steeper transition around the midpoint.
    logistic_slope = np.float32(ds.attrs.get("AE_LOGISTIC_SLOPE", 6.0))
    logistic_mid = np.float32(ds.attrs.get("AE_LOGISTIC_MID", 0.5))

    # logistic in [0,1]
    s = 1.0 / (1.0 + np.exp(-logistic_slope * (t - logistic_mid)))

    # Map: low ang (t~0 -> s small) => ae ~ AE_MAX; high ang (t~1 -> s large) => ae ~ AE_MIN
    ae_um = ae_max_um + (ae_min_um - ae_max_um) * s

    # Hygroscopic growth (light): at higher RH, effective radius increases.
    # growth = 1 + g0 * max(0, RH - RH_GROWTH_ONSET)^(GROWTH_POW)
    if all(v in ds for v in ("QV2M", "T2M", "PS")):
        RH = _estimate_relative_humidity(ds)
        rh_onset = np.float32(ds.attrs.get("RH_GROWTH_ONSET", 0.6))
        g0 = np.float32(ds.attrs.get("AE_RH_GROWTH", 0.25))
        gp = np.float32(ds.attrs.get("GROWTH_POW", 1.3))
        rh_excess = (RH.astype("float32") - rh_onset).clip(min=0.0)
        growth = (1.0 + g0 * (rh_excess ** gp)).astype("float32")
        growth = growth.clip(min=1.0, max=float(ds.attrs.get("AE_GROWTH_MAX", 2.0)))
        ae_um = (ae_um * growth).astype("float32")
    else:
        RH = None

    # Convert to meters
    ae_m = (ae_um * np.float32(1e-6)).astype("float32")
    ae_m.name = "DUST_PARTICLE_RADIUS_M"
    ae_m.attrs["units"] = "m"
    ae_m.attrs["derived_from"] = "DUANGSTR"
    ae_m.attrs["ae_min_um"] = float(ae_min_um)
    ae_m.attrs["ae_max_um"] = float(ae_max_um)
    ae_m.attrs["ang_min"] = float(ang_min)
    ae_m.attrs["ang_max"] = float(ang_max)
    ae_m.attrs["mapping"] = "logistic_duangstr + hygroscopic_growth"
    ae_m.attrs["logistic_slope"] = float(ds.attrs.get("AE_LOGISTIC_SLOPE", 6.0))
    ae_m.attrs["logistic_mid"] = float(ds.attrs.get("AE_LOGISTIC_MID", 0.5))
    ae_m.attrs["rh_growth_onset"] = float(ds.attrs.get("RH_GROWTH_ONSET", 0.6))
    ae_m.attrs["ae_rh_growth"] = float(ds.attrs.get("AE_RH_GROWTH", 0.25))
    ae_m.attrs["growth_pow"] = float(ds.attrs.get("GROWTH_POW", 1.3))
    ae_m.attrs["ae_growth_max"] = float(ds.attrs.get("AE_GROWTH_MAX", 2.0))
    return ae_m


# --- Helper: Estimate 2m relative humidity from QV2M, T2M, PS ---
def _estimate_relative_humidity(ds: xr.Dataset) -> xr.DataArray:
    """Estimate 2m relative humidity (0-1) from MERRA-2 QV2M, T2M, PS.

    Uses standard meteorological approximations:
      - vapor pressure e from specific humidity q and surface pressure p
      - saturation vapor pressure es from temperature (Bolton 1980 style)
      - RH = e / es

    Returns:
      xr.DataArray in [0,1].
    """
    q = _get_var(ds, "QV2M").astype("float32")          # kg/kg
    T = _get_var(ds, "T2M").astype("float32")          # K
    p = _get_var(ds, "PS").astype("float32")           # Pa

    # Convert to hPa for numerics
    p_hpa = p / np.float32(100.0)

    # Vapor pressure from specific humidity (approx): e = q * p / (0.622 + 0.378 q)
    e_hpa = (q * p_hpa) / (np.float32(0.622) + np.float32(0.378) * q)

    # Saturation vapor pressure (hPa). Bolton (1980): es = 6.112 * exp(17.67 Tc / (Tc + 243.5))
    Tc = T - np.float32(273.15)
    es_hpa = np.float32(6.112) * np.exp((np.float32(17.67) * Tc) / (Tc + np.float32(243.5)))

    rh = (e_hpa / (es_hpa + np.float32(1e-6))).astype("float32")
    rh = rh.clip(min=0.0, max=1.0)
    rh.name = "RH2M"
    rh.attrs["units"] = "1"
    rh.attrs["description"] = "Estimated 2m relative humidity from QV2M, T2M, PS"
    return rh

# --- Helper: Dynamic dust layer height (km) from 10m winds ---
def _estimate_dust_layer_height_km(ds: xr.Dataset) -> xr.DataArray:
    """Estimate effective dust layer height H (km) as a per-sample field.

    Upgrade: instead of a constant H in the visibility-from-extinction step,
    we let H vary with near-surface wind speed (uplift/mixing proxy).

    H_km = clip(H0_km + k_w * WSPD10, H_min_km, H_max_km)

    where WSPD10 = sqrt(U10M^2 + V10M^2).

    Tuning knobs (dataset attrs):
      - DUST_LAYER_HEIGHT_KM: baseline H0 (default 1.5 km)
      - H_WIND_SENS: k_w (default 0.08 km per (m/s))
      - H_MIN_KM / H_MAX_KM: bounds (defaults 0.3 and 5.0)

    If U10M/V10M are missing, returns a constant DataArray (or scalar) using H0.
    """
    H0 = np.float32(ds.attrs.get("DUST_LAYER_HEIGHT_KM", 1.5))
    k_w = np.float32(ds.attrs.get("H_WIND_SENS", 0.08))
    H_min = np.float32(ds.attrs.get("H_MIN_KM", 0.3))
    H_max = np.float32(ds.attrs.get("H_MAX_KM", 5.0))

    if "U10M" in ds and "V10M" in ds:
        u = _get_var(ds, "U10M").astype("float32")
        v = _get_var(ds, "V10M").astype("float32")
        wspd10 = np.sqrt(u * u + v * v).astype("float32")
        H = (H0 + k_w * wspd10).astype("float32")
        H = H.clip(min=float(H_min), max=float(H_max))
        H.name = "DUST_LAYER_HEIGHT_KM"
        H.attrs["units"] = "km"
        H.attrs["derived_from"] = "U10M,V10M"
        H.attrs["method"] = "Linear wind-mixing proxy"
        H.attrs["H0_km"] = float(H0)
        H.attrs["k_w_km_per_mps"] = float(k_w)
        H.attrs["H_min_km"] = float(H_min)
        H.attrs["H_max_km"] = float(H_max)
        return H

    # Fallback: constant H0 with a reasonable shape if possible
    for vn in ("DUEXTTAU", "DUSCATAU", "DUCMASS", "DUSMASS", "T2M"):
        if vn in ds:
            return xr.full_like(ds[vn].astype("float32"), H0, dtype="float32")
    return xr.DataArray(np.float32(H0))

# --- Helper: Adaptive Koschmieder constant (dimensionless) ---
def _estimate_koschmieder_constant(ds: xr.Dataset) -> xr.DataArray:
    """Estimate an adaptive Koschmieder constant K (dimensionless) for visibility.

    Classic Koschmieder uses K=3.912 (contrast threshold ~0.02). For SDS conditions,
    perceived contrast can vary with particle size regime (DUANGSTR) and humidity.

    Upgrade: K becomes a per-sample field:

      K = clip(K0 * (1 + k_ang*(DUANGSTR-ANG_REF)) * (1 + k_rh*(RH2M-RH_REF_VIS)), K_MIN, K_MAX)

    - Higher DUANGSTR (finer dust) -> typically stronger forward scattering / haze -> lower contrast -> higher K
      (positive k_ang makes K increase with DUANGSTR)
    - Higher RH -> haze / reduced contrast -> higher K (positive k_rh)

    Tuning knobs (dataset attrs):
      - K_VIS_BASE: K0 (default 3.912)
      - K_ANG_SENS: k_ang (default 0.20)
      - ANG_REF: reference DUANGSTR (default 0.6)
      - K_RH_SENS: k_rh (default 0.50)
      - RH_REF_VIS: reference RH (default 0.3)
      - K_MIN / K_MAX: bounds (defaults 2.0 and 8.0)

    Returns:
      xr.DataArray K with broadcastable shape.
    """
    K0 = np.float32(ds.attrs.get("K_VIS_BASE", 3.912))
    k_ang = np.float32(ds.attrs.get("K_ANG_SENS", 0.20))
    ang_ref = np.float32(ds.attrs.get("ANG_REF", 0.6))
    k_rh = np.float32(ds.attrs.get("K_RH_SENS", 0.50))
    rh_ref_vis = np.float32(ds.attrs.get("RH_REF_VIS", 0.3))
    K_min = np.float32(ds.attrs.get("K_MIN", 2.0))
    K_max = np.float32(ds.attrs.get("K_MAX", 8.0))

    # Build a constant field with a reasonable shape
    def _const_like_any(var_names: tuple[str, ...]) -> xr.DataArray:
        for vn in var_names:
            if vn in ds:
                return xr.full_like(ds[vn].astype("float32"), K0, dtype="float32")
        return xr.DataArray(np.float32(K0))

    K = _const_like_any(("DUEXTTAU", "DUSCATAU", "DUANGSTR", "T2M"))

    if "DUANGSTR" in ds:
        ang = ds["DUANGSTR"].astype("float32")
        K = K * (1.0 + k_ang * (ang - ang_ref))

    if all(v in ds for v in ("QV2M", "T2M", "PS")):
        RH = _estimate_relative_humidity(ds)
        K = K * (1.0 + k_rh * (RH - rh_ref_vis))

    K = K.astype("float32").clip(min=float(K_min), max=float(K_max))
    K.name = "KOSCHMIEDER_K"
    K.attrs["units"] = "1"
    K.attrs["K0"] = float(K0)
    K.attrs["k_ang"] = float(k_ang)
    K.attrs["ang_ref"] = float(ang_ref)
    K.attrs["k_rh"] = float(k_rh)
    K.attrs["rh_ref_vis"] = float(rh_ref_vis)
    K.attrs["K_min"] = float(K_min)
    K.attrs["K_max"] = float(K_max)
    return K

def compute_attenuation_db_per_km(ds: xr.Dataset, freq_ghz: float) -> xr.DataArray:
    """Compute frequency-specific dust attenuation label (dB/km) for a given frequency.

    Physics-inspired label generator based on Elabdin et al. (PIER M, 2009), where
    specific attenuation depends on frequency f (GHz), visibility V (km), particle
    radius a_e (m), and complex permittivity (ε', ε'').

    Reference equations:
      - Ad = (α a_e f)/((0.3) V) + (β a_e^3 f^3)/((0.3)^3 V^3) + (θ a_e^4 f^4)/((0.3)^4 V^4)
      - α, β, θ are functions of ε', ε''.

    We estimate V from available dust proxies via `_estimate_visibility_km`.
    """
    # --- Inputs required for V estimation (from our pipeline) ---
    V_km = _estimate_visibility_km(ds)  # km

    # --- Model constants (tuneable, but with sensible defaults) ---
    # Equivalent particle radius a_e (meters) as a per-sample DataArray.
    a_e_m = _estimate_particle_radius_m(ds)
    a_e_m = a_e_m.astype("float32").clip(min=1e-6, max=5e-4)

    # Complex permittivity for dust at Ka-band can be set via attrs.
    # Defaults below align with values commonly used for Ka-band in the cited work.
    eps_real_base = float(ds.attrs.get("DUST_EPS_REAL", 4.0))
    eps_imag_base = float(ds.attrs.get("DUST_EPS_IMAG", 1.33))

    # coupled microphysics (size + humidity) into dielectric loss ---
    # RH modulation parameters
    rh_k = float(ds.attrs.get("EPS_IMAG_RH_SENS", 0.6))   # strength
    rh_ref = float(ds.attrs.get("RH_REF", 0.3))          # baseline RH

    # DUANGSTR modulation parameters
    ang_k = float(ds.attrs.get("EPS_IMAG_ANG_SENS", 0.25))  # strength
    ang_ref = float(ds.attrs.get("EPS_ANG_REF", 0.6))       # reference DUANGSTR

    # Optional temperature modulation parameters (very mild)
    t_k = float(ds.attrs.get("EPS_IMAG_T_SENS", 0.02))       # per 10 K
    t_ref = float(ds.attrs.get("T_REF_K", 300.0))            # reference temperature

    # Build eps_imag as a broadcastable field
    base_field = None
    for vn in ("DUEXTTAU", "DUSCATAU", "DUANGSTR", "T2M", "PS"):
        if vn in ds:
            base_field = ds[vn].astype("float32")
            break
    if base_field is None:
        base_field = xr.DataArray(np.float32(1.0))

    eps_imag = xr.full_like(base_field, np.float32(eps_imag_base), dtype="float32")

    # RH term
    if all(v in ds for v in ("QV2M", "T2M", "PS")):
        RH = _estimate_relative_humidity(ds)
        eps_imag = eps_imag * (1.0 + np.float32(rh_k) * (RH - np.float32(rh_ref)))
    else:
        RH = None

    # DUANGSTR term
    if "DUANGSTR" in ds:
        ANG = ds["DUANGSTR"].astype("float32")
        eps_imag = eps_imag * (1.0 + np.float32(ang_k) * (ANG - np.float32(ang_ref)))
    else:
        ANG = None

    # Temperature term (optional)
    if "T2M" in ds:
        T = ds["T2M"].astype("float32")
        eps_imag = eps_imag * (1.0 + np.float32(t_k) * ((T - np.float32(t_ref)) / np.float32(10.0)))

    # Keep eps'' in a sane range
    eps_imag = eps_imag.astype("float32").clip(min=0.05, max=10.0)

    # eps' kept as a constant baseline (can be upgraded later)
    eps_real = float(eps_real_base)

    # --- Frequency-aware calibration knob ---
    # Prefer explicit per-frequency scales (e.g., ATTEN_LABEL_SCALE_28, ATTEN_LABEL_SCALE_38).
    # If not provided, optionally use a simple power-law scale: base * (f/f_ref)^p.
    f_tag = int(round(float(freq_ghz)))

    # 1) Explicit per-frequency override
    per_f_key = f"ATTEN_LABEL_SCALE_{f_tag}"
    if per_f_key in ds.attrs:
        label_scale = float(ds.attrs.get(per_f_key))
        label_scale_mode = per_f_key
    else:
        # 2) Power-law (optional): base * (f/f_ref)^p
        base = float(ds.attrs.get("ATTEN_LABEL_SCALE_BASE", ds.attrs.get("ATTEN_LABEL_SCALE", 1.0)))
        p = float(ds.attrs.get("ATTEN_LABEL_SCALE_P", 0.0))
        f_ref = float(ds.attrs.get("ATTEN_LABEL_SCALE_FREF", 28.0))
        label_scale = base * ((float(freq_ghz) / f_ref) ** p)
        label_scale_mode = "power_law" if ("ATTEN_LABEL_SCALE_P" in ds.attrs or "ATTEN_LABEL_SCALE_BASE" in ds.attrs) else "global"

    # --- Guard: enforce monotonic frequency ordering when per-frequency scales invert 28 vs 38 ---
    # If ATTEN_LABEL_SCALE_28 is much larger than ATTEN_LABEL_SCALE_38, the calibrated labels can
    # incorrectly produce 28 GHz attenuation > 38 GHz for typical conditions.
    # We correct this by applying a smooth frequency correction factor (f/28)^p_corr that:
    #   - leaves 28 GHz unchanged (factor = 1)
    #   - boosts higher frequencies enough so that the *effective* 38-GHz scale is >= the 28-GHz scale
    # This is only applied when BOTH per-frequency scales exist.
    p_corr = 0.0
    if ("ATTEN_LABEL_SCALE_28" in ds.attrs) and ("ATTEN_LABEL_SCALE_38" in ds.attrs):
        s28 = float(ds.attrs.get("ATTEN_LABEL_SCALE_28"))
        s38 = float(ds.attrs.get("ATTEN_LABEL_SCALE_38"))
        # Only correct if both are positive and would invert ordering.
        if (s28 > 0.0) and (s38 > 0.0) and (s28 > s38):
            # Choose p_corr so that: s38 * (38/28)^p_corr == s28
            p_corr = float(np.log(s28 / s38) / np.log(38.0 / 28.0))
            # Apply correction (28 unchanged; higher freqs boosted)
            label_scale = float(label_scale) * ((float(freq_ghz) / 28.0) ** p_corr)
            label_scale_mode = f"{label_scale_mode}+freq_corr"

    # --- Coefficients α, β, θ (from Elabdin et al.) ---
    denom = (eps_real + 2.0) ** 2 + (eps_imag ** 2)

    # Eq. (20)
    alpha = 565.8 * eps_imag / denom

    # Eq. (21)
    beta = 3.7e3 * eps_imag * (
        (6.0 / 5.0) * ((7.0 * (eps_real ** 2) + 7.0 * (eps_imag ** 2) + 4.0 * eps_real - 20.0) / (denom ** 2))
        + (1.0 / 15.0)
        + (5.0 / (3.0 * ((2.0 * eps_real + 3.0) ** 2 + 4.0 * (eps_imag ** 2))))
    )

    # Eq. (22)
    theta = 3.12e4 * (
        (((eps_real - 1.0) ** 2) * (eps_real + 2.0) + (2.0 * (eps_real - 1.0) * (eps_real + 2.0) - 9.0) + (eps_imag ** 4))
        / (denom ** 2)
    )
    alpha = alpha.clip(min=0.0)
    beta  = beta.clip(min=0.0)
    theta = theta.clip(min=0.0)

    # --- Specific attenuation (dB/km), Eq. (23) ---
    f = float(freq_ghz)
    V = V_km.astype("float32").clip(min=0.05)  # km

    term1 = (alpha * a_e_m * (f ** 1)) / ((0.3) * V)
    term2 = (beta * (a_e_m ** 3) * (f ** 3)) / (((0.3) ** 3) * (V ** 3))
    term3 = (theta * (a_e_m ** 4) * (f ** 4)) / (((0.3) ** 4) * (V ** 4))

    y = (label_scale * (term1 + term2 + term3)).astype("float32")
    y = y.where(np.isfinite(y), 0.0)
    # Defensive cleanup: no negatives, no inf
    y = y.clip(min=0.0)

    y.name = f"ATTEN_DB_PER_KM_{int(f)}GHZ"
    y.attrs["units"] = "dB/km"
    y.attrs["description"] = (
        f"Specific attenuation label at {f} GHz via Elabdin et al. (2009) with visibility estimated from dust proxies"
    )
    y.attrs["freq_ghz"] = f
    y.attrs["particle_radius_median_m"] = _scalar(a_e_m) if hasattr(a_e_m, "median") else float(a_e_m)
    y.attrs["ae_growth_enabled"] = "yes" if all(v in ds for v in ("QV2M", "T2M", "PS")) else "no"
    y.attrs["ae_median_um"] = (_scalar(a_e_m) * 1e6) if hasattr(a_e_m, "median") else (float(a_e_m) * 1e6)
    y.attrs["eps_real"] = float(eps_real)
    y.attrs["eps_imag_median"] = _scalar(eps_imag) if hasattr(eps_imag, "median") else float(eps_imag)
    y.attrs["eps_imag_base"] = float(eps_imag_base)

    # Coupled dielectric modulation metadata
    y.attrs["eps_imag_rh_sens"] = float(rh_k)
    y.attrs["rh_ref"] = float(rh_ref)
    y.attrs["rh_median"] = _scalar(RH) if RH is not None else np.nan

    y.attrs["eps_imag_ang_sens"] = float(ang_k)
    y.attrs["eps_ang_ref"] = float(ang_ref)
    y.attrs["duangstr_median"] = _scalar(ds["DUANGSTR"]) if "DUANGSTR" in ds else np.nan

    y.attrs["eps_imag_t_sens"] = float(t_k)
    y.attrs["t_ref_k"] = float(t_ref)

    y.attrs["met_modulation"] = "RH2M+DUANGSTR+T2M->eps_imag"
    y.attrs["label_scale"] = label_scale
    y.attrs["label_scale_mode"] = label_scale_mode
    y.attrs["label_scale_p_corr"] = float(p_corr)
    y.attrs["label_scale_f_tag"] = int(round(float(freq_ghz)))
    y.attrs["label_scale_base"] = float(ds.attrs.get("ATTEN_LABEL_SCALE_BASE", ds.attrs.get("ATTEN_LABEL_SCALE", 1.0)))
    y.attrs["label_scale_p"] = float(ds.attrs.get("ATTEN_LABEL_SCALE_P", 0.0))
    y.attrs["label_scale_fref"] = float(ds.attrs.get("ATTEN_LABEL_SCALE_FREF", 28.0))
    y.attrs["particle_radius_source"] = "DUANGSTR_logistic+RH_growth" if "DUANGSTR" in ds else "fixed"
    y.attrs["visibility_median_km"] = _scalar(V) if hasattr(V, "median") else float(V)
    y.attrs["visibility_source"] = V_km.attrs.get("derived_from", "unknown") if hasattr(V_km, "attrs") else "unknown"
    y.attrs["koschmieder_median"] = float(V_km.attrs.get("koschmieder_median", np.nan)) if hasattr(V_km, "attrs") else np.nan
    y.attrs["aod_eff_used"] = V_km.attrs.get("aod_eff_used", "unknown") if hasattr(V_km, "attrs") else "unknown"
    return y
