"""
BRDF (Bidirectional Reflectance Distribution Function) Correction Module

This module provides functions for correcting TROPOMI SIF observations for 
bidirectional reflectance effects using the BRDF kernel-based model.

The module implements:
1. Ross-thick kernel for volumetric scattering
2. Li-sparse kernel for geometric scattering
3. Vectorized BRDF correction for efficient processing

Author: Research Team
Date: 2024
License: MIT
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Union, Tuple, Optional


@njit(fastmath=True, cache=True, parallel=True)
def ross_thick_kernel(sun_zenith: np.ndarray, view_zenith: np.ndarray, 
                      relative_azimuth: np.ndarray) -> np.ndarray:
    """
    Calculate the Ross-thick (volumetric) BRDF kernel.
    
    This kernel accounts for scattering within a canopy layer, commonly used
    for vegetation reflectance modeling. The Ross-thick model is based on
    single scattering theory within a horizontally homogeneous canopy.
    
    Parameters:
    -----------
    sun_zenith : np.ndarray
        Solar zenith angles in radians. Shape: (n_samples,)
    view_zenith : np.ndarray
        Viewing zenith angles in radians. Shape: (n_samples,)
    relative_azimuth : np.ndarray
        Relative azimuth angles between sun and view in radians. Shape: (n_samples,)
    
    Returns:
    --------
    np.ndarray
        Ross-thick kernel values. Shape: (n_samples,)
    
    References:
    -----------
    Ross, J. K. (1981). The radiation regime and architecture of plant stands.
    Tasks for vegetation science, 3, 391.
    """
    # Calculate phase angle between sun and view directions
    cosxi = (np.cos(sun_zenith) * np.cos(view_zenith) + 
             np.sin(sun_zenith) * np.sin(view_zenith) * np.cos(relative_azimuth))
    
    # Clip to prevent numerical errors in arccos
    cosxi = np.clip(cosxi, -1.0, 1.0)
    xi = np.arccos(cosxi)
    
    # Ross-thick kernel formula
    k1 = ((np.pi / 2 - xi) * cosxi + np.sin(xi))
    k = (k1 / (np.cos(sun_zenith) + np.cos(view_zenith))) - np.pi / 4
    
    return k


@njit(fastmath=True, cache=True, parallel=True)
def li_sparse_kernel(sun_zenith: np.ndarray, view_zenith: np.ndarray,
                     relative_azimuth: np.ndarray) -> np.ndarray:
    """
    Calculate the Li-sparse (geometric-optical) BRDF kernel.
    
    This kernel models the geometric scattering effects from surface roughness
    and shadows. The Li-sparse model (also called Li-transit) is based on
    geometrical-optical theory assuming sparse canopy elements.
    
    Parameters:
    -----------
    sun_zenith : np.ndarray
        Solar zenith angles in radians. Shape: (n_samples,)
    view_zenith : np.ndarray
        Viewing zenith angles in radians. Shape: (n_samples,)
    relative_azimuth : np.ndarray
        Relative azimuth angles between sun and view in radians. Shape: (n_samples,)
    
    Returns:
    --------
    np.ndarray
        Li-sparse kernel values. Shape: (n_samples,)
    
    References:
    -----------
    Li, X., & Strahler, A. H. (1986). Geometric-optical bidirectional reflectance
    modeling of the discrete crown vegetation canopy: effect of crown shape,
    crown density, and mutual shadowing. IEEE Transactions on Geoscience and
    Remote Sensing, (5), 681-695.
    """
    # Normalize relative azimuth to [0, π]
    relative_azimuth_normalized = np.abs(relative_azimuth)
    relative_azimuth_normalized = np.where(
        relative_azimuth_normalized >= np.pi,
        2 * np.pi - relative_azimuth_normalized,
        relative_azimuth_normalized
    )
    
    # Li-sparse model parameters
    # These are typical values for vegetation canopies
    b_ratio = 1.0  # Ratio of crown width to height
    h_b_ratio = 2.0  # Height to width ratio
    
    # Compute projected angles
    t1 = b_ratio * np.tan(sun_zenith)
    theta_sp = np.arctan(t1)  # Projected sun angle
    
    t2 = b_ratio * np.tan(view_zenith)
    theta_vp = np.arctan(t2)  # Projected view angle
    
    cos_theta_sp = np.cos(theta_sp)
    cos_theta_vp = np.cos(theta_vp)
    
    # Calculate phase angle in projected space
    cos_xi_p = (cos_theta_sp * cos_theta_vp + 
                np.sin(theta_sp) * np.sin(theta_vp) * np.cos(relative_azimuth_normalized))
    
    # Distance parameter
    D1 = (np.tan(theta_sp) ** 2 + np.tan(theta_vp) ** 2 - 
          2 * np.tan(theta_sp) * np.tan(theta_vp) * np.cos(relative_azimuth_normalized))
    
    D = np.sqrt(np.maximum(D1, 0.0))  # Avoid sqrt of negative numbers
    
    # Shadow overlap parameter
    cost_temp1 = np.tan(theta_sp) * np.tan(theta_vp) * np.sin(relative_azimuth_normalized)
    cost_temp2 = D1 + cost_temp1 ** 2
    
    temp3 = 1.0 / (cos_theta_sp + cos_theta_vp)
    cost = h_b_ratio * np.sqrt(np.maximum(cost_temp2, 0.0)) * temp3
    cost = np.clip(cost, -1.0, 1.0)
    
    t = np.arccos(cost)
    
    # Calculate occlusion function O and backscatter alignment function B
    O = ((t - np.sin(t) * cost) * temp3 / np.pi)
    B = temp3 - O
    
    # Compute kernel with conditional logic
    k = np.where(
        B > 2.0,
        (1.0 + cos_xi_p) / (cos_theta_vp * cos_theta_sp * B) - 2.0,
        -B + (1.0 + cos_xi_p) / (2.0 * cos_theta_vp * cos_theta_sp)
    )
    
    return k


def validate_brdf_inputs(sun_zenith: Union[pd.Series, np.ndarray],
                         view_zenith: Union[pd.Series, np.ndarray],
                         relative_azimuth: Union[pd.Series, np.ndarray],
                         iso_coefficient: Union[pd.Series, np.ndarray],
                         vol_coefficient: Union[pd.Series, np.ndarray],
                         geo_coefficient: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, ...]:
    """
    Validate and convert input data to numpy arrays.
    
    Parameters:
    -----------
    sun_zenith : pd.Series or np.ndarray
        Solar zenith angles (degrees)
    view_zenith : pd.Series or np.ndarray
        Viewing zenith angles (degrees)
    relative_azimuth : pd.Series or np.ndarray
        Relative azimuth angles (degrees)
    iso_coefficient : pd.Series or np.ndarray
        Isotropic BRDF coefficient
    vol_coefficient : pd.Series or np.ndarray
        Volumetric kernel coefficient
    geo_coefficient : pd.Series or np.ndarray
        Geometric kernel coefficient
    
    Returns:
    --------
    Tuple[np.ndarray, ...]
        Validated numpy arrays in order: sun_zenith, view_zenith, relative_azimuth,
        iso_coefficient, vol_coefficient, geo_coefficient
    
    Raises:
    -------
    ValueError
        If inputs have mismatched lengths or invalid shapes
    """
    # Convert pandas Series to numpy arrays
    def to_numpy(x):
        if isinstance(x, pd.Series):
            return x.values.astype(np.float32)
        elif isinstance(x, np.ndarray):
            return x.astype(np.float32)
        else:
            return np.array(x, dtype=np.float32)
    
    sza_arr = to_numpy(sun_zenith)
    vza_arr = to_numpy(view_zenith)
    raa_arr = to_numpy(relative_azimuth)
    iso_arr = to_numpy(iso_coefficient)
    vol_arr = to_numpy(vol_coefficient)
    geo_arr = to_numpy(geo_coefficient)
    
    # Check shapes match
    expected_shape = sza_arr.shape[0]
    arrays = [sza_arr, vza_arr, raa_arr, iso_arr, vol_arr, geo_arr]
    array_names = ['sun_zenith', 'view_zenith', 'relative_azimuth', 
                   'iso_coefficient', 'vol_coefficient', 'geo_coefficient']
    
    for arr, name in zip(arrays, array_names):
        if arr.shape[0] != expected_shape:
            raise ValueError(
                f"Input '{name}' has length {arr.shape[0]}, "
                f"but expected {expected_shape}"
            )
    
    # Check for NaN values
    nan_mask = np.isnan(sza_arr) | np.isnan(vza_arr) | np.isnan(raa_arr)
    if nan_mask.any():
        print(f"Warning: Found {nan_mask.sum()} rows with NaN in angle data")
    
    return sza_arr, vza_arr, raa_arr, iso_arr, vol_arr, geo_arr


def brdf_correction(sun_zenith: Union[pd.Series, np.ndarray],
                   view_zenith: Union[pd.Series, np.ndarray],
                   relative_azimuth: Union[pd.Series, np.ndarray],
                   iso_coefficient: Union[pd.Series, np.ndarray],
                   vol_coefficient: Union[pd.Series, np.ndarray],
                   geo_coefficient: Union[pd.Series, np.ndarray],
                   return_kernels: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Perform vectorized BRDF correction on input data.
    
    The BRDF is modeled as a linear combination of three kernels:
    BRDF = iso_coefficient * 1 
           + vol_coefficient * Ross_thick(angles)
           + geo_coefficient * Li_sparse(angles)
    
    This is the standard linear kernel-driven BRDF model used in remote sensing.
    
    Parameters:
    -----------
    sun_zenith : pd.Series or np.ndarray
        Solar zenith angles in degrees. Shape: (n_samples,)
    view_zenith : pd.Series or np.ndarray
        Viewing zenith angles in degrees. Shape: (n_samples,)
    relative_azimuth : pd.Series or np.ndarray
        Relative azimuth angles in degrees. Shape: (n_samples,)
    iso_coefficient : pd.Series or np.ndarray
        Isotropic BRDF coefficient (scalar weight)
    vol_coefficient : pd.Series or np.ndarray
        Volumetric kernel coefficient (scalar weight)
    geo_coefficient : pd.Series or np.ndarray
        Geometric kernel coefficient (scalar weight)
    return_kernels : bool, optional
        If True, also return the kernel values for inspection. Default: False
    
    Returns:
    --------
    np.ndarray
        BRDF-corrected reflectance values (or SIF values if applied to SIF)
    
    Optional (if return_kernels=True):
        Tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns (corrected_values, ross_kernel, li_kernel) for inspection
    
    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from brdf_correction import brdf_correction
    >>>
    >>> # Sample TROPOMI data
    >>> sza = pd.Series([30.0, 35.0, 40.0])
    >>> vza = pd.Series([25.0, 30.0, 35.0])
    >>> raa = pd.Series([120.0, 130.0, 140.0])
    >>> iso_r = pd.Series([0.05, 0.06, 0.07])
    >>> vol_r = pd.Series([0.10, 0.11, 0.12])
    >>> geo_r = pd.Series([0.02, 0.03, 0.04])
    >>>
    >>> # Perform BRDF correction
    >>> reflectance_corrected = brdf_correction(
    ...     sun_zenith=sza,
    ...     view_zenith=vza,
    ...     relative_azimuth=raa,
    ...     iso_coefficient=iso_r,
    ...     vol_coefficient=vol_r,
    ...     geo_coefficient=geo_r
    ... )
    """
    # Validate and convert inputs
    sza, vza, raa, iso, vol, geo = validate_brdf_inputs(
        sun_zenith, view_zenith, relative_azimuth,
        iso_coefficient, vol_coefficient, geo_coefficient
    )
    
    # Convert angles from degrees to radians
    sza_rad = np.radians(sza)
    vza_rad = np.radians(vza)
    raa_rad = np.radians(raa)
    
    # Calculate BRDF kernels
    ross_k = ross_thick_kernel(sza_rad, vza_rad, raa_rad)
    li_k = li_sparse_kernel(sza_rad, vza_rad, raa_rad)
    
    # Linear kernel-driven BRDF model
    # BRDF = iso + vol * K_vol + geo * K_geo
    brdf_corrected = iso + vol * ross_k + geo * li_k
    
    # Replace infinite values with NaN
    brdf_corrected = np.where(np.isinf(brdf_corrected), np.nan, brdf_corrected)
    
    if return_kernels:
        return brdf_corrected, ross_k, li_k
    else:
        return brdf_corrected


def apply_multi_angle_correction(df: pd.DataFrame,
                                sun_zenith_col: str = 'sza',
                                view_zenith_col: str = 'vza',
                                relative_azimuth_col: str = 'raa',
                                iso_r_col: str = 'iso_r',
                                vol_r_col: str = 'vol_r',
                                geo_r_col: str = 'geo_r',
                                iso_n_col: str = 'iso_n',
                                vol_n_col: str = 'vol_n',
                                geo_n_col: str = 'geo_n',
                                view_zenith_steps: Optional[list] = None,
                                sun_zenith_steps: Optional[list] = None,
                                verbose: bool = True) -> pd.DataFrame:
    """
    Apply BRDF correction across multiple viewing geometries for TROPOMI data.
    
    This function generates hyperspectral SIF and vegetation indices at different
    viewing angles by correcting for viewing geometry effects. This simulates
    observations from different satellite viewing angles to characterize angular
    dependencies of the signal.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with TROPOMI observations including angle and BRDF coefficients
    sun_zenith_col : str
        Column name for solar zenith angle (degrees)
    view_zenith_col : str
        Column name for viewing zenith angle (degrees)
    relative_azimuth_col : str
        Column name for relative azimuth (degrees)
    iso_r_col, vol_r_col, geo_r_col : str
        Column names for red band BRDF coefficients
    iso_n_col, vol_n_col, geo_n_col : str
        Column names for NIR band BRDF coefficients
    view_zenith_steps : list, optional
        VZA values to generate (default: 0-60° in 5° steps)
    sun_zenith_steps : list, optional
        SZA values to generate (default: 0-60° in 5° steps)
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with original columns plus new multi-angle corrected columns
    
    Example:
    --------
    >>> df_corrected = apply_multi_angle_correction(
    ...     df_tropomi,
    ...     view_zenith_steps=[0, 15, 30, 45, 60],
    ...     sun_zenith_steps=[20, 30, 40, 50]
    ... )
    """
    from tqdm import tqdm
    
    df_result = df.copy()
    
    # Default angle steps
    if view_zenith_steps is None:
        view_zenith_steps = list(range(0, 61, 5))
    if sun_zenith_steps is None:
        sun_zenith_steps = list(range(0, 61, 5))
    
    # Calculate reference conditions (original viewing geometry)
    if verbose:
        print("Calculating reference reflectances at original geometry...")
    
    red_ref = brdf_correction(
        df_result[sun_zenith_col],
        df_result[view_zenith_col],
        df_result[relative_azimuth_col],
        df_result[iso_r_col],
        df_result[vol_r_col],
        df_result[geo_r_col]
    )
    
    nir_ref = brdf_correction(
        df_result[sun_zenith_col],
        df_result[view_zenith_col],
        df_result[relative_azimuth_col],
        df_result[iso_n_col],
        df_result[vol_n_col],
        df_result[geo_n_col]
    )
    
    # Calculate reference NDVI and NIRv
    ndvi_ref = (nir_ref - red_ref) / (nir_ref + red_ref + 1e-9)
    nirv_ref = ndvi_ref * nir_ref
    
    # Store original SIF
    if 'sif743' in df_result.columns:
        sif_original = df_result['sif743'].values
    else:
        sif_original = None
    
    # Generate corrected values for each angle combination
    total_combinations = len(view_zenith_steps) * len(sun_zenith_steps)
    
    if verbose:
        pbar = tqdm(total=total_combinations, 
                   desc="Multi-angle correction progress", 
                   unit="angles")
    
    for vza_step in view_zenith_steps:
        for sza_step in sun_zenith_steps:
            # Create angle series
            sza_series = pd.Series(sza_step, index=df_result.index)
            vza_series = pd.Series(vza_step, index=df_result.index)
            
            # Correct red and NIR bands
            red_corrected = brdf_correction(
                sza_series, vza_series, df_result[relative_azimuth_col],
                df_result[iso_r_col], df_result[vol_r_col], df_result[geo_r_col]
            )
            
            nir_corrected = brdf_correction(
                sza_series, vza_series, df_result[relative_azimuth_col],
                df_result[iso_n_col], df_result[vol_n_col], df_result[geo_n_col]
            )
            
            # Calculate vegetation indices
            ndvi_corrected = (nir_corrected - red_corrected) / (nir_corrected + red_corrected + 1e-9)
            nirv_corrected = ndvi_corrected * nir_corrected
            evi2_corrected = (2.5 * (nir_corrected - red_corrected) / 
                            (nir_corrected + 2.4 * red_corrected + 1.0))
            
            # Apply ratio to original SIF
            if sif_original is not None:
                nirv_ratio = nirv_corrected / (nirv_ref + 1e-9)
                sif_corrected = sif_original * nirv_ratio
            else:
                sif_corrected = np.full_like(red_corrected, np.nan)
            
            # Store results in dataframe
            prefix = f"angle_v{vza_step:02d}_s{sza_step:02d}"
            df_result[f"{prefix}_RED"] = red_corrected.astype(np.float16)
            df_result[f"{prefix}_NIR"] = nir_corrected.astype(np.float16)
            df_result[f"{prefix}_NDVI"] = ndvi_corrected.astype(np.float16)
            df_result[f"{prefix}_NIRv"] = nirv_corrected.astype(np.float16)
            df_result[f"{prefix}_EVI2"] = evi2_corrected.astype(np.float16)
            df_result[f"{prefix}_SIF"] = sif_corrected.astype(np.float16)
            
            if verbose:
                pbar.update(1)
    
    if verbose:
        pbar.close()
        print(f"Added {total_combinations} multi-angle corrected columns")
    
    return df_result


# Backward compatibility: Legacy function name
def BRDF_degree_vectorized(sun_zenith, view_zenith, relative_azimuth,
                          iso_coefficient, vol_coefficient, geo_coefficient):
    """
    Legacy interface for BRDF correction (deprecated, use brdf_correction instead).
    
    This function maintains backward compatibility with older code.
    New code should use brdf_correction() instead.
    """
    return brdf_correction(
        sun_zenith, view_zenith, relative_azimuth,
        iso_coefficient, vol_coefficient, geo_coefficient
    )


if __name__ == "__main__":
    # Example usage
    print("BRDF Correction Module")
    print("=" * 50)
    
    # Create sample data
    n_samples = 100
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'sza': np.random.uniform(10, 50, n_samples),
        'vza': np.random.uniform(0, 45, n_samples),
        'raa': np.random.uniform(0, 180, n_samples),
        'iso_r': np.random.uniform(0.03, 0.08, n_samples),
        'vol_r': np.random.uniform(0.08, 0.15, n_samples),
        'geo_r': np.random.uniform(0.01, 0.05, n_samples),
        'iso_n': np.random.uniform(0.05, 0.12, n_samples),
        'vol_n': np.random.uniform(0.15, 0.35, n_samples),
        'geo_n': np.random.uniform(0.03, 0.10, n_samples),
        'sif743': np.random.uniform(0.5, 2.5, n_samples),
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample data columns: {list(sample_data.columns)}")
    
    # Apply BRDF correction
    print("\nApplying BRDF correction...")
    result = brdf_correction(
        sample_data['sza'],
        sample_data['vza'],
        sample_data['raa'],
        sample_data['iso_r'],
        sample_data['vol_r'],
        sample_data['geo_r'],
        return_kernels=True
    )
    
    corrected, ross_k, li_k = result
    print(f"Corrected reflectance range: [{np.nanmin(corrected):.4f}, {np.nanmax(corrected):.4f}]")
    print(f"Ross kernel range: [{np.nanmin(ross_k):.4f}, {np.nanmax(ross_k):.4f}]")
    print(f"Li kernel range: [{np.nanmin(li_k):.4f}, {np.nanmax(li_k):.4f}]")
    
    print("\nExample completed successfully!")
