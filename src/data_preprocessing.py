# File: data_preprocessing.py
# Description: Data preprocessing and feature engineering pipeline

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from tqdm import tqdm


# ============================================================================
# 1. Configuration
# ============================================================================

# Input paths
SIF_CSV_PATH = './data/raw_sif_data.csv'  # Raw SIF TROPOMI data
VIS_CSV_PATH = './data/raw_vis_data.csv'  # Raw vegetation indices data
YIELD_CSV_PATH = './data/raw_yield_data.csv'  # Raw yield data

# Output paths
OUTPUT_DIR = './data/'
FINAL_FEATURES_CSV = os.path.join(OUTPUT_DIR, 'final_yield_features.csv')
FINAL_FEATURES_PKL = os.path.join(OUTPUT_DIR, 'final_yield_features.pkl')
TIME_INDEX_MAPPING = os.path.join(OUTPUT_DIR, 'time_index_mapping.csv')

# Processing parameters
AGGREGATION_METHOD = 'dekad'  # 10-day periods
START_DATE = '03-01'  # March 1st
END_DATE = '05-30'  # May 30th


# ============================================================================
# 2. BRDF Correction Function
# ============================================================================

def brdf_degree(sza, vza, raa, iso, vol, geo):
    """
    Apply BRDF (Bidirectional Reflectance Distribution Function) correction.
    
    This function corrects reflectance measurements for viewing geometry effects
    based on the Rahman-Pinty-Verstraete (RPV) BRDF model.
    
    Parameters
    ----------
    sza : float or np.ndarray
        Solar zenith angle (degrees)
    vza : float or np.ndarray
        Viewing zenith angle (degrees)
    raa : float or np.ndarray
        Relative azimuth angle (degrees)
    iso : float or np.ndarray
        Isotropic BRDF parameter
    vol : float or np.ndarray
        Volumetric BRDF parameter
    geo : float or np.ndarray
        Geometric BRDF parameter
    
    Returns
    -------
    np.ndarray
        BRDF-corrected reflectance
    """
    return iso + vol * np.cos(np.radians(sza)) * np.cos(np.radians(vza)) + \
           geo * np.cos(np.radians(raa))


# ============================================================================
# 3. Feature Engineering Functions
# ============================================================================

def compute_vegetation_indices(red, nir):
    """
    Compute vegetation indices from red and NIR reflectance.
    
    Parameters
    ----------
    red : np.ndarray
        Red band reflectance (400-700 nm)
    nir : np.ndarray
        Near-infrared band reflectance (700-1300 nm)
    
    Returns
    -------
    dict
        Dictionary containing NDVI, NIRv, and EVI2 indices
    """
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
        nirv = ndvi * nir
        evi2 = 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)
    
    # Replace inf and nan with 0
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    nirv = np.nan_to_num(nirv, nan=0.0, posinf=0.0, neginf=0.0)
    evi2 = np.nan_to_num(evi2, nan=0.0, posinf=0.0, neginf=0.0)
    
    return {
        'NDVI': ndvi,
        'NIRv': nirv,
        'EVI2': evi2
    }


def compute_multi_angle_features(df):
    """
    Compute multi-angle corrected reflectance and indices.
    
    This function generates BRDF-corrected features across different
    viewing geometries (VZA: 0-60°, SZA: 0-60°).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing BRDF parameters and reflectance data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added multi-angle features
    """
    print("Computing multi-angle corrected features...")
    
    vza_steps = np.arange(0, 61, 5)
    sza_steps = np.arange(0, 61, 5)
    
    for v in tqdm(vza_steps, desc="VZA"):
        for s in sza_steps:
            # Compute BRDF-corrected reflectance
            m_red = brdf_degree(s, v, df['raa'], df['iso_r'], df['vol_r'], df['geo_r'])
            m_nir = brdf_degree(s, v, df['raa'], df['iso_n'], df['vol_n'], df['geo_n'])
            
            # Compute vegetation indices
            indices = compute_vegetation_indices(m_red, m_nir)
            
            # Store in dataframe
            df[f'M-RED-v{v}-s{s}'] = m_red.astype(np.float16)
            df[f'M-NIR-v{v}-s{s}'] = m_nir.astype(np.float16)
            df[f'M-NDVI-v{v}-s{s}'] = indices['NDVI'].astype(np.float16)
            df[f'M-NIRv-v{v}-s{s}'] = indices['NIRv'].astype(np.float16)
            df[f'M-EVI2-v{v}-s{s}'] = indices['EVI2'].astype(np.float16)
    
    return df


# ============================================================================
# 4. Temporal Aggregation Functions
# ============================================================================

def get_temporal_index(date_series, method='dekad'):
    """
    Generate temporal aggregation index for dates.
    
    Parameters
    ----------
    date_series : pd.Series
        Series of datetime objects
    method : str, default='dekad'
        Aggregation method: 'dekad' (10-day periods) or 'month'
    
    Returns
    -------
    pd.Series
        Temporal indices (e.g., '03-1', '03-2', '03-3' for March dekads)
    """
    if method == 'dekad':
        day = date_series.dt.day
        dekad = np.select(
            [day <= 10, (day > 10) & (day <= 20), day > 20],
            [1, 2, 3],
            default=0
        )
        return date_series.dt.month.astype(str).str.zfill(2) + '-' + dekad.astype(str)
    else:
        return date_series.dt.month.astype(str).str.zfill(2)


def aggregate_temporal_data(df, aggregation_method='dekad'):
    """
    Aggregate features and SIF data by temporal periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series data
    aggregation_method : str
        Method for temporal aggregation
    
    Returns
    -------
    pd.DataFrame
        Aggregated time series data
    """
    print("Aggregating data by temporal periods...")
    
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['time_index'] = get_temporal_index(df['date'], aggregation_method)
    
    # Group by sample_id, year, and time_index
    agg_funcs = {col: 'mean' for col in df.columns if col not in 
                 ['sample_id', 'year', 'time_index', 'yield', 'date']}
    
    aggregated = df.groupby(['sample_id', 'year', 'time_index']).agg(agg_funcs).reset_index()
    
    # Preserve yield value
    yield_map = df[['sample_id', 'year', 'yield']].drop_duplicates()
    aggregated = aggregated.merge(yield_map, on=['sample_id', 'year'], how='left')
    
    return aggregated


# ============================================================================
# 5. Main Preprocessing Pipeline
# ============================================================================

def main():
    """Execute the complete data preprocessing pipeline"""
    
    print("="*70)
    print("CROP YIELD PREDICTION - DATA PREPROCESSING PIPELINE")
    print("="*70)
    
    # Step 1: Load raw data
    print("\nStep 1: Loading raw data...")
    try:
        df_sif = pd.read_csv(SIF_CSV_PATH)
        df_vis = pd.read_csv(VIS_CSV_PATH)
        df_yield = pd.read_csv(YIELD_CSV_PATH)
        print(f"  ✓ SIF data: {df_sif.shape}")
        print(f"  ✓ VIS data: {df_vis.shape}")
        print(f"  ✓ Yield data: {df_yield.shape}")
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        print("  Please ensure raw data files exist in ./data/raw/")
        return
    
    # Step 2: Merge datasets
    print("\nStep 2: Merging datasets...")
    df = df_sif.merge(df_vis, on=['sample_id', 'year', 'month', 'day'], how='inner')
    df = df.merge(df_yield, on=['sample_id', 'year'], how='inner')
    print(f"  ✓ Merged data: {df.shape}")
    
    # Step 3: Filter date range
    print(f"\nStep 3: Filtering date range ({START_DATE} to {END_DATE})...")
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    month_day_str = df['date'].dt.strftime('%m-%d')
    df = df[(month_day_str >= START_DATE) & (month_day_str <= END_DATE)]
    print(f"  ✓ Filtered data: {df.shape}")
    
    # Step 4: Compute multi-angle features (optional, removes for computational efficiency)
    # Uncomment the line below to include multi-angle BRDF corrections
    # df = compute_multi_angle_features(df)
    
    # Step 5: Aggregate temporal data
    print(f"\nStep 4: Aggregating by {AGGREGATION_METHOD}...")
    df_agg = aggregate_temporal_data(df, AGGREGATION_METHOD)
    print(f"  ✓ Aggregated data: {df_agg.shape}")
    
    # Step 6: Pivot to feature matrix
    print("\nStep 5: Pivoting to feature matrix...")
    df_pivot = df_agg.pivot(index=['sample_id', 'year'], columns='time_index')
    
    # Flatten column names
    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    print(f"  ✓ Feature matrix shape: {df_pivot.shape}")
    
    # Step 7: Save results
    print("\nStep 6: Saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_pivot.to_csv(FINAL_FEATURES_CSV, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved to: {FINAL_FEATURES_CSV}")
    
    with open(FINAL_FEATURES_PKL, 'wb') as f:
        pickle.dump(df_pivot, f)
    print(f"  ✓ Saved to: {FINAL_FEATURES_PKL}")
    
    print("\n" + "="*70)
    print("DATA PREPROCESSING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
