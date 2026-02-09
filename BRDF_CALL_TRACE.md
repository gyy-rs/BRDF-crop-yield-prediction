# BRDF Function Call Tracing & Usage Map

This document traces how BRDF correction functions are called throughout the codebase and shows where they integrate with other modules.

## Call Graph Overview

```
┌─────────────────────────────────────────────────────────────┐
│  User Code / Data Preprocessing                              │
└────────────────┬────────────────────────────────────────────┘
                 │
       ┌─────────┴──────────┐
       │                    │
       ▼                    ▼
┌──────────────────┐  ┌──────────────────────────────┐
│ brdf_correction  │  │ apply_multi_angle_correction │
└──────────────────┘  └──────────────────────────────┘
       │                    │
       ├─────────┬──────────┤
       │         │          │
       ▼         ▼          ▼
┌─────────────────────────────────────┐
│ validate_brdf_inputs                │
│ ross_thick_kernel                   │
│ li_sparse_kernel                    │
└─────────────────────────────────────┘
```

## Function Hierarchy

### Level 1: User-Facing Functions

#### `brdf_correction()`
```
Input:  sun_zenith, view_zenith, relative_azimuth
        iso_coefficient, vol_coefficient, geo_coefficient
        [return_kernels: bool]

Output: corrected_reflectance
        [optional: ross_kernel, li_kernel]

Calls:  validate_brdf_inputs()
        ross_thick_kernel()
        li_sparse_kernel()

Used In:
- Data preprocessing pipeline
- Multi-angle correction
- Standalone reflectance correction
```

#### `apply_multi_angle_correction()`
```
Input:  DataFrame with angle and BRDF coefficient columns
        view_zenith_steps: list
        sun_zenith_steps: list
        [verbose: bool]

Output: DataFrame with added multi-angle columns

Calls:  brdf_correction() [169 times for default parameters]
        validate_brdf_inputs() [indirectly]

Used In:
- Data preprocessing pipeline
- Advanced multi-angle analysis
```

### Level 2: Validation & Kernels

#### `validate_brdf_inputs()`
```
Input:  All 6 input arrays/series

Output: 6 validated numpy arrays

Called By:
- brdf_correction() [direct]
- apply_multi_angle_correction() [indirect]
```

#### `ross_thick_kernel()`
```
Input:  sun_zenith_rad, view_zenith_rad, relative_azimuth_rad
        (all in radians)

Output: ross_kernel values

Called By:
- brdf_correction() [direct]

Details:
- @njit(fastmath=True, cache=True, parallel=True)
- Vectorized over arrays
- No loops in Python layer
```

#### `li_sparse_kernel()`
```
Input:  sun_zenith_rad, view_zenith_rad, relative_azimuth_rad
        (all in radians)

Output: li_kernel values

Called By:
- brdf_correction() [direct]

Details:
- @njit(fastmath=True, cache=True, parallel=True)
- Vectorized over arrays
- Handles edge cases (azimuth wrapping)
```

## Integration Points with Other Modules

### With `data_preprocessing.py`

**Current Integration** (Optional):
```python
# In data_preprocessing.py, line ~95
from src.brdf_correction import apply_multi_angle_correction

def process_sif_dataframe(df):
    """Apply multi-angle BRDF correction"""
    print("-> Applying BRDF corrections...")
    
    df = apply_multi_angle_correction(
        df,
        view_zenith_steps=[0, 15, 30, 45, 60],
        sun_zenith_steps=[20, 30, 40, 50, 60],
        verbose=True
    )
    
    # Add generated columns to aggregation list
    # ...
```

**Data Flow**:
```
Raw SIF Data (TROPOMI)
        ↓
  [BRDF Correction]  ← apply_multi_angle_correction()
        ↓
  Merge with VIs
        ↓
Temporal Aggregation
        ↓
Feature Matrix
```

### With `model.py`

**Indirect Integration** (through preprocessed features):
```python
# model.py uses features from apply_multi_angle_correction()

# Example: If multi-angle correction is applied, features might include:
# - Original: 'sif743', 'NDVI', 'NIRv'
# - Multi-angle: 'angle_v00_s30_SIF', 'angle_v30_s30_SIF', ...
#                'angle_v00_s30_NDVI', 'angle_v30_s30_NDVI', ...

# The model automatically uses all available features
model = AttentionLSTMModel(
    input_size=n_features,  # Increased if multi-angle enabled
    hidden_size=64,
    n_heads=4,
    output_size=1
)
```

### With `train.py`

**Indirect Integration** (through data loader):
```python
# train.py receives preprocessed features

# If multi-angle correction was applied:
# Features include angle-corrected SIF and vegetation indices
# Model learns patterns across different viewing geometries

# Validation strategy remains the same (10x5-fold CV)
# But now with richer feature set
```

## Current Usage in Original Code

### File 1: `1 预处理合并 SIF 和 VIs、产量.py`

**Location**: Lines 10, 120-160
```python
# Line 10: Import
from utils.Make_mat_vectorized import BRDF_degree_vectorized as BRDF_degree

# Lines 118-140: Basic correction at original geometry
red_original = BRDF_degree(df['sza'], df['vza'], df['raa'], 
                          df['iso_r'], df['vol_r'], df['geo_r'])

# Lines 140-165: Multi-angle loop
for v in VZA_steps:
    for s in SZA_steps:
        m_red = BRDF_degree(sza_series, vza_series, df['raa'],
                           df['iso_r'], df['vol_r'], df['geo_r'])
        # ... compute indices and ratios
```

**Mapping to New Module**:
```python
# OLD CODE:
BRDF_degree(df['sza'], df['vza'], df['raa'], 
            df['iso_r'], df['vol_r'], df['geo_r'])

# NEW CODE:
brdf_correction(df['sza'], df['vza'], df['raa'],
                df['iso_r'], df['vol_r'], df['geo_r'])

# OLD CODE (multi-angle loop):
for v in VZA_steps:
    for s in SZA_steps:
        m_red = BRDF_degree(...)

# NEW CODE (simplified):
df = apply_multi_angle_correction(
    df,
    view_zenith_steps=VZA_steps,
    sun_zenith_steps=SZA_steps
)
# Already has all multi-angle columns!
```

### File 2: `@V1数据合并.py`

**Location**: Lines 13-14
```python
try:
    from utils.Make_mat_vectorized import BRDF_degree_vectorized as BRDF_degree
    print("-> 成功加载 'utils.Make_mat_vectorized.BRDF_degree_vectorized'")
except:
    print("-> 未找到 'utils.Make_mat_vectorized'，使用模拟 BRDF 函数。")
```

**Usage**: Similar to File 1, uses BRDF_degree for multi-angle correction

## Call Trace Example

### Example 1: Basic BRDF Correction

```
User Code:
  red_corrected = brdf_correction(
    df['sza'], df['vza'], df['raa'],
    df['iso_r'], df['vol_r'], df['geo_r']
  )
      ↓
  validate_brdf_inputs()
    - Convert to numpy arrays
    - Check NaN
    - Validate shapes
      ↓
  Convert angles to radians
      ↓
  ross_thick_kernel(sza_rad, vza_rad, raa_rad)
    - Calculate phase angle
    - Apply Ross formula
    - Return k values (~10 ms)
      ↓
  li_sparse_kernel(sza_rad, vza_rad, raa_rad)
    - Normalize azimuth
    - Calculate projected angles
    - Apply Li formula
    - Return k values (~10 ms)
      ↓
  Linear combination:
    BRDF = iso + vol*k_ross + geo*k_li
      ↓
  Replace inf with nan
      ↓
  Return corrected values
```

### Example 2: Multi-Angle Correction

```
User Code:
  df = apply_multi_angle_correction(
    df_tropomi,
    view_zenith_steps=[0, 30, 60],
    sun_zenith_steps=[30, 50]
  )
      ↓
  Calculate reference (original geometry):
    - brdf_correction(...) × 2 (red, NIR)
    - Calculate NDVI, NIRv from reference
      ↓
  For each (VZA, SZA) combination (6 total):
    ├─ Create angle series
    ├─ brdf_correction() for red → red_corrected
    ├─ brdf_correction() for NIR → nir_corrected
    ├─ Calculate NDVI, NIRv, EVI2
    ├─ Calculate ratio (corrected/reference)
    ├─ Apply ratio to SIF
    └─ Store 6 columns in dataframe
      ↓
  Return dataframe with 6×6=36 new columns
```

## Performance Characteristics

### Time Complexity

```
Operation                    | O(n) where n = number of samples
─────────────────────────────────────────────────────
brdf_correction()            | O(n)
  validate_brdf_inputs()     | O(n)
  ross_thick_kernel()        | O(n) [Numba JIT]
  li_sparse_kernel()         | O(n) [Numba JIT]
  
apply_multi_angle_correction | O(n × n_angles)
  ├─ For each angle: brdf_correction()
  ├─ Plus: aggregation to df
```

### Space Complexity

```
brdf_correction()            | O(n) for output arrays
apply_multi_angle_correction | O(n × n_new_cols)
                             | Default: n × 1014
```

## Memory Usage Example

```
For 1000 TROPOMI observations:

brdf_correction():
  Input: 6 arrays × 1000 × 4 bytes = 24 KB
  Kernels: 2 arrays × 1000 × 8 bytes = 16 KB
  Output: 1 array × 1000 × 8 bytes = 8 KB
  Total: ~50 KB

apply_multi_angle_correction():
  Input DataFrame: 1000 × 15 columns ≈ 60 KB
  New columns: 1000 × 1014 × 2 bytes (float16) ≈ 2 MB
  Peak memory: ~3-5 MB
```

## Call Order in Data Pipeline

```
1. Load raw SIF data
   └─ Format: 100k rows × 12 columns (includes angles, BRDF coef)

2. [Optional] apply_multi_angle_correction()
   └─ Calls brdf_correction() 169 times
   └─ Output: 100k rows × 200+ columns

3. Load vegetation indices data
   └─ Format: 100k rows × 20 columns (VI + metadata)

4. Merge on sample_id, date

5. Aggregate into dekads
   └─ Uses all available columns from step 2

6. Pivot to time-series matrix
   └─ Shape: n_samples × (n_dekads × n_features)

7. Input to LSTM model
   └─ If multi-angle: ~1000+ features per timestep
   └─ If not: ~50 features per timestep
```

## Debug Tracing

### Enable Tracing in Your Code

```python
from src.brdf_correction import brdf_correction, apply_multi_angle_correction

# Trace 1: Check inputs
import pandas as pd
sif_df = pd.read_csv('raw_sif_data.csv')

print(f"Loaded {len(sif_df)} observations")
print(f"Columns: {list(sif_df.columns)}")
print(f"Angle ranges:")
print(f"  SZA: {sif_df['sza'].min():.1f}° - {sif_df['sza'].max():.1f}°")
print(f"  VZA: {sif_df['vza'].min():.1f}° - {sif_df['vza'].max():.1f}°")

# Trace 2: Apply correction
red_corrected = brdf_correction(
    sif_df['sza'], sif_df['vza'], sif_df['raa'],
    sif_df['iso_r'], sif_df['vol_r'], sif_df['geo_r'],
    return_kernels=True  # Inspect kernels
)
corrected, ross_k, li_k = red_corrected

print(f"Kernel ranges:")
print(f"  Ross: {ross_k.min():.3f} - {ross_k.max():.3f}")
print(f"  Li: {li_k.min():.3f} - {li_k.max():.3f}")

# Trace 3: Multi-angle correction
sif_df = apply_multi_angle_correction(
    sif_df,
    view_zenith_steps=[0, 20, 40],
    sun_zenith_steps=[30, 50],
    verbose=True  # Shows progress
)

print(f"Output shape: {sif_df.shape}")
print(f"New columns: {len(sif_df.columns) - 12}")
```

## Optimization Opportunities

### Potential Improvements

1. **Batch GPU Processing**
   - Use CuPy instead of NumPy for GPU acceleration
   - Suitable for 100k+ observations

2. **Caching**
   - Cache kernel values for repeated angle combinations
   - Reduces redundant computation in loops

3. **Parallel Multi-Angle**
   - Process angle combinations in parallel
   - Use multiprocessing for each (VZA, SZA) pair

4. **Memory Optimization**
   - Use float16 for intermediate results
   - Already implemented in apply_multi_angle_correction()

---

**Document Version**: 1.0
**Last Updated**: February 2024
**Status**: Complete
