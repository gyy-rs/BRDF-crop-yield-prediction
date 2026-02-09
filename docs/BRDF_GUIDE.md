# BRDF Correction Guide

## Overview

The BRDF (Bidirectional Reflectance Distribution Function) correction module corrects TROPOMI SIF observations for angular effects caused by different sun-sensor geometries.

**Why BRDF Correction?**
- TROPOMI observations vary with sun and view angles
- Same surface can appear different depending on measurement geometry
- BRDF correction removes these geometric artifacts
- Enables meaningful comparisons across different observation angles

## Theory

### The BRDF Model

The kernel-driven BRDF model expresses the bidirectional reflectance as:

$$\text{BRDF} = f_0 + k_{\text{vol}} \cdot K_{\text{vol}} + k_{\text{geo}} \cdot K_{\text{geo}}$$

Where:
- $f_0$: Isotropic reflectance (angle-independent)
- $k_{\text{vol}}$: Volumetric scattering coefficient
- $k_{\text{geo}}$: Geometric scattering coefficient
- $K_{\text{vol}}$: Ross-thick kernel (canopy scattering)
- $K_{\text{geo}}$: Li-sparse kernel (shadow effects)

### Kernels

**Ross-thick Kernel** (Volumetric Scattering)
- Models light scattering within vegetation canopy
- Based on single scattering theory
- Increases with phase angle
- Reference: Ross (1981)

**Li-sparse Kernel** (Geometric Scattering)
- Models shadowing effects from vegetation elements
- Based on geometric-optical theory
- Accounts for mutual shadowing
- Reference: Li & Strahler (1986)

## Installation

```bash
# The brdf_correction module is part of the package
# No additional installation needed beyond requirements.txt
pip install -r requirements.txt
```

## Basic Usage

### Single Observation Correction

```python
import pandas as pd
from src.brdf_correction import brdf_correction

# TROPOMI data (single observation)
sun_zenith = 30.0  # degrees
view_zenith = 25.0  # degrees
relative_azimuth = 120.0  # degrees

# BRDF coefficients
iso_r = 0.05  # Isotropic reflectance for red band
vol_r = 0.10  # Volumetric coefficient for red band
geo_r = 0.02  # Geometric coefficient for red band

# Perform correction
red_corrected = brdf_correction(
    sun_zenith=sun_zenith,
    view_zenith=view_zenith,
    relative_azimuth=relative_azimuth,
    iso_coefficient=iso_r,
    vol_coefficient=vol_r,
    geo_coefficient=geo_r
)

print(f"Corrected red reflectance: {red_corrected}")
```

### Batch Processing (DataFrame)

```python
import pandas as pd
from src.brdf_correction import brdf_correction

# Load TROPOMI data
df = pd.read_csv('tropomi_data.csv')

# Apply BRDF correction to entire dataset
red_corrected = brdf_correction(
    sun_zenith=df['sza'],
    view_zenith=df['vza'],
    relative_azimuth=df['raa'],
    iso_coefficient=df['iso_r'],
    vol_coefficient=df['vol_r'],
    geo_coefficient=df['geo_r']
)

# Store results
df['red_corrected'] = red_corrected
```

### Multi-Band Correction

```python
from src.brdf_correction import brdf_correction

# Correct both red and NIR bands
red_corrected = brdf_correction(
    df['sza'], df['vza'], df['raa'],
    df['iso_r'], df['vol_r'], df['geo_r']
)

nir_corrected = brdf_correction(
    df['sza'], df['vza'], df['raa'],
    df['iso_n'], df['vol_n'], df['geo_n']
)

# Calculate NDVI
ndvi = (nir_corrected - red_corrected) / (nir_corrected + red_corrected)
```

### Kernel Inspection

```python
from src.brdf_correction import brdf_correction

# Get kernel values for analysis
corrected, ross_kernel, li_kernel = brdf_correction(
    df['sza'], df['vza'], df['raa'],
    df['iso_r'], df['vol_r'], df['geo_r'],
    return_kernels=True
)

print(f"Ross kernel range: [{ross_kernel.min():.3f}, {ross_kernel.max():.3f}]")
print(f"Li kernel range: [{li_kernel.min():.3f}, {li_kernel.max():.3f}]")
```

## Advanced Usage

### Multi-Angle Correction

Generate SIF and vegetation indices across different viewing angles to characterize angular dependencies:

```python
from src.brdf_correction import apply_multi_angle_correction

# Specify viewing angles to generate
vza_angles = [0, 15, 30, 45, 60]  # degrees
sza_angles = [20, 30, 40, 50]  # degrees

df_corrected = apply_multi_angle_correction(
    df_tropomi,
    view_zenith_steps=vza_angles,
    sun_zenith_steps=sza_angles,
    verbose=True
)

# New columns like:
# - angle_v00_s20_RED, angle_v00_s20_NIR, angle_v00_s20_NDVI, etc.
# - angle_v15_s20_RED, angle_v15_s20_NIR, angle_v15_s20_NDVI, etc.
# - ... (for each angle combination)

print(f"Output shape: {df_corrected.shape}")
print(f"New columns: {len(df_corrected.columns) - len(df_tropomi.columns)} added")
```

### Integration with Preprocessing Pipeline

```python
from src.data_preprocessing import DataPreprocessor
from src.brdf_correction import apply_multi_angle_correction

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Load data
df = preprocessor.load_and_prepare_data(
    sif_path='raw_sif_data.csv',
    vis_path='raw_vis_data.csv',
    yield_path='raw_yield_data.csv'
)

# Apply BRDF correction
df = apply_multi_angle_correction(df)

# Continue with aggregation and training
aggregated = preprocessor.aggregate_temporal_data(df)
```

## Input Data Format

### Required Columns for BRDF Correction

```
Column              | Type      | Unit      | Description
--------------------|-----------|-----------|------------------
sza                 | float     | degrees   | Solar zenith angle
vza                 | float     | degrees   | Viewing zenith angle
raa                 | float     | degrees   | Relative azimuth angle

iso_r               | float     | unitless  | Isotropic coefficient (red)
vol_r               | float     | unitless  | Volumetric coefficient (red)
geo_r               | float     | unitless  | Geometric coefficient (red)

iso_n               | float     | unitless  | Isotropic coefficient (NIR)
vol_n               | float     | unitless  | Volumetric coefficient (NIR)
geo_n               | float     | unitless  | Geometric coefficient (NIR)
```

### Example Input Data

```csv
sza,vza,raa,iso_r,vol_r,geo_r,iso_n,vol_n,geo_n,sif743
30.5,25.3,120.2,0.052,0.098,0.021,0.085,0.265,0.052,1.234
31.2,26.1,119.5,0.048,0.102,0.025,0.082,0.272,0.048,1.456
29.8,24.5,121.8,0.055,0.095,0.019,0.088,0.258,0.055,1.345
```

## Output Data Format

### BRDF Correction Output

```python
# Single correction
corrected_reflectance = brdf_correction(...)
# Output: 1D numpy array of corrected reflectance values

# With kernels
corrected, ross_k, li_k = brdf_correction(..., return_kernels=True)
# Output: Three 1D numpy arrays
```

### Multi-Angle Correction Output

```
New columns generated for each (VZA, SZA) combination:
- angle_v{VZA:02d}_s{SZA:02d}_RED     : Red band reflectance
- angle_v{VZA:02d}_s{SZA:02d}_NIR     : NIR band reflectance
- angle_v{VZA:02d}_s{SZA:02d}_NDVI    : NDVI = (NIR-RED)/(NIR+RED)
- angle_v{VZA:02d}_s{SZA:02d}_NIRv    : NIRv = NDVI * NIR
- angle_v{VZA:02d}_s{SZA:02d}_EVI2    : EVI2 = 2.5*(NIR-RED)/(NIR+2.4*RED+1)
- angle_v{VZA:02d}_s{SZA:02d}_SIF     : SIF corrected to viewing angle

Example for VZA=30°, SZA=40°:
- angle_v30_s40_RED
- angle_v30_s40_NIR
- angle_v30_s40_NDVI
- angle_v30_s40_NIRv
- angle_v30_s40_EVI2
- angle_v30_s40_SIF
```

## Complete Workflow Example

```python
"""
Complete BRDF correction workflow for TROPOMI data
"""

import pandas as pd
import numpy as np
from src.brdf_correction import (
    brdf_correction,
    apply_multi_angle_correction,
    validate_brdf_inputs
)

# 1. Load TROPOMI data
print("Loading TROPOMI data...")
df = pd.read_csv('tropomi_sample.csv')
print(f"Loaded {len(df)} observations")

# 2. Validate input data
print("Validating input data...")
try:
    sza, vza, raa, iso_r, vol_r, geo_r = validate_brdf_inputs(
        df['sza'], df['vza'], df['raa'],
        df['iso_r'], df['vol_r'], df['geo_r']
    )
    print("✓ Input validation passed")
except ValueError as e:
    print(f"✗ Input validation failed: {e}")
    exit(1)

# 3. Apply BRDF correction at original geometry
print("Applying BRDF correction at original geometry...")
red_original = brdf_correction(df['sza'], df['vza'], df['raa'],
                              df['iso_r'], df['vol_r'], df['geo_r'])
nir_original = brdf_correction(df['sza'], df['vza'], df['raa'],
                              df['iso_n'], df['vol_n'], df['geo_n'])

df['red_corrected'] = red_original
df['nir_corrected'] = nir_original
df['ndvi_original'] = (nir_original - red_original) / (nir_original + red_original + 1e-9)
print("✓ Original geometry correction completed")

# 4. Apply multi-angle correction (optional, computationally intensive)
print("Applying multi-angle correction...")
df = apply_multi_angle_correction(
    df,
    view_zenith_steps=[0, 15, 30, 45, 60],
    sun_zenith_steps=[30, 40, 50],
    verbose=True
)
print(f"✓ Multi-angle correction completed")
print(f"  Total columns: {df.shape[1]}")

# 5. Save results
print("Saving results...")
df.to_csv('tropomi_brdf_corrected.csv', index=False)
print("✓ Results saved to 'tropomi_brdf_corrected.csv'")

# 6. Summary statistics
print("\nSummary Statistics:")
print(f"Red reflectance: {red_original.mean():.4f} ± {red_original.std():.4f}")
print(f"NIR reflectance: {nir_original.mean():.4f} ± {nir_original.std():.4f}")
print(f"NDVI: {df['ndvi_original'].mean():.4f} ± {df['ndvi_original'].std():.4f}")
print("✓ Workflow completed successfully!")
```

## Performance Considerations

### Computational Cost

| Operation | Time (per 1000 samples) | Memory |
|-----------|----------------------|--------|
| Single band correction | ~10 ms | ~8 KB |
| Multi-band correction | ~20 ms | ~16 KB |
| Multi-angle correction (169 angles) | ~5-10 sec | ~50 MB |

### Optimization Tips

1. **Use NumPy arrays instead of pandas Series when possible**
   ```python
   # Slower
   result = brdf_correction(df['sza'], df['vza'], ...)
   
   # Faster
   result = brdf_correction(df['sza'].values, df['vza'].values, ...)
   ```

2. **Vectorize operations**
   - The module uses Numba JIT compilation for speed
   - Avoid loops over observations

3. **Process in batches**
   ```python
   batch_size = 10000
   for i in range(0, len(df), batch_size):
       batch = df.iloc[i:i+batch_size]
       process_batch(batch)
   ```

4. **Disable multi-angle correction for large datasets**
   ```python
   # Use simple correction instead of multi-angle
   red_corrected = brdf_correction(df['sza'], df['vza'], ...)
   ```

## Troubleshooting

### Issue: NaN values in output

**Cause**: Invalid angle values or missing data

**Solution**:
```python
# Check for NaN in inputs
print(df[['sza', 'vza', 'raa']].isna().sum())

# Fill or remove NaN rows
df = df.dropna(subset=['sza', 'vza', 'raa'])

# Or fill with defaults
df['raa'].fillna(90, inplace=True)
```

### Issue: Corrected values are very large or very small

**Cause**: BRDF coefficients are unrealistic or angles are extreme

**Solution**:
```python
# Check angle ranges
print(f"SZA range: {df['sza'].min():.1f}° - {df['sza'].max():.1f}°")
print(f"VZA range: {df['vza'].min():.1f}° - {df['vza'].max():.1f}°")

# Check coefficient ranges
print(f"Isotropic range: {df['iso_r'].min():.4f} - {df['iso_r'].max():.4f}")
print(f"Volumetric range: {df['vol_r'].min():.4f} - {df['vol_r'].max():.4f}")

# Filter unrealistic values
df = df[(df['sza'] < 70) & (df['vza'] < 70)]
```

### Issue: Memory error with multi-angle correction

**Cause**: Too many angle combinations on large dataset

**Solution**:
```python
# Reduce angle combinations
df = apply_multi_angle_correction(
    df,
    view_zenith_steps=[0, 30, 60],      # Fewer angles
    sun_zenith_steps=[30, 50],           # Fewer angles
)

# Or process in batches
for i in range(0, len(df), 1000):
    batch = df.iloc[i:i+1000]
    batch = apply_multi_angle_correction(batch)
    batch.to_csv(f'batch_{i}.csv', index=False)
```

## References

1. **Kernel Models**
   - Ross, J. K. (1981). The radiation regime and architecture of plant stands. Tasks for vegetation science, 3, 391.
   - Li, X., & Strahler, A. H. (1986). Geometric-optical bidirectional reflectance modeling of the discrete crown vegetation canopy. IEEE Transactions on Geoscience and Remote Sensing, 24(5), 681-695.

2. **BRDF Modeling**
   - Wanner, W., et al. (1995). Global retrieval of bidirectional reflectance and albedo over land from EOS MODIS and MISR data. Journal of Geophysical Research, 102(D14), 17143-17161.
   - Roujean, J. L., Leroy, M., & Deschamps, P. Y. (1992). Bidirectional reflectance function of Earth surfaces. Journal of Geophysical Research, 97(D18), 20455-20468.

3. **TROPOMI SIF**
   - Köhler, P., et al. (2018). Global retrievals of solar-induced chlorophyll fluorescence with TROPOMI. Geophysical Research Letters, 45(5), 1656-1664.

## Citation

If you use this BRDF correction module in your research, please cite:

```bibtex
@software{brdf_correction_2024,
  author = {Research Team},
  title = {BRDF Correction Module for TROPOMI SIF Data},
  year = {2024},
  url = {https://github.com/gyy-rs/BRDF-crop-yield-prediction},
  note = {Part of Crop Yield Prediction Project}
}
```

## Contact

For questions or issues with BRDF correction:
- Create an issue on GitHub
- Contact: your.email@institution.edu

---

**Last Updated**: February 2024
