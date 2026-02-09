"""
BRDF Correction Example Script

This script demonstrates how to apply BRDF corrections to TROPOMI SIF data.
It shows:
1. Loading raw TROPOMI data
2. Basic BRDF correction
3. Multi-angle correction
4. Integration with vegetation indices computation

Usage:
    python examples/brdf_correction_example.py

Author: Research Team
Date: 2024
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.brdf_correction import (
    brdf_correction,
    apply_multi_angle_correction,
    validate_brdf_inputs,
    ross_thick_kernel,
    li_sparse_kernel
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def example_1_basic_correction():
    """
    Example 1: Basic BRDF correction on sample data
    
    This demonstrates the simplest use case: correcting reflectance
    values for a small dataset using original geometry.
    """
    print_section("Example 1: Basic BRDF Correction")
    
    # Create sample TROPOMI data
    print("\n1. Creating sample TROPOMI data...")
    sample_data = pd.DataFrame({
        'observation_id': range(5),
        'sza': [25.0, 30.0, 35.0, 40.0, 45.0],  # Solar zenith angle
        'vza': [20.0, 25.0, 30.0, 35.0, 40.0],  # Viewing zenith angle
        'raa': [120.0, 130.0, 140.0, 150.0, 160.0],  # Relative azimuth
        'iso_r': [0.05, 0.055, 0.06, 0.065, 0.07],  # Red isotropic coef
        'vol_r': [0.10, 0.105, 0.11, 0.115, 0.12],  # Red volumetric coef
        'geo_r': [0.02, 0.021, 0.022, 0.023, 0.024],  # Red geometric coef
        'iso_n': [0.08, 0.085, 0.09, 0.095, 0.10],  # NIR isotropic coef
        'vol_n': [0.25, 0.26, 0.27, 0.28, 0.29],  # NIR volumetric coef
        'geo_n': [0.05, 0.052, 0.054, 0.056, 0.058],  # NIR geometric coef
        'sif743': [1.2, 1.3, 1.4, 1.5, 1.6],  # SIF at 743 nm
    })
    
    print(f"   Created {len(sample_data)} sample observations")
    print(f"   Columns: {list(sample_data.columns)}")
    
    # Apply BRDF correction to red band
    print("\n2. Applying BRDF correction to red band...")
    red_corrected = brdf_correction(
        sun_zenith=sample_data['sza'],
        view_zenith=sample_data['vza'],
        relative_azimuth=sample_data['raa'],
        iso_coefficient=sample_data['iso_r'],
        vol_coefficient=sample_data['vol_r'],
        geo_coefficient=sample_data['geo_r']
    )
    
    sample_data['red_corrected'] = red_corrected
    print(f"   Red reflectance range: [{red_corrected.min():.4f}, {red_corrected.max():.4f}]")
    
    # Apply BRDF correction to NIR band
    print("\n3. Applying BRDF correction to NIR band...")
    nir_corrected = brdf_correction(
        sun_zenith=sample_data['sza'],
        view_zenith=sample_data['vza'],
        relative_azimuth=sample_data['raa'],
        iso_coefficient=sample_data['iso_n'],
        vol_coefficient=sample_data['vol_n'],
        geo_coefficient=sample_data['geo_n']
    )
    
    sample_data['nir_corrected'] = nir_corrected
    print(f"   NIR reflectance range: [{nir_corrected.min():.4f}, {nir_corrected.max():.4f}]")
    
    # Calculate vegetation indices
    print("\n4. Computing vegetation indices...")
    sample_data['ndvi'] = (nir_corrected - red_corrected) / (nir_corrected + red_corrected + 1e-9)
    sample_data['nirv'] = sample_data['ndvi'] * nir_corrected
    sample_data['evi2'] = (2.5 * (nir_corrected - red_corrected) / 
                          (nir_corrected + 2.4 * red_corrected + 1.0))
    
    print(f"   NDVI range: [{sample_data['ndvi'].min():.4f}, {sample_data['ndvi'].max():.4f}]")
    print(f"   NIRv range: [{sample_data['nirv'].min():.4f}, {sample_data['nirv'].max():.4f}]")
    print(f"   EVI2 range: [{sample_data['evi2'].min():.4f}, {sample_data['evi2'].max():.4f}]")
    
    # Display results
    print("\n5. Results:")
    display_cols = ['observation_id', 'sza', 'vza', 'red_corrected', 
                   'nir_corrected', 'ndvi', 'nirv']
    print(sample_data[display_cols].to_string(index=False))
    
    return sample_data


def example_2_kernel_inspection():
    """
    Example 2: Inspect BRDF kernel values
    
    This demonstrates how to examine the individual kernel contributions
    to understand the BRDF behavior.
    """
    print_section("Example 2: BRDF Kernel Inspection")
    
    # Create sample data with different geometries
    print("\n1. Creating sample data with varying angles...")
    angles = np.linspace(10, 60, 6)
    sample_data = pd.DataFrame({
        'geometry': [f"sza={sza:.0f}° vza={vza:.0f}°" 
                    for sza, vza in zip(angles, angles[::-1])],
        'sza': angles,
        'vza': angles[::-1],
        'raa': np.full_like(angles, 120.0),
        'iso_r': np.full_like(angles, 0.06),
        'vol_r': np.full_like(angles, 0.11),
        'geo_r': np.full_like(angles, 0.022),
    })
    
    # Get kernels
    print("\n2. Computing BRDF kernels...")
    corrected, ross_k, li_k = brdf_correction(
        sample_data['sza'],
        sample_data['vza'],
        sample_data['raa'],
        sample_data['iso_r'],
        sample_data['vol_r'],
        sample_data['geo_r'],
        return_kernels=True
    )
    
    # Create results dataframe
    results = pd.DataFrame({
        'Geometry': sample_data['geometry'],
        'SZA': sample_data['sza'],
        'VZA': sample_data['vza'],
        'Ross_Kernel': np.around(ross_k, 4),
        'Li_Kernel': np.around(li_k, 4),
        'BRDF': np.around(corrected, 4),
    })
    
    print("\n3. Kernel Values:")
    print(results.to_string(index=False))
    
    # Analyze kernel behavior
    print("\n4. Kernel Analysis:")
    print(f"   Ross kernel range: [{ross_k.min():.4f}, {ross_k.max():.4f}]")
    print(f"   Li kernel range: [{li_k.min():.4f}, {li_k.max():.4f}]")
    print(f"   BRDF range: [{corrected.min():.4f}, {corrected.max():.4f}]")
    
    # Calculate contributions
    iso_contrib = sample_data['iso_r'].mean()
    vol_contrib = (sample_data['vol_r'].mean() * ross_k).mean()
    geo_contrib = (sample_data['geo_r'].mean() * li_k).mean()
    total = iso_contrib + vol_contrib + geo_contrib
    
    print(f"\n   Kernel Contributions to BRDF:")
    print(f"   - Isotropic: {iso_contrib:.4f} ({100*iso_contrib/total:.1f}%)")
    print(f"   - Volumetric: {vol_contrib:.4f} ({100*vol_contrib/total:.1f}%)")
    print(f"   - Geometric: {geo_contrib:.4f} ({100*geo_contrib/total:.1f}%)")
    
    return results


def example_3_multi_angle_correction():
    """
    Example 3: Multi-angle correction
    
    This demonstrates generating SIF and vegetation indices at different
    viewing angles to characterize angular dependencies.
    """
    print_section("Example 3: Multi-Angle BRDF Correction")
    
    # Create sample TROPOMI data
    print("\n1. Creating sample TROPOMI data...")
    np.random.seed(42)
    n_samples = 3
    
    sample_data = pd.DataFrame({
        'sample_id': range(n_samples),
        'sza': np.random.uniform(25, 45, n_samples),
        'vza': np.random.uniform(20, 40, n_samples),
        'raa': np.random.uniform(100, 160, n_samples),
        'iso_r': np.random.uniform(0.05, 0.07, n_samples),
        'vol_r': np.random.uniform(0.10, 0.12, n_samples),
        'geo_r': np.random.uniform(0.02, 0.025, n_samples),
        'iso_n': np.random.uniform(0.08, 0.10, n_samples),
        'vol_n': np.random.uniform(0.25, 0.30, n_samples),
        'geo_n': np.random.uniform(0.05, 0.06, n_samples),
        'sif743': np.random.uniform(1.0, 2.0, n_samples),
    })
    
    print(f"   Created {len(sample_data)} samples")
    
    # Apply multi-angle correction with limited angles for speed
    print("\n2. Applying multi-angle BRDF correction...")
    print("   Viewing angles: 0°, 20°, 40°, 60°")
    print("   Solar angles: 30°, 45°, 60°")
    
    df_corrected = apply_multi_angle_correction(
        sample_data,
        view_zenith_steps=[0, 20, 40, 60],
        sun_zenith_steps=[30, 45, 60],
        verbose=True
    )
    
    # Show generated columns
    print("\n3. Generated columns:")
    new_cols = [col for col in df_corrected.columns 
               if col not in sample_data.columns]
    print(f"   Total new columns: {len(new_cols)}")
    print(f"   Examples:")
    for col in new_cols[:9]:
        print(f"   - {col}")
    print(f"   ... and {len(new_cols)-9} more")
    
    # Sample some results
    print("\n4. Sample Results (first sample):")
    sample_idx = 0
    print(f"   Original geometry: SZA={sample_data.loc[sample_idx, 'sza']:.1f}° VZA={sample_data.loc[sample_idx, 'vza']:.1f}°")
    print(f"   Original SIF: {sample_data.loc[sample_idx, 'sif743']:.3f}")
    print(f"\n   Corrected to different viewing angles:")
    
    for vza in [0, 20, 40]:
        sza = 30
        col = f"angle_v{vza:02d}_s{sza:02d}_SIF"
        if col in df_corrected.columns:
            sif_corrected = df_corrected.loc[sample_idx, col]
            print(f"   - VZA={vza}° SZA={sza}°: {sif_corrected:.3f}")
    
    return df_corrected


def example_4_validation():
    """
    Example 4: Input validation
    
    This demonstrates how to validate input data before BRDF correction.
    """
    print_section("Example 4: Input Data Validation")
    
    # Create sample data with potential issues
    print("\n1. Creating sample data...")
    sample_data = pd.DataFrame({
        'sza': [25.0, 30.0, np.nan, 40.0],  # Contains NaN
        'vza': [20.0, 25.0, 30.0, 35.0],
        'raa': [120.0, 130.0, 140.0, 150.0],
        'iso_r': [0.05, 0.055, 0.06, 0.065],
        'vol_r': [0.10, 0.105, 0.11, 0.115],
        'geo_r': [0.02, 0.021, 0.022, 0.023],
    })
    
    print(f"   Created {len(sample_data)} samples")
    print(f"   Note: Contains 1 NaN value in 'sza' column")
    
    # Try validation
    print("\n2. Validating input data...")
    try:
        sza, vza, raa, iso, vol, geo = validate_brdf_inputs(
            sample_data['sza'],
            sample_data['vza'],
            sample_data['raa'],
            sample_data['iso_r'],
            sample_data['vol_r'],
            sample_data['geo_r']
        )
        print("   ✓ Validation passed (NaN handled automatically)")
        print(f"   Converted to numpy arrays")
        print(f"   Data type: {sza.dtype}")
    except ValueError as e:
        print(f"   ✗ Validation failed: {e}")
    
    # Clean data
    print("\n3. Cleaning data...")
    sample_data_clean = sample_data.dropna()
    print(f"   Removed rows with NaN: {len(sample_data)} → {len(sample_data_clean)} rows")
    
    # Validate cleaned data
    print("\n4. Validating cleaned data...")
    try:
        sza, vza, raa, iso, vol, geo = validate_brdf_inputs(
            sample_data_clean['sza'],
            sample_data_clean['vza'],
            sample_data_clean['raa'],
            sample_data_clean['iso_r'],
            sample_data_clean['vol_r'],
            sample_data_clean['geo_r']
        )
        print("   ✓ Validation passed")
        print(f"   Ready for BRDF correction")
    except ValueError as e:
        print(f"   ✗ Validation failed: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  BRDF Correction Module - Examples")
    print("="*70)
    print("\nThis script demonstrates BRDF correction for TROPOMI SIF data")
    print("Location: examples/brdf_correction_example.py")
    
    # Run examples
    try:
        df1 = example_1_basic_correction()
        df2 = example_2_kernel_inspection()
        df3 = example_3_multi_angle_correction()
        example_4_validation()
        
        # Final summary
        print_section("Summary")
        print("\n✓ All examples completed successfully!")
        print("\nWhat you learned:")
        print("  1. Basic BRDF correction on small datasets")
        print("  2. Understanding BRDF kernel contributions")
        print("  3. Multi-angle correction for angular characterization")
        print("  4. Input data validation")
        
        print("\nNext steps:")
        print("  - Read docs/BRDF_GUIDE.md for detailed documentation")
        print("  - Check src/brdf_correction.py for function signatures")
        print("  - Use apply_multi_angle_correction for full processing")
        
    except Exception as e:
        print(f"\n✗ Error during examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*70)
    return 0


if __name__ == "__main__":
    exit(main())
