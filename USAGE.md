# Detailed Usage Guide

This guide provides step-by-step instructions for using the crop yield prediction model.

---

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Running the Pipeline](#running-the-pipeline)
4. [Understanding Results](#understanding-results)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB+ RAM (GPU recommended for faster training)

### Step 1: Clone Repository

```bash
git clone https://github.com/gyy-rs/BRDF-crop-yield-prediction.git
cd BRDF-crop-yield-prediction
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (on Linux/macOS)
source venv/bin/activate

# Activate (on Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import torch, pandas, sklearn; print('✓ All packages installed')"
```

---

## Data Preparation

### Input Data Requirements

Your data must be organized as three CSV files:

#### 1. SIF Data (`raw_sif_data.csv`)

Contains Solar-Induced Fluorescence measurements:

```csv
sample_id,year,month,day,sif743,par,sza,vza,raa,iso_r,vol_r,geo_r,iso_n,vol_n,geo_n
1,2019,3,1,1.25,45.3,35.2,15.0,120.5,0.12,0.08,-0.05,0.18,0.15,-0.02
1,2019,3,2,1.32,46.1,34.8,16.2,118.3,0.13,0.09,-0.04,0.19,0.16,-0.01
```

**Required columns:**
- `sample_id` (int): Unique field identifier
- `year`, `month`, `day` (int): Date
- `sif743` (float): SIF at 743nm (mW/m²/sr/nm)
- `par` (float): Photosynthetically active radiation
- `sza`, `vza`, `raa` (float): Solar/viewing angles (degrees)
- `iso_r`, `vol_r`, `geo_r` (float): Red BRDF parameters
- `iso_n`, `vol_n`, `geo_n` (float): NIR BRDF parameters

#### 2. Vegetation Indices Data (`raw_vis_data.csv`)

Contains spectral indices:

```csv
sample_id,year,month,day,NDVI,NIRv,EVI2,GVMI,CSI,CIredEdge,EVI
1,2019,3,1,0.45,0.38,0.42,0.52,0.35,0.48,0.40
1,2019,3,2,0.48,0.41,0.44,0.55,0.37,0.50,0.42
```

**Required columns:**
- `sample_id`, `year`, `month`, `day` (must match SIF data)
- **At least one** vegetation index column (NDVI, NIRv, EVI2, etc.)

#### 3. Yield Data (`raw_yield_data.csv`)

Contains crop yield target values:

```csv
sample_id,year,yield
1,2019,5200
2,2019,5450
3,2019,5100
```

**Required columns:**
- `sample_id` (int): Unique field identifier
- `year` (int): Harvest year
- `yield` (float): Crop yield in kg/ha

### Data Organization

Place your CSV files in `data/raw/`:

```
crop-yield-prediction/
└── data/
    └── raw/
        ├── raw_sif_data.csv          ← Place here
        ├── raw_vis_data.csv          ← Place here
        └── raw_yield_data.csv        ← Place here
```

### Data Quality Checks

Before running the pipeline, verify your data:

```python
import pandas as pd

# Load and check SIF data
df_sif = pd.read_csv('data/raw/raw_sif_data.csv')
print(f"SIF data shape: {df_sif.shape}")
print(f"Date range: {df_sif['year'].min()}-{df_sif['month'].min()}-{df_sif['day'].min()} "
      f"to {df_sif['year'].max()}-{df_sif['month'].max()}-{df_sif['day'].max()}")
print(f"Samples: {df_sif['sample_id'].nunique()}")

# Check for missing values
print(f"Missing values:\n{df_sif.isnull().sum()}")
```

---

## Running the Pipeline

### Option 1: Quick Test (2 minutes)

Test with included sample data:

```bash
cd src
python train.py
```

This uses pre-prepared sample data in `data/sample/`.

### Option 2: Full Pipeline (With Your Data)

#### Step 1: Prepare Data

Place your CSV files in `data/raw/` as described above.

#### Step 2: Run Preprocessing

```bash
cd src
python data_preprocessing.py
```

**Expected output:**
```
======================================================================
CROP YIELD PREDICTION - DATA PREPROCESSING PIPELINE
======================================================================

Step 1: Loading raw data...
  ✓ SIF data: (50000, 15)
  ✓ VIS data: (50000, 8)
  ✓ Yield data: (500, 2)

Step 2: Merging datasets...
  ✓ Merged data: (50000, 25)

Step 3: Filtering date range (03-01 to 05-30)...
  ✓ Filtered data: (12000, 25)

Step 4: Aggregating by dekad...
  ✓ Aggregated data: (1200, 15)

Step 5: Pivoting to feature matrix...
  ✓ Feature matrix shape: (500, 91)

Step 6: Saving results...
  ✓ Saved to: data/final_yield_features.csv
  ✓ Saved to: data/final_yield_features.pkl

======================================================================
DATA PREPROCESSING COMPLETE
======================================================================
```

**Output files:**
- `data/final_yield_features.pkl` - Preprocessed features (binary)
- `data/final_yield_features.csv` - For inspection (CSV)
- `data/time_index_mapping.csv` - Temporal indices

#### Step 3: Train Model

```bash
python train.py
```

**Expected output:**
```
Loading preprocessed data...
Data loaded: 500 samples, 91 features

Reshaped data: (500, 9, 10)

Cross-validation folds: [████████████████████] 50/50

========================================================
FINAL MODEL PERFORMANCE (Based on CV folds)
========================================================
R² Score:  0.79 ± 0.08
MSE:       15234.50 ± 2145.30
MAE:       89.23
========================================================

Results saved to ./results/
```

---

## Understanding Results

### Cross-Validation Results

The training script outputs:

```
R² Score:  0.79 ± 0.08
MSE:       15234.50 ± 2145.30
MAE:       89.23
```

**Interpretation:**

| Metric | Meaning | Interpretation |
|--------|---------|-----------------|
| **R²** | Coefficient of determination | 79% of yield variance explained by model |
| **R² ± 0.08** | Uncertainty across 50 CV folds | Consistent performance (low ±) is good |
| **MSE** | Mean squared error | Average squared prediction error |
| **MAE** | Mean absolute error | Average prediction error in kg/ha |

**Performance Guidelines:**

| R² Range | Interpretation |
|----------|-----------------|
| 0.90-1.00 | Excellent (unlikely without overfitting) |
| 0.75-0.90 | Very good (typical for yield prediction) |
| 0.60-0.75 | Good (acceptable for decision making) |
| 0.40-0.60 | Fair (needs improvement) |
| <0.40 | Poor (needs more data or features) |

### Detailed Results File

View detailed cross-validation results:

```bash
cat results/cv_results.csv
```

Output (first 5 folds):
```
r2,mse,mae
0.7842,12345.67,87.23
0.7956,11892.34,85.12
0.7658,13456.78,89.45
0.7920,12100.23,86.78
0.8015,11567.89,84.92
...
```

**Analyze results in Python:**

```python
import pandas as pd
import numpy as np

# Load results
results = pd.read_csv('results/cv_results.csv')

# Summary statistics
print(f"Mean R²: {results['r2'].mean():.4f}")
print(f"Std R²:  {results['r2'].std():.4f}")
print(f"Min R²:  {results['r2'].min():.4f}")
print(f"Max R²:  {results['r2'].max():.4f}")

# Identify best and worst folds
print(f"\nBest fold R²: {results['r2'].max():.4f}")
print(f"Worst fold R²: {results['r2'].min():.4f}")
print(f"Variance: {results['r2'].var():.4f}")
```

---

## Advanced Usage

### Customize Model Architecture

Edit `src/train.py`:

```python
# Reduce model size for faster training
LSTM_HIDDEN_DIM = 32      # Default: 64
N_HEADS = 2               # Default: 4
N_LAYERS = 1              # Default: 2
DROPOUT = 0.5             # Default: 0.3

# Adjust training
NUM_EPOCHS = 50           # Default: 100
LEARNING_RATE = 0.005     # Default: 0.001
BATCH_SIZE = 32           # Default: 16
```

### Modify Data Aggregation

Edit `src/data_preprocessing.py`:

```python
# Change temporal aggregation
AGGREGATION_METHOD = 'month'    # Default: 'dekad'
START_DATE = '01-01'            # Default: '03-01'
END_DATE = '12-31'              # Default: '05-30'
```

### Enable BRDF Corrections

In `src/data_preprocessing.py`, uncomment line 150:

```python
# Step 4: Compute multi-angle features
df = compute_multi_angle_features(df)  # ← Uncomment this
```

### Using Different Features

Modify feature columns in `src/data_preprocessing.py`:

```python
# In aggregate_temporal_data(), adjust which columns to aggregate
agg_funcs = {
    'NDVI': 'mean',
    'NIRv': 'mean',
    'EVI2': 'mean',
    # Add or remove features here
}
```

### Changing Cross-Validation Strategy

Edit `src/train.py`:

```python
NUM_FOLDS = 10      # Increase folds for more iterations
N_REPEATS = 5       # Reduce repeats for faster training

# Total iterations = NUM_FOLDS * N_REPEATS
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Cause:** PyTorch not installed

**Solution:**
```bash
pip install torch torchvision torchaudio
```

For GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/raw_sif_data.csv'"

**Cause:** Data files not in correct location

**Solution:**
```bash
# Verify file location
ls -la data/raw/

# Expected output:
# raw_sif_data.csv
# raw_vis_data.csv
# raw_yield_data.csv
```

If files have different names, copy them:
```bash
cp your_sif.csv data/raw/raw_sif_data.csv
cp your_vis.csv data/raw/raw_vis_data.csv
cp your_yield.csv data/raw/raw_yield_data.csv
```

### Issue: CUDA out of memory

**Cause:** Batch size too large for GPU

**Solution:**
```python
# In src/train.py
BATCH_SIZE = 8      # Reduce from 16
```

Or disable GPU:
```python
device = 'cpu'  # Force CPU usage
```

### Issue: Training is extremely slow

**Cause:** Using CPU instead of GPU

**Solution:**
```bash
# Check GPU availability
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# If False, reinstall PyTorch with GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "ValueError: not enough values to unpack"

**Cause:** Data columns don't match expected format

**Solution:**
```bash
# Check column names
python -c "import pandas as pd; print(pd.read_csv('data/raw/raw_sif_data.csv').columns.tolist())"

# Should include:
# ['sample_id', 'year', 'month', 'day', 'sif743', 'par', 'sza', 'vza', 'raa', 
#  'iso_r', 'vol_r', 'geo_r', 'iso_n', 'vol_n', 'geo_n']
```

### Issue: Very low R² scores (< 0.3)

**Causes:**
1. Insufficient temporal coverage
2. Too few samples
3. Feature scaling issues
4. Mismatched sample IDs

**Solutions:**
1. Include more time points (more frequent observations)
2. Collect more field samples (target: 500+)
3. Check data preprocessing outputs
4. Verify sample_id matches across files

---

## Next Steps

After successful training:

1. **Analyze Results**
   - Review metrics and performance
   - Identify underperforming periods

2. **Improve Model**
   - Collect more training data
   - Add more vegetation indices
   - Tune hyperparameters

3. **Deploy**
   - Save trained model
   - Create prediction script for new data
   - Integrate with operational systems

4. **Research**
   - Analyze feature importance
   - Study model attention patterns
   - Compare with baseline methods

---

## Performance Tips

1. **Data Quality**: Clean data is more important than quantity
2. **Feature Engineering**: Include domain-specific indices
3. **Temporal Coverage**: 10+ observations per field per season
4. **Sample Size**: Minimum 100 fields, recommended 500+
5. **GPU Usage**: Can reduce training time by 10-50x

---

## Citation

If you use this code, please cite:

```bibtex
@software{yield_prediction_2024,
  author = {Your Name},
  title = {Crop Yield Prediction with LSTM and Multi-Head Attention},
  year = {2024},
  url = {https://github.com/gyy-rs/BRDF-crop-yield-prediction}
}
```

---

## Support

For issues or questions:
1. Check GitHub Issues
2. Review this guide and README.md
3. Create detailed issue report with:
   - Command executed
   - Error message (full traceback)
   - Your system info (Python version, OS, GPU)
   - Sample data (if possible)

---

**Last Updated**: February 2024
