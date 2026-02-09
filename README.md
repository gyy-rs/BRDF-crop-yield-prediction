# Crop Yield Prediction with LSTM and Multi-Head Attention

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Deep learning model for crop yield estimation using multi-temporal vegetation indices (VIs) and Solar-Induced Fluorescence (SIF) data with LSTM and multi-head attention mechanisms.

## 📋 Overview

This repository contains the complete pipeline for crop yield prediction using:

- **LSTM networks** for temporal sequence modeling
- **Multi-head attention mechanism** for identifying key temporal patterns
- **10-times repeated 5-fold cross-validation** for robust evaluation
- **Multi-angle BRDF correction** for viewing geometry normalization

### Key Features

- ✅ Handles multi-temporal vegetation indices (NDVI, NIRv, EVI2, etc.)
- ✅ Incorporates Solar-Induced Fluorescence (SIF) measurements
- ✅ **BRDF radiative transfer corrections** across viewing angles (0-60°)
  - Ross-thick kernel (volumetric scattering)
  - Li-sparse kernel (geometric scattering)
  - Multi-angle SIF generation
- ✅ Robust validation strategy (50 training iterations)
- ✅ Uncertainty quantification (Mean ± Std from CV folds)
- ✅ Fully reproducible with fixed random seeds

---

## 📊 Model Architecture

```
Input (Time steps, Features)
        ↓
    LSTM Layer 1 (64 units)
    LSTM Layer 2 (64 units)
        ↓
Multi-Head Attention (4 heads)
        ↓
     Linear (64 → 32)
        ↓
       ReLU
        ↓
Dropout (0.3)
        ↓
     Linear (32 → 1)
        ↓
   Yield Output
```

### Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| LSTM Layers | 2 |
| Hidden Units | 64 |
| Attention Heads | 4 |
| Dropout Rate | 0.3 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 16 |
| Epochs | 100 |

### Validation Strategy

- **Cross-validation**: 5-fold (repeated 10 times)
- **Total training iterations**: 50
- **Performance metrics**: R², MSE, MAE reported as Mean ± Std

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- GPU recommended (CUDA 11.8+)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/gyy-rs/BRDF-crop-yield-prediction.git
cd BRDF-crop-yield-prediction
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📦 Data Preparation

### Dataset Structure

Your input data should contain:

1. **SIF Data** (`raw_sif_data.csv`):
   - `sample_id`: Sample identifier
   - `year`, `month`, `day`: Date information
   - `sif743`: Solar-Induced Fluorescence at 743nm
   - `par`: Photosynthetically Active Radiation
   - `sza`, `vza`, `raa`: Solar/viewing angles
   - `iso_r`, `vol_r`, `geo_r`: Red BRDF parameters
   - `iso_n`, `vol_n`, `geo_n`: NIR BRDF parameters

2. **Vegetation Indices Data** (`raw_vis_data.csv`):
   - `sample_id`, `year`, `month`, `day`
   - `NDVI`, `NIRv`, `EVI2`: Vegetation indices
   - Other spectral indices

3. **Yield Data** (`raw_yield_data.csv`):
   - `sample_id`, `year`
   - `yield`: Crop yield (target variable)

### Example Data

See `data/sample/sample_data.csv` for a minimal working example.

---

## 🔄 Workflow

### 0. BRDF Correction (Optional but Recommended)

**What is BRDF Correction?**

BRDF (Bidirectional Reflectance Distribution Function) correction removes the effects of viewing geometry on reflectance measurements. This is important because:
- TROPOMI observations vary with sun and view angles
- Same surface appears different at different viewing angles
- BRDF correction normalizes measurements to a standard geometry

**Module**: `src/brdf_correction.py`

**Key Functions**:
- `brdf_correction()`: Apply BRDF correction to reflectance values
- `apply_multi_angle_correction()`: Generate SIF/VI at different angles

**Example Usage**:
```python
from src.brdf_correction import apply_multi_angle_correction

# Generate SIF and vegetation indices at 169 different angles
df_corrected = apply_multi_angle_correction(
    df_tropomi,
    view_zenith_steps=[0, 15, 30, 45, 60],
    sun_zenith_steps=[20, 30, 40, 50, 60],
)
# Output: Original columns + 169 × 6 new columns (RED, NIR, NDVI, NIRv, EVI2, SIF)
```

**For Detailed Instructions**: See [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md)

**Run Example**:
```bash
python examples/brdf_correction_example.py
```

### 1. Data Preprocessing

```bash
cd src
python data_preprocessing.py
```

**What it does:**
- Loads raw SIF, vegetation indices, and yield data
- Merges datasets by sample_id and date
- **Applies BRDF corrections** across viewing geometries (if enabled)
- Aggregates data into 10-day periods (dekads)
- Pivots time series into feature matrix
- Saves preprocessed features

**Output files:**
- `data/final_yield_features.pkl` - Feature matrix with yield labels
- `data/time_index_mapping.csv` - Temporal index mapping
- `data/final_yield_features.csv` - CSV format for inspection

### 2. Model Training

```bash
python train.py
```

**What it does:**
- Loads preprocessed features
- Reshapes data to (samples, timesteps, features)
- Performs 10×5-fold cross-validation
- Trains AttentionLSTM model on each fold
- Reports Mean ± Std metrics

**Output:**
- `results/cv_results.csv` - CV fold results
- Console output with final performance metrics

**Expected Output:**
```
========================================================
FINAL MODEL PERFORMANCE (Based on CV folds)
========================================================
R² Score:  0.79 ± 0.08
MSE:       15234.50 ± 2145.30
MAE:       89.23
========================================================
```

---

## 📁 Project Structure

```
crop-yield-prediction/
├── src/
│   ├── model.py                 # LSTM + Attention model definition
│   ├── train.py                 # Training and evaluation script
│   ├── data_preprocessing.py    # Data preparation pipeline
│   └── config.py                # Configuration parameters (optional)
├── data/
│   ├── sample/
│   │   └── sample_data.csv      # Example dataset for testing
│   ├── raw/                     # Put raw CSV files here
│   │   ├── raw_sif_data.csv
│   │   ├── raw_vis_data.csv
│   │   └── raw_yield_data.csv
│   └── final_yield_features.pkl # Output: processed features
├── results/
│   └── cv_results.csv           # Cross-validation results
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── LICENSE
```

---

## 💾 Input Data Format

### SIF Data Format (`raw_sif_data.csv`)

| sample_id | year | month | day | sif743 | par | sza | vza | raa | iso_r | vol_r | geo_r | iso_n | vol_n | geo_n |
|-----------|------|-------|-----|--------|-----|-----|-----|-----|-------|-------|-------|-------|-------|-------|
| 1 | 2019 | 3 | 1 | 1.25 | 45.3 | 35.2 | 15.0 | 120.5 | 0.12 | 0.08 | -0.05 | 0.18 | 0.15 | -0.02 |
| 1 | 2019 | 3 | 2 | 1.32 | 46.1 | 34.8 | 16.2 | 118.3 | 0.13 | 0.09 | -0.04 | 0.19 | 0.16 | -0.01 |

### Vegetation Indices Format (`raw_vis_data.csv`)

| sample_id | year | month | day | NDVI | NIRv | EVI2 | GVMI | ... |
|-----------|------|-------|-----|------|------|------|------|-----|
| 1 | 2019 | 3 | 1 | 0.45 | 0.38 | 0.42 | 0.52 | ... |
| 1 | 2019 | 3 | 2 | 0.48 | 0.41 | 0.44 | 0.55 | ... |

### Yield Data Format (`raw_yield_data.csv`)

| sample_id | year | yield |
|-----------|------|-------|
| 1 | 2019 | 5200 |
| 2 | 2019 | 5450 |

---

## 📊 Output and Results

### Cross-Validation Results

After training, `results/cv_results.csv` contains:

```
r2,mse,mae
0.374144,382868.89,465.14
0.293687,405234.56,478.32
...
0.338134,391245.67,472.18
```

### Performance Metrics

The model reports:
- **R² (Coefficient of Determination)**: Proportion of yield variance explained (0-1)
- **MSE (Mean Squared Error)**: Average squared prediction error
- **MAE (Mean Absolute Error)**: Average absolute prediction error

All metrics reported as **Mean ± Standard Deviation** from 50 CV folds.

---

## 🔧 Customization

### Modify Model Architecture

Edit hyperparameters in `src/train.py`:

```python
# Model parameters
LSTM_HIDDEN_DIM = 64      # Change to 128 for larger model
N_HEADS = 4               # Number of attention heads
N_LAYERS = 2              # Number of LSTM layers
DROPOUT = 0.3             # Regularization

# Training parameters
NUM_EPOCHS = 100          # Increase for better convergence
LEARNING_RATE = 0.001     # Adjust if training is unstable
BATCH_SIZE = 16           # Increase for faster training (if GPU memory allows)
```

### Modify Data Aggregation

Edit in `src/data_preprocessing.py`:

```python
AGGREGATION_METHOD = 'dekad'  # Change to 'month' for monthly aggregation
START_DATE = '03-01'          # Change growing season start
END_DATE = '05-30'            # Change growing season end
```

### Enable BRDF Corrections

Uncomment in `src/data_preprocessing.py`:

```python
# df = compute_multi_angle_features(df)  # ← Uncomment this line
```

---

## 📈 Viewing Geometry Sensitivity Analysis

The model's robustness to viewing angles (VZA) and solar angles (SZA) can be analyzed by:

1. Computing features at different angle combinations
2. Training separate models for each combination
3. Comparing performance stability

Example angles tested:
- **VZA (Viewing Zenith Angle)**: 0°, 5°, 10°, ..., 60°
- **SZA (Solar Zenith Angle)**: 0°, 5°, 10°, ..., 60°

---

## 🔬 Reproducibility

For full reproducibility:

```python
# All components use fixed random seeds
RANDOM_STATE = 42

# PyTorch deterministic mode (optional)
torch.manual_seed(42)
np.random.seed(42)
```

---

## 📚 References

### Model Architecture
- Hochreiter & Schmidhuber (1997): LSTM networks
- Vaswani et al. (2017): Attention is All You Need

### Datasets
- TROPOMI SIF: European Commission Copernicus Programme
- Vegetation Indices: MODIS, Sentinel-2, or similar

### BRDF Correction
- Rahman et al. (1993): Surface reflectance model
- Lewis et al. (1999): BRDF model comparison

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{yourpaper2024,
  title={Crop Yield Prediction Using LSTM with Multi-Head Attention},
  author={Your Name et al.},
  journal={Journal Name},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ❓ FAQ

**Q: What if I don't have BRDF parameters?**
- A: Set `iso_r, vol_r, geo_r, iso_n, vol_n, geo_n` to 0 or comment out BRDF correction step.

**Q: Can I use different vegetation indices?**
- A: Yes! Modify the feature list in `data_preprocessing.py`. The model adapts to any number of features.

**Q: How much data do I need?**
- A: Minimum ~100 samples. More data (1000+) typically improves generalization.

**Q: What GPU do I need?**
- A: Any GPU with 4GB+ VRAM. CPU training is supported but slower.

---

## 👥 Support

For questions and issues:
1. Check existing GitHub Issues
2. Create a new Issue with detailed description
3. Email: gaoyy@cau.edu.cn

---

## 🔗 Links

- 📖 [Documentation](https://github.com/gyy-rs/BRDF-crop-yield-prediction/wiki)
- 📰 [Paper](https://doi.org/xxxxx)
- 🌐 [Project Website](https://example.com)

---

**Last Updated**: February 2026
**Status**: Stable Release v1.0
