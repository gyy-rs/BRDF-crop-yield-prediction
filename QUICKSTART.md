# Quick Start Guide - Crop Yield Prediction

## 5-Minute Quick Start

Follow these steps to get the model running on your system.

---

## Step 1: Installation (2 minutes)

### 1.1 Clone or Download Repository
```bash
git clone https://github.com/gyy-rs/BRDF-crop-yield-prediction.git
cd crop-yield-prediction
```

### 1.2 Create Virtual Environment
```bash
# Create environment
python -m venv venv

# Activate environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 1.3 Install Dependencies
```bash
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed torch-2.0.0 numpy-1.24.0 pandas-2.0.0 ...
```

---

## Step 2: Prepare Data (1 minute)

### Option A: Use Sample Data (Test Run)

The repository includes sample data. No preparation needed!

```bash
cd src
python train.py  # Skip to Step 3
```

### Option B: Use Your Own Data

1. **Organize your raw data files:**
   ```
   data/raw/
   ├── raw_sif_data.csv           # SIF measurements
   ├── raw_vis_data.csv           # Vegetation indices
   └── raw_yield_data.csv         # Crop yields
   ```

2. **Check data format** (see README.md for detailed format)

3. **Run preprocessing:**
   ```bash
   cd src
   python data_preprocessing.py
   ```

   **Expected output:**
   ```
   ==================================================
   CROP YIELD PREDICTION - DATA PREPROCESSING
   ==================================================
   
   Step 1: Loading raw data...
     ✓ SIF data: (10000, 15)
     ✓ VIS data: (10000, 8)
     ✓ Yield data: (500, 2)
   
   Step 2: Merging datasets...
     ✓ Merged data: (10000, 25)
   ...
   DATA PREPROCESSING COMPLETE
   ==================================================
   ```

---

## Step 3: Train Model (2 minutes with GPU, 10 min with CPU)

```bash
python train.py
```

**What the model does:**
1. Loads preprocessed data
2. Reshapes to time series format
3. Performs 10×5-fold cross-validation (50 training runs)
4. Reports performance metrics

**Expected output:**
```
Loading preprocessed data...
Data loaded: 500 samples, 80 features

Reshaped data: (500, 8, 10)

Cross-validation folds: [████████████████████] 50/50

============================================================
FINAL MODEL PERFORMANCE (Based on CV folds)
============================================================
R² Score:  0.79 ± 0.08
MSE:       15234.50 ± 2145.30
MAE:       89.23
============================================================

Results saved to ./results/
```

---

## What Each Script Does

### `data_preprocessing.py`
- Loads raw SIF, vegetation indices, and yield data
- Merges by sample ID and date
- Aggregates into 10-day periods (dekads)
- Saves as pickle and CSV

**Run this once** when you have new data.

### `train.py`
- Loads preprocessed features
- Performs repeated 5-fold cross-validation
- Trains LSTM+Attention model
- Reports evaluation metrics

**Run this** to train and evaluate.

---

## Common Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: "FileNotFoundError: data/raw_sif_data.csv not found"
**Solution:**
```bash
# Copy your data files to the data/ folder
cp your_sif_data.csv data/raw/raw_sif_data.csv
cp your_vis_data.csv data/raw/raw_vis_data.csv
cp your_yield_data.csv data/raw/raw_yield_data.csv

# Then run preprocessing
python data_preprocessing.py
```

### Issue 3: CUDA out of memory error
**Solution:**
```python
# In src/train.py, reduce batch size:
BATCH_SIZE = 8  # Instead of 16
```

### Issue 4: Training is very slow
**Solution:**
- Check if GPU is being used: `python -c "import torch; print(torch.cuda.is_available())"`
- If False, install GPU-enabled PyTorch
- If True, GPU is working correctly

---

## Viewing Results

### Check CV Results
```bash
# View cross-validation results
cat results/cv_results.csv
```

Sample output:
```
r2,mse,mae
0.7842,12345.67,87.23
0.7956,11892.34,85.12
0.7658,13456.78,89.45
...
```

### Interpretation
- **R² = 0.79**: Model explains 79% of yield variance
- **MAE = 89 kg/ha**: Average prediction error is 89 kg/ha
- **STD R² = 0.08**: Consistent performance across folds

---

## Next Steps

### Understand the Model
- Open `src/model.py` to see the LSTM architecture
- Check `src/train.py` to understand training procedure
- Read `README.md` for detailed explanations

### Customize the Model
- Change hyperparameters in `src/train.py`
- Modify aggregation method in `src/data_preprocessing.py`
- Use different vegetation indices

### Improve Results
1. Collect more samples (>1000 recommended)
2. Expand temporal range (longer growing season)
3. Include more vegetation indices
4. Tune hyperparameters (learning rate, batch size, etc.)

---

## Performance Benchmarks

| Dataset Size | GPU (NVIDIA RTX 3090) | CPU (Intel i7) |
|--------------|----------------------|----------------|
| 100 samples | 1 min | 8 min |
| 500 samples | 2 min | 15 min |
| 1000 samples | 3 min | 30 min |
| 5000 samples | 5 min | 2 hours |

---

## File Locations After Running

```
crop-yield-prediction/
├── data/
│   ├── raw/                          # Your input CSV files
│   ├── final_yield_features.pkl      # ← Generated by preprocessing
│   ├── final_yield_features.csv      # ← For inspection
│   └── time_index_mapping.csv        # ← Temporal indices
├── results/
│   └── cv_results.csv                # ← Generated by training
└── src/
    └── (model.py, train.py, ...)
```

---

## Validation Strategy Explained

The model uses **10-times repeated 5-fold cross-validation**:

1. **5-Fold**: Split 500 samples into 5 groups (100 each)
2. **Repeat 10×**: Randomly reshuffle and repeat
3. **Total**: 50 independent training runs

Each run:
- Uses 400 samples for training
- Uses 100 samples for testing
- Reports R², MSE, MAE

**Final result**: Mean ± Std from all 50 runs

This ensures robust evaluation and accounts for randomness in neural networks.

---

## Getting Help

1. **Check README.md** for detailed documentation
2. **Review comments** in source code (src/*.py)
3. **Open an Issue** on GitHub with:
   - Your command/error
   - Output/traceback
   - Your system info (Python version, OS, GPU)

---

## Reference

For more information:
- 📖 Full Documentation: [README.md](../README.md)
- 🔧 Model Architecture: [model.py](../src/model.py)
- 🚀 Training Script: [train.py](../src/train.py)
- 📊 Data Preprocessing: [data_preprocessing.py](../src/data_preprocessing.py)

---

**Last Updated**: February 2024
**Status**: Stable Release
