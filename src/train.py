# File: train.py
# Description: Training script for LSTM-based crop yield prediction model

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
from tqdm import tqdm

from model import AttentionLSTMModel

# ============================================================================
# 1. Configuration
# ============================================================================

# File paths
DATA_PKL_PATH = './data/final_yield_features.pkl'
TIME_INDEX_MAPPING_CSV = './data/time_index_mapping.csv'
OUTPUT_DIR = './results/'

# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 100
RANDOM_STATE = 42

# Model architecture parameters
LSTM_HIDDEN_DIM = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.3

# Cross-validation settings
NUM_FOLDS = 5
N_REPEATS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 2. PyTorch Dataset Class
# ============================================================================

class YieldDataset(Dataset):
    """Custom PyTorch Dataset for yield estimation"""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# 3. Training and Evaluation Functions
# ============================================================================

def train_and_evaluate_fold(X_train, y_train, X_test, y_test, num_timesteps, 
                           num_features_per_step, fold_num, total_folds):
    """
    Train and evaluate model on a single fold.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features (batch_size, timesteps, features)
    y_train : np.ndarray
        Training labels (batch_size, 1)
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    num_timesteps : int
        Number of time steps
    num_features_per_step : int
        Number of features per time step
    fold_num : int
        Current fold number
    total_folds : int
        Total number of folds
    
    Returns
    -------
    r2, mse, mae : float
        Performance metrics
    """
    # Feature scaling
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, num_features_per_step))
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, num_features_per_step))
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    # Create datasets and dataloaders
    train_dataset = YieldDataset(X_train_scaled, y_train_scaled)
    test_dataset = YieldDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = AttentionLSTMModel(
        input_dim=num_features_per_step,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout_rate=DROPOUT
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
    
    # Evaluation on test set
    model.eval()
    test_predictions_scaled = []
    test_true = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            test_predictions_scaled.extend(outputs.squeeze().cpu().numpy())
            test_true.extend(labels.squeeze().cpu().numpy())
    
    # Inverse transform predictions
    test_predictions_scaled = np.array(test_predictions_scaled).reshape(-1, 1)
    test_true = np.array(test_true).reshape(-1, 1)
    
    test_predictions = scaler_y.inverse_transform(test_predictions_scaled)
    
    # Calculate metrics
    r2 = r2_score(test_true, test_predictions)
    mse = mean_squared_error(test_true, test_predictions)
    mae = mean_absolute_error(test_true, test_predictions)
    
    return r2, mse, mae


def main():
    """Main training pipeline"""
    
    print("Loading preprocessed data...")
    try:
        df = pd.read_pickle(DATA_PKL_PATH)
        time_map = pd.read_csv(TIME_INDEX_MAPPING_CSV)
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}")
        print("Please run the data preprocessing scripts first.")
        return
    
    print(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Extract features and labels
    X = df.drop(['yield'], axis=1).values
    y = df['yield'].values.reshape(-1, 1)
    
    num_timesteps = len(time_map)
    num_features_per_step = X.shape[1] // num_timesteps
    
    # Reshape to (samples, timesteps, features)
    X = X.reshape(-1, num_timesteps, num_features_per_step)
    
    print(f"Reshaped data: {X.shape}")
    
    # Initialize cross-validation
    total_folds = NUM_FOLDS * N_REPEATS
    kf = RepeatedKFold(n_splits=NUM_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    
    fold_results = []
    fold_pbar = tqdm(enumerate(kf.split(X)), total=total_folds, desc="Cross-validation folds")
    
    for fold_idx, (train_idx, test_idx) in fold_pbar:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        r2, mse, mae = train_and_evaluate_fold(
            X_train, y_train, X_test, y_test, 
            num_timesteps, num_features_per_step,
            fold_idx + 1, total_folds
        )
        
        fold_results.append({'r2': r2, 'mse': mse, 'mae': mae})
        fold_pbar.update(1)
    
    # Aggregate results
    mean_r2 = np.mean([r['r2'] for r in fold_results])
    std_r2 = np.std([r['r2'] for r in fold_results])
    mean_mse = np.mean([r['mse'] for r in fold_results])
    std_mse = np.std([r['mse'] for r in fold_results])
    mean_mae = np.mean([r['mae'] for r in fold_results])
    
    # Print results
    print("\n" + "="*60)
    print("FINAL MODEL PERFORMANCE (Based on CV folds)")
    print("="*60)
    print(f"R² Score:  {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"MSE:       {mean_mse:.2f} ± {std_mse:.2f}")
    print(f"MAE:       {mean_mae:.2f}")
    print("="*60)
    
    # Save results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'cv_results.csv'), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
