#!/usr/bin/env python3
"""
Improved Flow-Speed Dataset Creation & CNN Training
Enhanced version with better architecture, data preprocessing, and training techniques.
"""

import glob
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- Enhanced Config ---
FLOW_DIR   = 'labeled_data/optical_flow'
SPEED_CSV  = 'actual_speeds.csv'
MODEL_OUT  = 'improved_cnn_speed_regressor.pt'
IMG_SIZE   = 128  # Increased resolution
BATCH_SIZE = 32   # Smaller batch size for better gradients
NUM_EPOCHS = 50   # More epochs
LR         = 5e-4  # Lower learning rate
WEIGHT_DECAY = 1e-4  # L2 regularization
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

try:
    from ptlflow.utils.io import read_flo
except ImportError:
    def read_flo(_): raise RuntimeError('Install ptlflow or convert .flo to .npy')

def load_flow_enhanced(fp):
    """Enhanced flow loading with better preprocessing"""
    flow = np.load(fp) if fp.endswith('.npy') else read_flo(fp)
    
    # Handle 4D arrays (batch dimension) by squeezing
    if flow.ndim == 4:
        flow = np.squeeze(flow)
    
    # Calculate flow magnitude and direction as additional features
    flow_x, flow_y = flow[0], flow[1]
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    direction = np.arctan2(flow_y, flow_x)
    
    # Resize all channels
    flow_resized = np.zeros((4, IMG_SIZE, IMG_SIZE), dtype=np.float32)  # 4 channels now
    flow_resized[0] = cv2.resize(flow_x, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    flow_resized[1] = cv2.resize(flow_y, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    flow_resized[2] = cv2.resize(magnitude, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    flow_resized[3] = cv2.resize(direction, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    # Normalize each channel
    for i in range(4):
        channel = flow_resized[i]
        if channel.std() > 0:
            flow_resized[i] = (channel - channel.mean()) / channel.std()
    
    return flow_resized

class EnhancedFlowDS(Dataset):
    """Enhanced dataset with data augmentation"""
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)
        self.augment = augment
        
    def __len__(self): 
        return len(self.X)
    
    def __getitem__(self, i):
        x, y = self.X[i], self.y[i]
        
        # Data augmentation for training
        if self.augment and np.random.rand() > 0.5:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                x = torch.flip(x, [2])  # Flip width
                x[0] = -x[0]  # Flip x-component of flow
            
            # Add small amount of noise
            if np.random.rand() > 0.5:
                noise = torch.randn_like(x) * 0.01
                x = x + noise
        
        return x, y

class ImprovedRegressor(nn.Module):
    """Improved CNN architecture with residual connections and attention"""
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        # Feature extraction with residual blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, 7, padding=3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # Feature extraction
        x1 = self.pool1(self.conv1(x))
        x2 = self.pool2(self.conv2(x1))
        x3 = self.pool3(self.conv3(x2))
        
        # Attention
        attention_weights = self.attention(x3)
        x3_attended = x3 * attention_weights
        
        # Global pooling and classification
        x_pooled = self.global_pool(x3_attended)
        x_flat = x_pooled.view(x_pooled.size(0), -1)
        
        return self.classifier(x_flat)

def main():
    # Find flow files
    flow_files = sorted(glob.glob(os.path.join(FLOW_DIR, 'flow_*.*')))
    assert flow_files, 'No flow files found!'
    print(f'Found {len(flow_files)} flow files')

    # Load labels
    if os.path.exists(SPEED_CSV):
        df = pd.read_csv(SPEED_CSV).sort_values('timestamp')
        times, speeds = df['timestamp'].values, df['actual_speed'].values
        def idx(fp): return int(os.path.basename(fp).split('_')[1].split('.')[0])
        y = np.array([np.interp(idx(fp), np.arange(len(times)), speeds) for fp in flow_files], dtype=np.float32)
    else:
        def speed_from_name(fp): return float(fp.split('_speed_')[-1].split('.')[0])
        y = np.array([speed_from_name(fp) for fp in flow_files], dtype=np.float32)

    # Normalize target values
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Load and preprocess flow data with enhanced features
    print("Loading flow files with enhanced preprocessing...")
    batch_size = 50
    X_list = []
    
    for i in range(0, len(flow_files), batch_size):
        batch_files = flow_files[i:i+batch_size]
        batch_data = [load_flow_enhanced(fp) for fp in batch_files]
        X_list.extend(batch_data)
        print(f"Processed {min(i+batch_size, len(flow_files))}/{len(flow_files)} files")
    
    X = np.stack(X_list)
    print('Enhanced Dataset:', X.shape, y_scaled.shape)

    # Split data
    X_tr, X_val, y_tr, y_val = train_test_split(X, y_scaled, test_size=0.2, shuffle=True, random_state=SEED)
    y_tr_orig, y_val_orig = train_test_split(y, test_size=0.2, shuffle=True, random_state=SEED)

    # Create datasets with augmentation for training
    tr_dataset = EnhancedFlowDS(X_tr, y_tr, augment=True)
    val_dataset = EnhancedFlowDS(X_val, y_val, augment=False)
    
    tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model setup
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dev}")
    
    model = ImprovedRegressor().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    # Training with early stopping
    print("Starting enhanced training...")
    tr_loss, val_loss = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for ep in range(1, NUM_EPOCHS+1):
        # Training
        model.train()
        train_running_loss = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_running_loss += loss.item() * xb.size(0)
        
        avg_train_loss = train_running_loss / len(tr_loader.dataset)
        tr_loss.append(avg_train_loss)

        # Validation
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_running_loss += loss.item() * xb.size(0)
        
        avg_val_loss = val_running_loss / len(val_loader.dataset)
        val_loss.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            patience_counter += 1
        
        print(f'Epoch {ep:02d}/{NUM_EPOCHS} | train {avg_train_loss:.4f} | val {avg_val_loss:.4f} | lr {optimizer.param_groups[0]["lr"]:.6f}')
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {ep}")
            break

    # Load best model
    model.load_state_dict(torch.load(MODEL_OUT))
    
    # Final evaluation with original scale
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(X_val).to(dev)).cpu().numpy().squeeze()
        preds_original = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = np.mean((preds_original - y_val_orig)**2)
    mae = np.mean(np.abs(preds_original - y_val_orig))
    rmse = np.sqrt(mse)
    
    print(f"\nFinal Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Enhanced plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(tr_loss, label='train', alpha=0.8)
    plt.plot(val_loss, label='val', alpha=0.8)
    plt.legend()
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    
    plt.subplot(1, 3, 2)
    plt.scatter(y_val_orig, preds_original, alpha=0.6)
    plt.plot([y_val_orig.min(), y_val_orig.max()], [y_val_orig.min(), y_val_orig.max()], 'r--')
    plt.xlabel('Actual Speed')
    plt.ylabel('Predicted Speed')
    plt.title('Actual vs Predicted')
    
    plt.subplot(1, 3, 3)
    residuals = preds_original - y_val_orig
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig('improved_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'Saved improved model → {MODEL_OUT}')
    print(f'Saved plots → improved_training_results.png')

if __name__ == "__main__":
    main() 