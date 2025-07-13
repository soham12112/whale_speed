#!/usr/bin/env python3
"""
Temporal Flow-Speed CNN with Huber Loss
Uses multiple consecutive flow frames to predict speed with temporal context.
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
MODEL_OUT  = 'temporal_cnn_speed_regressor.pt'
IMG_SIZE   = 128  # Higher resolution
BATCH_SIZE = 16   # Smaller due to temporal dimension
NUM_EPOCHS = 60   # More epochs for temporal learning
LR         = 3e-4  # Lower learning rate
WEIGHT_DECAY = 1e-4
TEMPORAL_WINDOW = 5  # Number of consecutive frames
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
    flow_resized = np.zeros((4, IMG_SIZE, IMG_SIZE), dtype=np.float32)
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

def create_temporal_sequences(flow_files, labels, window_size=TEMPORAL_WINDOW):
    """Create temporal sequences of flow data"""
    sequences = []
    sequence_labels = []
    
    print(f"Creating temporal sequences with window size {window_size}...")
    
    # Sort files by index to ensure temporal order
    indexed_files = []
    for fp in flow_files:
        try:
            idx = int(os.path.basename(fp).split('_')[1])
            indexed_files.append((idx, fp))
        except:
            continue
    
    indexed_files.sort(key=lambda x: x[0])
    
    # Create sequences
    for i in range(len(indexed_files) - window_size + 1):
        sequence_files = [indexed_files[i + j][1] for j in range(window_size)]
        sequence_indices = [indexed_files[i + j][0] for j in range(window_size)]
        
        # Check if sequence is continuous
        if all(sequence_indices[j+1] - sequence_indices[j] == 1 for j in range(window_size-1)):
            # Load sequence data
            sequence_data = []
            for fp in sequence_files:
                flow_data = load_flow_enhanced(fp)
                sequence_data.append(flow_data)
            
            sequences.append(np.stack(sequence_data))  # Shape: (T, C, H, W)
            
            # Use the label from the center frame
            center_idx = sequence_indices[window_size // 2]
            if center_idx < len(labels):
                sequence_labels.append(labels[center_idx])
        
        if len(sequences) % 500 == 0:
            print(f"Created {len(sequences)} sequences...")
    
    return np.array(sequences), np.array(sequence_labels)

class TemporalFlowDS(Dataset):
    """Temporal dataset with data augmentation"""
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X)  # Shape: (N, T, C, H, W)
        self.y = torch.from_numpy(y).unsqueeze(1)
        self.augment = augment
        
    def __len__(self): 
        return len(self.X)
    
    def __getitem__(self, i):
        x, y = self.X[i], self.y[i]  # x shape: (T, C, H, W)
        
        # Data augmentation for training
        if self.augment and np.random.rand() > 0.5:
            # Random horizontal flip (apply to all frames)
            if np.random.rand() > 0.5:
                x = torch.flip(x, [3])  # Flip width
                x[:, 0] = -x[:, 0]  # Flip x-component of flow
            
            # Add small amount of noise
            if np.random.rand() > 0.5:
                noise = torch.randn_like(x) * 0.01
                x = x + noise
            
            # Temporal dropout - randomly zero out some frames
            if np.random.rand() > 0.7:
                dropout_frame = np.random.randint(0, x.shape[0])
                x[dropout_frame] = 0
        
        return x, y

class TemporalCNN(nn.Module):
    """3D CNN for temporal flow analysis"""
    def __init__(self, temporal_window=TEMPORAL_WINDOW, dropout_rate=0.3):
        super().__init__()
        
        # 3D Convolutional layers for temporal-spatial feature extraction
        self.conv3d1 = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=(3, 7, 7), padding=(1, 3, 3)),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU(),
        )
        
        self.conv3d2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64), nn.ReLU(),
        )
        
        self.conv3d3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128), nn.ReLU(),
        )
        
        # Pooling layers
        self.pool3d1 = nn.MaxPool3d((1, 2, 2))  # Don't pool temporal dimension initially
        self.pool3d2 = nn.MaxPool3d((1, 2, 2))
        self.pool3d3 = nn.MaxPool3d((2, 2, 2))  # Pool temporal dimension here
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: (B, T, C, H, W) -> need (B, C, T, H, W) for 3D conv
        x = x.permute(0, 2, 1, 3, 4)
        
        # 3D convolutions
        x1 = self.pool3d1(self.conv3d1(x))
        x2 = self.pool3d2(self.conv3d2(x1))
        x3 = self.pool3d3(self.conv3d3(x2))
        
        # Temporal attention
        attention_weights = self.temporal_attention(x3)
        x3_attended = x3 * attention_weights
        
        # Global pooling and classification
        x_pooled = self.global_pool(x3_attended)
        x_flat = x_pooled.view(x_pooled.size(0), -1)
        
        return self.classifier(x_flat)

class HuberLoss(nn.Module):
    """Huber Loss for robust regression"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = pred - target
        is_small_error = torch.abs(error) <= self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * torch.abs(error) - 0.5 * self.delta ** 2
        return torch.where(is_small_error, squared_loss, linear_loss).mean()

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

    # Create temporal sequences
    X_temporal, y_temporal = create_temporal_sequences(flow_files, y, TEMPORAL_WINDOW)
    print(f'Temporal Dataset: {X_temporal.shape}, {y_temporal.shape}')

    # Normalize target values
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_temporal.reshape(-1, 1)).flatten()

    # Split data
    X_tr, X_val, y_tr, y_val = train_test_split(X_temporal, y_scaled, test_size=0.2, shuffle=True, random_state=SEED)
    y_tr_orig, y_val_orig = train_test_split(y_temporal, test_size=0.2, shuffle=True, random_state=SEED)

    # Create datasets
    tr_dataset = TemporalFlowDS(X_tr, y_tr, augment=True)
    val_dataset = TemporalFlowDS(X_val, y_val, augment=False)
    
    tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model setup
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dev}")
    
    model = TemporalCNN(TEMPORAL_WINDOW).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)
    
    # Use Huber Loss for robust training
    criterion = HuberLoss(delta=0.5)  # Adjust delta based on your data scale

    # Training with early stopping
    print("Starting temporal training with Huber loss...")
    tr_loss, val_loss = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 12
    
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
    plt.title('Training Loss (Huber)')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.yscale('log')
    
    plt.subplot(1, 3, 2)
    plt.scatter(y_val_orig, preds_original, alpha=0.6)
    plt.plot([y_val_orig.min(), y_val_orig.max()], [y_val_orig.min(), y_val_orig.max()], 'r--')
    plt.xlabel('Actual Speed')
    plt.ylabel('Predicted Speed')
    plt.title('Temporal Model: Actual vs Predicted')
    
    plt.subplot(1, 3, 3)
    residuals = preds_original - y_val_orig
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig('temporal_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'Saved temporal model → {MODEL_OUT}')
    print(f'Saved plots → temporal_training_results.png')

if __name__ == "__main__":
    main() 