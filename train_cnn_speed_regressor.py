#!/usr/bin/env python3
"""
Flow-Speed Dataset Creation & CNN Training
This script builds a dataset from optical-flow maps plus speed labels, 
splits 80/20, trains a small CNN regressor, plots metrics, and saves weights.
"""

import glob
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- Config ---
FLOW_DIR   = 'labeled_data/optical_flow'     # folder with flow_*.flo or .npy
SPEED_CSV  = 'actual_speeds.csv'       # timestamp,actual_speed
MODEL_OUT  = 'cnn_speed_regressor.pt'
IMG_SIZE   = 64
BATCH_SIZE = 64
NUM_EPOCHS = 25
LR         = 1e-3
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

try:
    from ptlflow.utils.io import read_flo
except ImportError:
    def read_flo(_): raise RuntimeError('Install ptlflow or convert .flo to .npy')

def load_flow(fp):
    """Load and preprocess flow file"""
    flow = np.load(fp) if fp.endswith('.npy') else read_flo(fp)
    # Handle 4D arrays (batch dimension) by squeezing
    if flow.ndim == 4:
        flow = np.squeeze(flow)  # Remove batch dimension
    # flow should now be (2, H, W) - resize each channel
    flow_resized = np.zeros((2, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i in range(2):
        flow_resized[i] = cv2.resize(flow[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return flow_resized  # Already in (2, H, W) format

def main():
    # Find flow files
    flow_files = sorted(glob.glob(os.path.join(FLOW_DIR, 'flow_*.*')))
    assert flow_files, 'No flow files found!'
    print(f'Found {len(flow_files)} flow files')

    # --- Load labels ---
    if os.path.exists(SPEED_CSV):
        df = pd.read_csv(SPEED_CSV).sort_values('timestamp')
        times, speeds = df['timestamp'].values, df['actual_speed'].values
        def idx(fp): return int(os.path.basename(fp).split('_')[1].split('.')[0])
        y = np.array([np.interp(idx(fp), np.arange(len(times)), speeds) for fp in flow_files], dtype=np.float32)
    else:
        def speed_from_name(fp): return float(fp.split('_speed_')[-1].split('.')[0])
        y = np.array([speed_from_name(fp) for fp in flow_files], dtype=np.float32)

    # Load and preprocess flow data
    print("Loading flow files...")
    
    # Process in batches to avoid memory issues
    batch_size = 100  # Process 100 files at a time
    X_list = []
    
    for i in range(0, len(flow_files), batch_size):
        batch_files = flow_files[i:i+batch_size]
        batch_data = [load_flow(fp) for fp in batch_files]
        X_list.extend(batch_data)
        print(f"Processed {min(i+batch_size, len(flow_files))}/{len(flow_files)} files")
    
    X = np.stack(X_list)
    print('Dataset:', X.shape, y.shape)

    # Split data
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=SEED)

    # Dataset class
    class FlowDS(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y).unsqueeze(1)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]

    tr_loader = DataLoader(FlowDS(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FlowDS(X_val, y_val), batch_size=BATCH_SIZE)

    # Model definition
    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(2, 16, 5, padding=2), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 1)
            )
        def forward(self, x): return self.net(x)

    # Setup training
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dev}")
    
    model = Regressor().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # Training loop
    print("Starting training...")
    tr_loss, val_loss = [], []
    for ep in range(1, NUM_EPOCHS+1):
        # Training
        model.train()
        run = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            run += loss.item() * xb.size(0)
        tr_loss.append(run / len(tr_loader.dataset))

        # Validation
        model.eval()
        run = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                run += loss_fn(model(xb), yb).item() * xb.size(0)
        val_loss.append(run / len(val_loader.dataset))
        print(f'Epoch {ep:02d}/{NUM_EPOCHS} | train {tr_loss[-1]:.4f} | val {val_loss[-1]:.4f}')

    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(tr_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        preds = model(torch.from_numpy(X_val).to(dev)).cpu().numpy().squeeze()
    plt.plot(y_val, label='actual', alpha=0.7)
    plt.plot(preds, label='pred', alpha=0.7)
    plt.legend()
    plt.title('Validation Predictions')
    plt.xlabel('Sample')
    plt.ylabel('Speed')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Save model
    torch.save(model.state_dict(), MODEL_OUT)
    print(f'Saved model → {MODEL_OUT}')
    print(f'Saved plots → training_results.png')

if __name__ == "__main__":
    main() 