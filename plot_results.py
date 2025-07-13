#!/usr/bin/env python3
"""
Plot Results - Generate line plots for actual vs predicted speeds
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2

# --- Config ---
FLOW_DIR = 'labeled_data/optical_flow'
MODEL_PATH = 'improved_cnn_speed_regressor.pt'
IMG_SIZE = 128
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

def load_flow_enhanced(fp):
    """Enhanced flow loading with better preprocessing"""
    flow = np.load(fp)
    
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
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found. Please train the model first.")
        return
    
    # Find flow files
    flow_files = sorted(glob.glob(os.path.join(FLOW_DIR, 'flow_*.*')))
    assert flow_files, 'No flow files found!'
    print(f'Found {len(flow_files)} flow files')

    # Load labels
    def speed_from_name(fp): return float(fp.split('_speed_')[-1].split('.')[0])
    y = np.array([speed_from_name(fp) for fp in flow_files], dtype=np.float32)

    # Load and preprocess flow data
    print("Loading flow files...")
    batch_size = 50
    X_list = []
    
    for i in range(0, len(flow_files), batch_size):
        batch_files = flow_files[i:i+batch_size]
        batch_data = [load_flow_enhanced(fp) for fp in batch_files]
        X_list.extend(batch_data)
        print(f"Processed {min(i+batch_size, len(flow_files))}/{len(flow_files)} files")
    
    X = np.stack(X_list)
    print('Dataset:', X.shape, y.shape)

    # Normalize target values (same as training)
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Split data (same as training)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y_scaled, test_size=0.2, shuffle=True, random_state=SEED)
    y_tr_orig, y_val_orig = train_test_split(y, test_size=0.2, shuffle=True, random_state=SEED)

    # Load model
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedRegressor().to(dev)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    model.eval()

    # Make predictions
    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(X_val).to(dev)).cpu().numpy().squeeze()
        preds_original = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = np.mean((preds_original - y_val_orig)**2)
    mae = np.mean(np.abs(preds_original - y_val_orig))
    rmse = np.sqrt(mse)
    
    print(f"\nModel Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Create comprehensive plots
    plt.figure(figsize=(18, 12))
    
    # 1. Line plot - Actual vs Predicted over samples
    plt.subplot(2, 3, 1)
    sample_indices = np.arange(len(y_val_orig))
    plt.plot(sample_indices, y_val_orig, 'b-', label='Actual', alpha=0.7, linewidth=2)
    plt.plot(sample_indices, preds_original, 'r-', label='Predicted', alpha=0.7, linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('Speed')
    plt.title('Actual vs Predicted Speed (Line Plot)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Scatter plot - Actual vs Predicted
    plt.subplot(2, 3, 2)
    plt.scatter(y_val_orig, preds_original, alpha=0.6, s=20)
    plt.plot([y_val_orig.min(), y_val_orig.max()], [y_val_orig.min(), y_val_orig.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Speed')
    plt.ylabel('Predicted Speed')
    plt.title('Actual vs Predicted (Scatter)')
    plt.grid(True, alpha=0.3)
    
    # 3. Residuals plot
    plt.subplot(2, 3, 3)
    residuals = preds_original - y_val_orig
    plt.scatter(y_val_orig, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Actual Speed')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Actual Speed')
    plt.grid(True, alpha=0.3)
    
    # 4. Residuals histogram
    plt.subplot(2, 3, 4)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    # 5. Error over time
    plt.subplot(2, 3, 5)
    absolute_errors = np.abs(residuals)
    plt.plot(sample_indices, absolute_errors, 'g-', alpha=0.7, linewidth=1)
    plt.xlabel('Sample Index')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error Over Samples')
    plt.grid(True, alpha=0.3)
    
    # 6. Box plot comparison
    plt.subplot(2, 3, 6)
    plt.boxplot([y_val_orig, preds_original], labels=['Actual', 'Predicted'])
    plt.ylabel('Speed')
    plt.title('Speed Distribution Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_results_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print some statistics
    print(f"\nDetailed Statistics:")
    print(f"Actual Speed - Mean: {y_val_orig.mean():.4f}, Std: {y_val_orig.std():.4f}")
    print(f"Predicted Speed - Mean: {preds_original.mean():.4f}, Std: {preds_original.std():.4f}")
    print(f"Correlation: {np.corrcoef(y_val_orig, preds_original)[0,1]:.4f}")
    
    # Show worst and best predictions
    errors = np.abs(residuals)
    worst_idx = np.argmax(errors)
    best_idx = np.argmin(errors)
    
    print(f"\nWorst prediction: Actual={y_val_orig[worst_idx]:.4f}, Predicted={preds_original[worst_idx]:.4f}, Error={errors[worst_idx]:.4f}")
    print(f"Best prediction: Actual={y_val_orig[best_idx]:.4f}, Predicted={preds_original[best_idx]:.4f}, Error={errors[best_idx]:.4f}")

if __name__ == "__main__":
    main() 