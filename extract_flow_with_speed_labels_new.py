#!/usr/bin/env python3
"""
Complete script to extract optical flow with speed labels for CNN training.
Fixed version with PTL-Flow API corrections and minor clean‑ups.
"""

import sys
import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. PTL‑Flow setup
# -----------------------------------------------------------------------------
ptlflow_path = os.path.join(os.getcwd(), "ptlflow")
sys.path.insert(0, ptlflow_path)
import ptlflow  # noqa: E402

# -----------------------------------------------------------------------------
# 2. Helper functions
# -----------------------------------------------------------------------------

def preprocess(frame):
    """Convert BGR frame → normalized tensor 1×3×H×W."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (512, 256))  # keep consistent resolution
    frame = frame.astype(np.float32) / 255.0
    tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
    return tensor


def flow_to_rgb(flow):
    """Visualize (2,H,W) optical‑flow as RGB."""
    # Ensure flow is in the right format
    if len(flow.shape) == 3 and flow.shape[0] == 2:
        # Flow is already (2,H,W) format
        flow = np.transpose(flow, (1, 2, 0))  # (H,W,2)
    elif len(flow.shape) == 4 and flow.shape[1] == 2:
        # Flow is (1,2,H,W) format, remove batch dimension
        flow = flow[0].transpose(1, 2, 0)  # (H,W,2)
    else:
        raise ValueError(f"Unexpected flow shape: {flow.shape}")
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

# -----------------------------------------------------------------------------
# 3. Pipeline steps
# -----------------------------------------------------------------------------

def extract_frames_with_speed_labels(video_path: str, csv_path: str, output_dir: str = "labeled_data"):
    """Step‑1: Extract all frames and pair each with nearest speed record."""

    print("=" * 60)
    print("STEP 1: Extracting frames and creating speed labels")
    print("=" * 60)

    # Convert timestamps (MM:SS.sss or float sec) → float seconds
    def to_seconds(x):
        if ":" in str(x):
            # Handle MM:SS.sss format
            parts = str(x).split(":")
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            # Handle float seconds
            return float(x)

    # Directories
    frames_dir = os.path.join(output_dir, "frames")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Load CSV
    #df = pd.read_csv(csv_path)
    # after reading CSV
    df = pd.read_csv(csv_path)
    df['timestamp'] = df['timestamp'].apply(to_seconds)

    # ---- NEW: shift to start at 0 s ----
    df['timestamp'] -= df['timestamp'].iloc[0]

    if {"timestamp", "speed"}.issubset(df.columns) is False:
        raise ValueError("CSV must contain 'timestamp' & 'speed' columns")

    print(df[['timestamp', 'speed']].head(10))
    print("CSV time range:", df['timestamp'].min(), "→", df['timestamp'].max())
    
    #df['timestamp'] = df['timestamp'].apply(to_seconds)

    print(df[['timestamp','speed']].head())       # should now show numeric seconds
    print("CSV time range (s):", df['timestamp'].min(), "→", df['timestamp'].max())

    # Video reader
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_speed_pairs = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_sec = idx / fps
        # nearest ground‑truth speed
        nearest = df.iloc[(df["timestamp"] - t_sec).abs().idxmin()]["speed"]
        frame_path = os.path.join(frames_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_speed_pairs.append({"frame_idx": idx, "frame_path": frame_path, "speed": nearest, "timestamp": t_sec})
        idx += 1
        if idx % 100 == 0:
            print(f"  saved {idx} frames …")

    cap.release()

    # Consecutive pairs → labels
    pairs = []
    for i in range(len(frame_speed_pairs) - 1):
        f1, f2 = frame_speed_pairs[i], frame_speed_pairs[i + 1]
        pairs.append({
            "pair_idx": i,
            "frame1_path": f1["frame_path"],
            "frame2_path": f2["frame_path"],
            "speed": (f1["speed"] + f2["speed"]) / 2,
            "frame1_idx": f1["frame_idx"],
            "frame2_idx": f2["frame_idx"],
        })

    labels_df = pd.DataFrame(pairs)
    labels_csv = os.path.join(labels_dir, "frame_pairs_labels.csv")
    labels_df.to_csv(labels_csv, index=False)

    # also emit a lightweight speed‑map CSV for the notebooks
    labels_df[["pair_idx", "speed"]].to_csv(os.path.join(output_dir, "speed_map.csv"), index=False)

    print(f"✓ Created {len(pairs)} pairs → {labels_csv}")
    return labels_csv, frames_dir


def extract_flow_with_labels(labels_path: str, frames_dir: str, ckpt_path: str, output_dir: str = "labeled_data"):
    """Step‑2: Run DPFlow on each frame pair and save (flow, speed)."""

    labels_df = pd.read_csv(labels_path)
    print("Unique speeds in labels_df:", labels_df['speed'].unique()[:10], "…")
    print("min/max:", labels_df['speed'].min(), labels_df['speed'].max())

    print(f"STEP 2: Extracting optical flow on {len(labels_df)} pairs")

    # --- load model correctly ---
    model = ptlflow.get_model("dpflow", ckpt_path=ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    flow_dir = os.path.join(output_dir, "optical_flow")
    vis_dir = os.path.join(output_dir, "flow_visualizations")
    os.makedirs(flow_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    for i, row in labels_df.iterrows():
        f1 = cv2.imread(row["frame1_path"])
        f2 = cv2.imread(row["frame2_path"])
        if f1 is None or f2 is None:
            print(f"⚠ Skipping {i}: cannot read frames")
            continue

        # tensors to correct device
        t1 = preprocess(f1).to(device)
        t2 = preprocess(f2).to(device)
        
        # Stack tensors to create 5D input: (batch, num_images, channels, height, width)
        images = torch.stack([t1, t2], dim=1)  # Shape: (1, 2, 3, H, W)
        
        with torch.no_grad():
            out = model({"images": images})
        flow = out["flows"][0].cpu().numpy()  # (2,H,W)
        
        # Debug: print flow shape
        print(f"Flow shape: {flow.shape}")
        
        np.save(os.path.join(flow_dir, f"flow_{i:06d}_speed_{row['speed']:.2f}.npy"), flow)
        plt.imsave(os.path.join(vis_dir, f"flow_{i:06d}.png"), flow_to_rgb(flow))
        if i % 50 == 0:
            print(f"  done {i}/{len(labels_df)}")

    print(f"✓ Flow files saved in {flow_dir}")
    return flow_dir, vis_dir


def create_training_dataset(output_dir: str = "labeled_data"):
    """Step‑3: Simple dataset stats & summary file."""

    labels = pd.read_csv(os.path.join(output_dir, "labels", "frame_pairs_labels.csv"))
    flow_files = [f for f in os.listdir(os.path.join(output_dir, "optical_flow")) if f.endswith(".npy")]

    summary = (
        f"Total pairs: {len(labels)}\n"
        f"Flows saved: {len(flow_files)}\n"
        f"Speed min/max: {labels['speed'].min():.2f}/{labels['speed'].max():.2f}\n"
        f"Speed mean±std: {labels['speed'].mean():.2f} ± {labels['speed'].std():.2f}\n"
    )
    with open(os.path.join(output_dir, "training_info.txt"), "w") as fh:
        fh.write(summary)
    print(summary)
    return labels

# -----------------------------------------------------------------------------
# 4. CLI entry‑point
# -----------------------------------------------------------------------------

def main():
    video_path = "vid5_1min.mp4"
    csv_path = "vid5_1min.csv"
    ckpt_path = "checkpoints/dpflow-sintel-b44b072c.ckpt"
    output_dir = "labeled_data"

    os.makedirs(output_dir, exist_ok=True)

    # Steps 1‑3
    labels_csv, frames_dir = extract_frames_with_speed_labels(video_path, csv_path, output_dir)
    extract_flow_with_labels(labels_csv, frames_dir, ckpt_path, output_dir)
    create_training_dataset(output_dir)
    print("\n✓ COMPLETE – dataset ready in", output_dir)


if __name__ == "__main__":
    main()
