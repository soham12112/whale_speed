# Whale Speed Estimation with Machine Learning

A machine learning system for estimating whale movement speed from video footage using optical flow analysis and convolutional neural networks.

## Overview

This project uses optical flow (motion between consecutive video frames) to predict whale movement speed. The system consists of three main components:

1. **Data Preparation**: Extract frames from whale videos and pair them with speed labels
2. **Optical Flow Extraction**: Use PTL-Flow library with DPFlow model to compute optical flow
3. **CNN Training**: Train various CNN architectures to predict speed from optical flow data

## Project Structure

```
Whale_Speed_Estimation_with_ML/
├── extract_flow_with_speed_labels_new.py    # Main data preparation script
├── train_cnn_speed_regressor.py             # Basic CNN training
├── improved_cnn_speed_regressor.py          # Enhanced CNN with better architecture
├── temporal_cnn_speed_regressor.py          # Advanced temporal CNN using sequences
├── plot_results.py                          # Results visualization
├── ptlflow/                                 # PTL-Flow optical flow library
├── checkpoints/                             # Pre-trained model checkpoints
├── requirements.txt                         # Python dependencies
└── README.md                               # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)
- Git

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Whale_Speed_Estimation_with_ML
```

### Step 2: Create Virtual Environment

```bash
python -m venv whale_env
source whale_env/bin/activate  # On Windows: whale_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install PTL-Flow

PTL-Flow is the core optical flow library. Install it using one of these methods:

**Option A: From PyPI (Recommended)**
```bash
pip install ptlflow
```

**Option B: From Source (Latest Features)**
```bash
pip install git+https://github.com/hmorimitsu/ptlflow.git
```

### Step 5: Download Pre-trained Checkpoints

Download the DPFlow checkpoint file:
```bash
mkdir -p checkpoints
wget https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dpflow-sintel-b44b072c.ckpt -O checkpoints/dpflow-sintel-b44b072c.ckpt
```

Or download manually from the [PTL-Flow releases page](https://github.com/hmorimitsu/ptlflow/releases).

## Data Preparation

### Input Data Format

You need two files:
1. **Video file**: MP4 format containing whale footage (e.g., `vid5_1min.mp4`)
2. **CSV file**: Speed labels with timestamps (e.g., `vid5_1min.csv`)

### CSV Format

The CSV file should contain at least these columns:
- `timestamp`: Time in seconds or MM:SS.sss format
- `speed`: Speed values (numerical)

Example:
```csv
timestamp,speed
0.0,2.5
0.5,2.7
1.0,3.1
1.5,3.0
```

## Usage

### Step 1: Extract Optical Flow with Speed Labels

This is the main data preparation step that extracts frames, computes optical flow, and creates training data:

```bash
python extract_flow_with_speed_labels_new.py
```

**Configuration**: Edit the file to specify your input files:
```python
video_path = "your_video.mp4"
csv_path = "your_labels.csv"
ckpt_path = "checkpoints/dpflow-sintel-b44b072c.ckpt"
```

**Output**: Creates `labeled_data/` directory with:
- `frames/`: Extracted video frames
- `optical_flow/`: Optical flow .npy files
- `flow_visualizations/`: Flow visualization images
- `labels/`: CSV files with frame pair labels

### Step 2: Train CNN Models

You can train different CNN architectures:

#### Basic CNN
```bash
python train_cnn_speed_regressor.py
```

#### Improved CNN (Recommended)
```bash
python improved_cnn_speed_regressor.py
```

#### Temporal CNN (Advanced)
```bash
python temporal_cnn_speed_regressor.py
```

**Configuration**: Each script has configuration constants at the top:
```python
FLOW_DIR = 'labeled_data/optical_flow'     # Input optical flow directory
SPEED_CSV = 'actual_speeds.csv'            # Optional: actual speed CSV
MODEL_OUT = 'model_name.pt'                # Output model file
IMG_SIZE = 128                             # Input image size
BATCH_SIZE = 32                            # Training batch size
NUM_EPOCHS = 50                            # Number of training epochs
LR = 5e-4                                  # Learning rate
```

### Step 3: Visualize Results

```bash
python plot_results.py
```

This generates comprehensive plots showing:
- Training loss curves
- Actual vs predicted speed comparisons
- Residual analysis
- Performance metrics

## Model Architectures

### Basic CNN (`train_cnn_speed_regressor.py`)
- Simple 2D CNN with pooling layers
- Input: 2-channel optical flow (u, v components)
- Output: Single speed prediction

### Improved CNN (`improved_cnn_speed_regressor.py`)
- Enhanced architecture with:
  - 4-channel input (u, v, magnitude, direction)
  - Batch normalization
  - Attention mechanism
  - Data augmentation
  - Early stopping

### Temporal CNN (`temporal_cnn_speed_regressor.py`)
- 3D CNN for temporal sequences
- Uses multiple consecutive frames
- Temporal attention mechanism
- Huber loss for robust training

## Configuration Options

### Data Processing
- `IMG_SIZE`: Resolution for flow processing (64, 128, 256)
- `TEMPORAL_WINDOW`: Number of consecutive frames for temporal models

### Training
- `BATCH_SIZE`: Adjust based on GPU memory
- `NUM_EPOCHS`: Maximum training epochs
- `LR`: Learning rate (3e-4 to 1e-3 typically work well)
- `WEIGHT_DECAY`: L2 regularization strength

### Hardware
- CUDA GPU: Automatically detected and used if available
- CPU: Falls back to CPU training (slower)

## Expected Output

### Data Preparation
```
STEP 1: Extracting frames and creating speed labels
  saved 100 frames …
  saved 200 frames …
✓ Created 1500 pairs → labeled_data/labels/frame_pairs_labels.csv

STEP 2: Extracting optical flow on 1500 pairs
  done 0/1500
  done 50/1500
✓ Flow files saved in labeled_data/optical_flow
```

### Training
```
Found 1500 flow files
Enhanced Dataset: (1500, 4, 128, 128) (1500,)
Using device: cuda
Starting enhanced training...
Epoch 01/50 | train 0.0245 | val 0.0198 | lr 0.000500
Epoch 02/50 | train 0.0189 | val 0.0156 | lr 0.000500
...
Early stopping at epoch 32

Final Metrics:
MSE: 0.0034
MAE: 0.0451
RMSE: 0.0583
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE` or `IMG_SIZE`
   - Use CPU training: `dev = torch.device('cpu')`

2. **PTL-Flow Installation Issues**
   - Install from source: `pip install git+https://github.com/hmorimitsu/ptlflow.git`
   - Check CUDA compatibility

3. **No Flow Files Found**
   - Ensure `extract_flow_with_speed_labels_new.py` completed successfully
   - Check `FLOW_DIR` path in training scripts

4. **Video Loading Issues**
   - Ensure video file is in a supported format (MP4, AVI, MOV)
   - Check video file path and accessibility

### Performance Tips

1. **Use GPU**: Significant speedup for training
2. **Batch Size**: Larger batches generally improve stability
3. **Data Augmentation**: Enabled in improved/temporal models
4. **Early Stopping**: Prevents overfitting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit a pull request

## License

Private 

## Acknowledgments

- [PTL-Flow](https://github.com/hmorimitsu/ptlflow) for optical flow estimation
- DPFlow model for high-quality optical flow computation
- PyTorch Lightning for training framework

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{whale_speed_estimation,
  title={Whale Speed Estimation with Machine Learning},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
``` 

### Note on Large Files
Due to GitHub's file size limitations, the following files are not included in this repository:
- Pre-trained checkpoints (download separately from PTL-Flow releases)
- Video data files (provide your own whale footage)
- Generated model files (created during training)

Please follow the installation instructions to download the required checkpoint files. 