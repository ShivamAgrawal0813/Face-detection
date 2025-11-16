# Face Detection in CCTV Footage using YOLOv11n

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![YOLO](https://img.shields.io/badge/YOLO-v11n-brightgreen)](https://github.com/ultralytics/ultralytics)

A production-ready Python project for detecting faces in CCTV video footage and live camera feeds using the WIDER FACE dataset and YOLOv11n (Ultralytics).

## 🎯 Features

- **Dataset Conversion**: Automated conversion from WIDER FACE to YOLO format
- **Model Training**: Train YOLOv11n on custom face detection dataset
- **Video Detection**: Detect faces in CCTV footage with bounding boxes
- **Real-time Detection**: Live face detection from webcam with adjustable confidence thresholds
- **GPU Support**: Automatic CUDA detection and GPU acceleration
- **Comprehensive Validation**: Pre-training dataset validation to prevent common issues
- **Performance Metrics**: Detailed mAP, Precision, and Recall metrics

## 📁 Project Structure

```
facedet/
├── data/                          # Dataset directory
│   ├── widerface/                 # Original WIDER FACE dataset
│   │   ├── train/                 # Training images and labels
│   │   └── val/                   # Validation images and labels
│   ├── images/                    # Converted YOLO format images
│   │   ├── train/
│   │   └── val/
│   ├── labels/                    # Converted YOLO format labels
│   │   ├── train/
│   │   └── val/
│   └── data.yaml                  # YOLO dataset configuration
├── models/                        # Pre-trained and trained models
│   ├── yolov11n.pt               # Pre-trained YOLOv11n
│   └── best.pt                   # Best trained model
├── outputs/                       # Training and inference outputs
│   ├── train/                    # Training results and metrics
│   ├── samples/                  # Sample detection frames
│   └── output_video.mp4          # Annotated output video
├── scripts/                       # Utility scripts
│   ├── convert_to_yolo.py        # Dataset conversion script
│   ├── train_yolo.py             # Training script
│   ├── detect_video.py           # Video detection script
│   ├── detect_realtime.py        # Real-time detection script
│   ├── validate_dataset.py       # Dataset validation
│   ├── plot_training.py          # Training visualization
│   └── utils.py                  # Utility functions
├── main.py                        # Main entry point
├── test_label_conversion.py       # Label conversion tests
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Setup

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for training, optional for inference)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd facedet
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Ensure YOLOv11n model is in the models directory:**
   - The model file should be at `models/yolov11n.pt` or `models/yolo11n.pt`
   - If not present, Ultralytics will automatically download it on first use

## 📖 Usage

### Step 0: Validate Label Conversion (Optional but Recommended)

Before converting the full dataset, test if your `label.txt` files can be correctly converted:

```bash
python test_label_conversion.py
```

Or use the validation script:

```bash
python scripts/validate_dataset.py --test-conversion --samples 10
```

This will verify that:
- Labels can be parsed correctly
- Bounding boxes convert to valid YOLO format
- Coordinates are properly normalized
- The dataset will produce valid mAP metrics after training

### Step 1: Convert Dataset to YOLO Format

Convert the WIDER FACE dataset from its original format to YOLO format:

```bash
python main.py --convert
```

This script will:
- Read `label.txt` files from `data/widerface/train/` and `data/widerface/val/`
- Convert bounding boxes to YOLO format (normalized coordinates)
- Save converted images to `data/images/train/` and `data/images/val/`
- Save YOLO labels to `data/labels/train/` and `data/labels/val/`
- Create `data/data.yaml` configuration file

**Expected Output:**
- Converted images and labels in YOLO format
- `data/data.yaml` configuration file

### Step 2: Train the Model

Train YOLOv11n on the converted dataset:

```bash
python main.py --train
```

**Optional: Validate dataset before training** (recommended to avoid 0 mAP issues):

```bash
python main.py --train --validate
```

This will:
- Test label conversion on samples
- Validate converted YOLO labels
- Check for corrupted or invalid labels
- Ensure the dataset is ready for training

This script will:
- Automatically detect GPU availability
- Load the pre-trained YOLOv11n model
- Train for 100 epochs (default)
- Save the best model to `models/best.pt`
- Generate training plots (loss curves, mAP curves) in `outputs/train/`

**Training Parameters:**
- Epochs: 100
- Image size: 640x640
- Batch size: 16
- Device: Auto-detected (CUDA if available, else CPU)

**Expected Output:**
- Trained model: `models/best.pt`
- Training plots: `outputs/train/`
- Metrics: mAP50, mAP50-95, Precision, Recall

### Step 3: Detect Faces in Video

Run face detection on a CCTV video:

```bash
python main.py --detect --video path/to/input_video.mp4
```

**Options:**
- `--video`: Path to input video file (required)
- `--model`: Path to trained model (default: `models/best.pt`)
- `--output`: Path to output video (default: `outputs/output_video.mp4`)
- `--conf`: Confidence threshold (default: 0.25)

**Example with custom settings:**
```bash
python main.py --detect --video input_video.mp4 --conf 0.5 --output my_output.mp4
```

**Expected Output:**
- Annotated video: `outputs/output_video.mp4`
- Sample frames: `outputs/samples/`
- Console output with FPS and detection statistics

### Step 4: Real-Time Face Detection

Perform real-time face detection from your webcam:

```bash
python main.py --realtime
```

**Options:**
- `--realtime`: Enable real-time detection from camera
- `--camera`: Camera device ID (default: 0 for first camera)
- `--model`: Path to trained model (default: `models/best.pt`)
- `--conf`: Confidence threshold (default: 0.25)
- `--save-video`: Save output video to file

**Example with custom settings:**
```bash
python main.py --realtime --camera 0 --conf 0.5 --save-video
```

**Controls:**
- Press `q` to quit
- Press `s` to save a screenshot
- Press `+` or `=` to increase confidence threshold
- Press `-` or `_` to decrease confidence threshold

**Expected Output:**
- Live video window with face detection annotations
- Real-time FPS and detection count display
- Optional video recording: `outputs/realtime_output.mp4`
- Screenshots saved when pressing 's'

## 📊 Model Performance

After training, the model will display metrics including:

- **mAP50**: Mean Average Precision at IoU=0.50
- **mAP50-95**: Mean Average Precision at IoU=0.50:0.95
- **Precision**: Ratio of true positives to all detections
- **Recall**: Ratio of true positives to all ground truth faces

These metrics are displayed in the console and saved in the training results.

## 🔧 Advanced Usage

### Using Individual Scripts

You can also run the scripts directly:

```bash
# Convert dataset
python scripts/convert_to_yolo.py

# Train model
python scripts/train_yolo.py

# Detect in video
python scripts/detect_video.py --video input.mp4 --model models/best.pt

# Real-time detection
python scripts/detect_realtime.py --model models/best.pt --camera 0
```

### Custom Training Parameters

To modify training parameters, edit `scripts/train_yolo.py` or `main.py`:

```python
train_model(
    epochs=150,      # Increase epochs
    batch=32,        # Increase batch size (if GPU memory allows)
    imgsz=1280,      # Larger image size for better accuracy
)
```

## 📝 Dataset Format

### WIDER FACE Format (Input)

The original WIDER FACE `label.txt` format:
```
# image_path
x1 y1 width height blur expression illumination invalid occlusion pose
...
```

### YOLO Format (Output)

Converted YOLO format:
```
class_id x_center y_center width height
0 0.5 0.5 0.2 0.3
```

All coordinates are normalized between 0 and 1.

## ✅ Validation

Before training, it's highly recommended to validate your dataset to ensure labels are correctly converted and will produce valid mAP metrics:

```bash
# Quick test of label conversion
python test_label_conversion.py

# Full validation of converted dataset
python scripts/validate_dataset.py --validate-dataset --samples 20

# Or validate before training
python main.py --train --validate
```

This helps prevent issues like:
- 0 mAP after training (corrupted labels)
- Invalid coordinate ranges
- Missing label files
- Format errors

## 🛠️ Troubleshooting

### Common Issues

1. **0 mAP after training:**
   - Run validation: `python test_label_conversion.py`
   - Check if labels were converted correctly: `python scripts/validate_dataset.py --validate-dataset`
   - Ensure `data/data.yaml` points to correct paths
   - Verify label files exist in `data/labels/train/` and `data/labels/val/`

2. **Model file not found:**
   - Ensure `models/yolov11n.pt` exists
   - Ultralytics will auto-download on first use if not present

3. **CUDA out of memory:**
   - Reduce batch size in `train_yolo.py` (e.g., `batch=8`)
   - Reduce image size (e.g., `imgsz=512`)

4. **Dataset conversion errors:**
   - Ensure `data/widerface/train/label.txt` and `data/widerface/val/label.txt` exist
   - Check that image paths in label files are correct

5. **Video processing is slow:**
   - Use GPU for inference (automatically detected)
   - Reduce video resolution if needed
   - Lower confidence threshold to reduce processing time

## 📦 Dependencies

- `ultralytics>=8.0.0` - YOLO model framework
- `opencv-python>=4.8.0` - Video and image processing
- `torch>=2.0.0` - Deep learning framework
- `tqdm>=4.66.0` - Progress bars
- `matplotlib>=3.7.0` - Plotting training curves
- `numpy>=1.24.0` - Numerical operations
- `pyyaml>=6.0` - YAML configuration files

## 🎓 Features

- ✅ Automatic dataset conversion from WIDER FACE to YOLO format
- ✅ GPU auto-detection for training and inference
- ✅ Real-time video processing with progress tracking
- ✅ **Real-time face detection from webcam/camera**
- ✅ Sample frame extraction during video processing
- ✅ Comprehensive training metrics and plots
- ✅ FPS and inference time monitoring
- ✅ Confidence threshold adjustment (adjustable in real-time)
- ✅ Screenshot capture during real-time detection
- ✅ Video recording of real-time detection
- ✅ Well-documented code with inline comments

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📧 Support

For issues or questions, please check the troubleshooting section or open an issue in the repository.

---

**Note**: This project uses the WIDER FACE dataset for training. Ensure you have proper permissions and comply with dataset usage terms.

