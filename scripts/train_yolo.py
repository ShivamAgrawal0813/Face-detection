"""
YOLO Training Script for Face Detection

This script trains a YOLOv11n model on the converted WIDER FACE dataset.
It automatically detects GPU availability and saves the best model.
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import ensure_dir
from scripts.plot_training import plot_training_curves


def train_model(
    model_path: str = 'models/yolov11n.pt',
    data_yaml: str = 'data/data.yaml',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = 'outputs',
    name: str = 'train',
    device: str = None
):
    """
    Train YOLOv11n model for face detection.
    
    Args:
        model_path: Path to pre-trained YOLOv11n model
        data_yaml: Path to data.yaml configuration file
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        project: Project directory for outputs
        name: Experiment name
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    """
    print("=" * 60)
    print("YOLOv11n Face Detection Training")
    print("=" * 60)
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            device = 'cpu'
            print("⚠ No GPU detected, using CPU (training will be slower)")
    
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        # Try alternative path
        alt_path = model_path.replace('yolov11n.pt', 'yolo11n.pt')
        if os.path.exists(alt_path):
            model_path = alt_path
            print(f"Using alternative model path: {model_path}")
        else:
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please ensure the YOLOv11n model is downloaded to the models/ directory."
            )
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"Data configuration file not found: {data_yaml}\n"
            f"Please run the conversion script first: python main.py --convert"
        )
    
    # Create output directories
    ensure_dir(project)
    ensure_dir(os.path.join(project, name))
    
    print(f"\nLoading model from: {model_path}")
    model = YOLO(model_path)
    
    print(f"\nTraining configuration:")
    print(f"  - Data: {data_yaml}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Batch size: {batch}")
    print(f"  - Device: {device}")
    print(f"  - Project: {project}")
    print(f"  - Name: {name}")
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        save=True,
        plots=True,  # Generate training plots
        val=True,   # Run validation during training
    )
    
    # Get the best model path
    best_model_path = os.path.join(project, name, 'weights', 'best.pt')
    
    # Copy best model to models directory
    if os.path.exists(best_model_path):
        import shutil
        models_dir = os.path.dirname(model_path)
        ensure_dir(models_dir)
        final_best_path = os.path.join(models_dir, 'best.pt')
        shutil.copy2(best_model_path, final_best_path)
        print(f"\n✓ Best model saved to: {final_best_path}")
    else:
        print(f"\n⚠ Best model not found at expected path: {best_model_path}")
    
    # Print training results summary
    print(f"\n{'='*60}")
    print("Training Results Summary")
    print(f"{'='*60}")
    
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
        print(f"  Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
    
    # Generate training plots from results CSV
    results_csv_path = os.path.join(project, name, 'results.csv')
    if os.path.exists(results_csv_path):
        print(f"\n{'='*60}")
        print("Generating Training Plots...")
        print(f"{'='*60}")
        try:
            plot_training_curves(results_csv_path, os.path.join(project, name))
        except Exception as e:
            print(f"⚠ Warning: Could not generate plots: {e}")
            print("  You can generate plots manually by running:")
            print(f"  python scripts/plot_training.py --results {results_csv_path}")
    else:
        print(f"\n⚠ Results CSV not found at: {results_csv_path}")
        print("  Plots cannot be generated without results.csv")
    
    print(f"\n✓ Training completed!")
    print(f"  - Training plots saved to: {os.path.join(project, name)}")
    print(f"  - Best model: {best_model_path}")
    print(f"{'='*60}")


def main():
    """Main function for training."""
    # Define paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model_path = os.path.join(project_root, 'models', 'yolov11n.pt')
    data_yaml = os.path.join(project_root, 'data', 'data.yaml')
    project = os.path.join(project_root, 'outputs')
    
    # Training parameters
    epochs = 100
    imgsz = 640
    batch = 16
    
    train_model(
        model_path=model_path,
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name='train'
    )


if __name__ == '__main__':
    main()

