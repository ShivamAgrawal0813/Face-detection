"""
Main CLI Script for Face Detection Project

This script provides a command-line interface to run dataset conversion,
model training, and video detection tasks.
"""

import os
import sys
import argparse
from pathlib import Path

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.convert_to_yolo import main as convert_main
from scripts.train_yolo import train_model
from scripts.detect_video import detect_faces_in_video
from scripts.detect_realtime import detect_faces_realtime
from scripts.validate_dataset import test_conversion_sample, validate_yolo_dataset


def convert_dataset():
    """Convert WIDER FACE dataset to YOLO format."""
    print("\n" + "="*60)
    print("DATASET CONVERSION")
    print("="*60 + "\n")
    convert_main()


def train():
    """Train YOLOv11n model on the converted dataset."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(project_root, 'models', 'yolov11n.pt')
    data_yaml = os.path.join(project_root, 'data', 'data.yaml')
    project = os.path.join(project_root, 'outputs')
    
    # Training parameters
    epochs = 100
    imgsz = 640
    batch = 16
    
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60 + "\n")
    
    train_model(
        model_path=model_path,
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name='train'
    )


def detect(video_path: str, model_path: str = None, output_path: str = None, conf: float = 0.25):
    """Detect faces in a video file."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    if model_path is None:
        model_path = os.path.join(project_root, 'models', 'best.pt')
    elif not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
    
    if output_path is None:
        output_path = os.path.join(project_root, 'outputs', 'output_video.mp4')
    elif not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    
    if not os.path.isabs(video_path):
        video_path = os.path.join(project_root, video_path)
    
    print("\n" + "="*60)
    print("VIDEO DETECTION")
    print("="*60 + "\n")
    
    detect_faces_in_video(
        model_path=model_path,
        input_video=video_path,
        output_video=output_path,
        conf_threshold=conf,
        save_samples=True
    )


def detect_realtime(model_path: str = None, camera_id: int = 0, conf: float = 0.25, save: bool = False):
    """Detect faces in real-time from camera."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    if model_path is None:
        model_path = os.path.join(project_root, 'models', 'best.pt')
    elif not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
    
    output_video = os.path.join(project_root, 'outputs', 'realtime_output.mp4')
    
    print("\n" + "="*60)
    print("REAL-TIME DETECTION")
    print("="*60 + "\n")
    
    detect_faces_realtime(
        model_path=model_path,
        camera_id=camera_id,
        conf_threshold=conf,
        save_output=save,
        output_video=output_video
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Face Detection in CCTV Footage using YOLOv11n',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --convert                    # Convert dataset to YOLO format
  python main.py --train                      # Train the model
  python main.py --detect --video input.mp4   # Detect faces in video
  python main.py --detect --video input.mp4 --conf 0.5  # With custom confidence
  python main.py --realtime                   # Real-time detection from camera
  python main.py --realtime --save-video     # Real-time with video recording
  python main.py --realtime --camera 1       # Use different camera device
        """
    )
    
    parser.add_argument(
        '--convert',
        action='store_true',
        help='Convert WIDER FACE dataset to YOLO format'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train YOLOv11n model on the dataset'
    )
    
    parser.add_argument(
        '--detect',
        action='store_true',
        help='Detect faces in a video file'
    )
    
    parser.add_argument(
        '--realtime',
        action='store_true',
        help='Real-time face detection from camera'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to input video file (required for --detect)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID for real-time detection (default: 0)'
    )
    
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Save real-time detection output to video file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/best.pt',
        help='Path to trained model (default: models/best.pt)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output video file (default: outputs/output_video.mp4)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detection (default: 0.25)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate dataset conversion before training'
    )
    
    args = parser.parse_args()
    
    # Check if at least one action is specified
    if not (args.convert or args.train or args.detect or args.realtime):
        parser.print_help()
        sys.exit(1)
    
    # Execute requested actions
    if args.convert:
        convert_dataset()
    
    if args.train:
        # Optionally validate dataset before training
        if args.validate:
            project_root = os.path.dirname(os.path.abspath(__file__))
            data_yaml = os.path.join(project_root, 'data', 'data.yaml')
            widerface_dir = os.path.join(project_root, 'data', 'widerface')
            
            print("\n" + "="*60)
            print("VALIDATING DATASET BEFORE TRAINING")
            print("="*60 + "\n")
            
            # Test conversion on samples
            label_file = os.path.join(widerface_dir, 'train', 'label.txt')
            images_dir = os.path.join(widerface_dir, 'train', 'images')
            
            if os.path.exists(label_file):
                print("Testing label conversion...")
                conversion_ok = test_conversion_sample(label_file, images_dir, num_samples=10)
                if not conversion_ok:
                    print("\n❌ Conversion test failed! Please check your dataset.")
                    sys.exit(1)
            
            # Validate converted dataset
            if os.path.exists(data_yaml):
                print("\nValidating converted dataset...")
                dataset_ok = validate_yolo_dataset(data_yaml, split='train', num_samples=20)
                if not dataset_ok:
                    print("\n❌ Dataset validation failed! Please check your converted dataset.")
                    sys.exit(1)
                print("\n✓ Dataset validation passed! Proceeding with training...\n")
            else:
                print("⚠ data.yaml not found. Run --convert first.")
        
        train()
    
    if args.detect:
        if not args.video:
            print("Error: --video is required when using --detect")
            sys.exit(1)
        detect(
            video_path=args.video,
            model_path=args.model,
            output_path=args.output,
            conf=args.conf
        )
    
    if args.realtime:
        detect_realtime(
            model_path=args.model,
            camera_id=args.camera,
            conf=args.conf,
            save=args.save_video
        )


if __name__ == '__main__':
    main()

