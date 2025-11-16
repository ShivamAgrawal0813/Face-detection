"""
Video Face Detection Script

This script performs face detection on CCTV video footage using a trained YOLO model.
It processes each frame, draws bounding boxes, and saves the output video.
"""

import os
import sys
import cv2
import time
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import ensure_dir


def detect_faces_in_video(
    model_path: str,
    input_video: str,
    output_video: str = 'outputs/output_video.mp4',
    conf_threshold: float = 0.25,
    save_samples: bool = True,
    sample_interval: int = 100
):
    """
    Detect faces in a video file and save annotated output.
    
    Args:
        model_path: Path to trained YOLO model
        input_video: Path to input video file
        output_video: Path to output video file
        conf_threshold: Confidence threshold for detections
        save_samples: Whether to save sample frames
        sample_interval: Save sample every N frames
    """
    print("=" * 60)
    print("Face Detection in Video")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please train the model first: python main.py --train"
        )
    
    # Check if input video exists
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    
    # Create output directory
    output_dir = os.path.dirname(output_video)
    ensure_dir(output_dir)
    
    if save_samples:
        samples_dir = os.path.join(output_dir, 'samples')
        ensure_dir(samples_dir)
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    print(f"Opening video: {input_video}")
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Confidence threshold: {conf_threshold}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Could not create output video: {output_video}")
    
    print(f"\nOutput video: {output_video}")
    print(f"\n{'='*60}")
    print("Processing video...")
    print(f"{'='*60}\n")
    
    # Statistics
    frame_count = 0
    total_faces = 0
    inference_times = []
    start_time = time.time()
    
    # Process each frame
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        inference_start = time.time()
        results = model(frame, conf=conf_threshold, verbose=False)
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Count detections
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        total_faces += num_detections
        
        # Add FPS and detection count overlay
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f} | Faces: {num_detections}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Save sample frames
        if save_samples and frame_count % sample_interval == 0:
            sample_path = os.path.join(samples_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(sample_path, annotated_frame)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    
    # Calculate statistics
    total_time = time.time() - start_time
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    overall_fps = frame_count / total_time if total_time > 0 else 0
    
    # Release resources
    cap.release()
    out.release()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Processing Summary")
    print(f"{'='*60}")
    print(f"  - Total frames processed: {frame_count}")
    print(f"  - Total faces detected: {total_faces}")
    print(f"  - Average faces per frame: {total_faces / frame_count:.2f}" if frame_count > 0 else "  - Average faces per frame: 0.00")
    print(f"  - Average inference time: {avg_inference_time*1000:.2f} ms")
    print(f"  - Average FPS: {avg_fps:.2f}")
    print(f"  - Overall processing FPS: {overall_fps:.2f}")
    print(f"  - Total processing time: {total_time:.2f} seconds")
    print(f"  - Output video: {output_video}")
    if save_samples:
        print(f"  - Sample frames saved to: {samples_dir}")
    print(f"{'='*60}")
    
    print("\n✓ Video processing completed successfully!")


def main():
    """Main function for video detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect faces in video')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Path to trained model')
    parser.add_argument('--output', type=str, default='outputs/output_video.mp4', help='Path to output video')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--no-samples', action='store_true', help='Disable saving sample frames')
    
    args = parser.parse_args()
    
    # Define paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model_path = os.path.join(project_root, args.model)
    input_video = args.video if os.path.isabs(args.video) else os.path.join(project_root, args.video)
    output_video = args.output if os.path.isabs(args.output) else os.path.join(project_root, args.output)
    
    detect_faces_in_video(
        model_path=model_path,
        input_video=input_video,
        output_video=output_video,
        conf_threshold=args.conf,
        save_samples=not args.no_samples
    )


if __name__ == '__main__':
    main()

