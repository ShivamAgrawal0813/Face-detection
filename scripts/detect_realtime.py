"""
Real-Time Face Detection Script

This script performs face detection on live camera feed using a trained YOLO model.
It processes frames in real-time and displays annotated results.
"""

import os
import sys
import cv2
import time
from ultralytics import YOLO

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import ensure_dir


def detect_faces_realtime(
    model_path: str,
    camera_id: int = 0,
    conf_threshold: float = 0.25,
    save_output: bool = False,
    output_video: str = 'outputs/realtime_output.mp4'
):
    """
    Detect faces in real-time from camera feed.
    
    Args:
        model_path: Path to trained YOLO model
        camera_id: Camera device ID (default: 0 for first camera)
        conf_threshold: Confidence threshold for detections
        save_output: Whether to save the output video
        output_video: Path to output video file (if save_output is True)
    """
    print("=" * 60)
    print("Real-Time Face Detection")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please train the model first: python main.py --train"
        )
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_id}. Please check if camera is connected.")
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if FPS not available
    
    print(f"Camera properties:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - Confidence threshold: {conf_threshold}")
    print(f"\nPress 'q' to quit, 's' to save screenshot")
    
    # Setup video writer if saving output
    out = None
    if save_output:
        output_dir = os.path.dirname(output_video)
        ensure_dir(output_dir)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        print(f"  - Saving output to: {output_video}")
    
    print(f"\n{'='*60}")
    print("Starting real-time detection...")
    print(f"{'='*60}\n")
    
    # Statistics
    frame_count = 0
    total_faces = 0
    inference_times = []
    start_time = time.time()
    fps_display = 0
    
    # Window name
    window_name = "Face Detection - Real-Time"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Warning: Could not read frame from camera")
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
            frame_count += 1
            
            # Calculate FPS (moving average)
            current_fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_display = 0.9 * fps_display + 0.1 * current_fps  # Exponential moving average
            
            # Add information overlay
            info_text = [
                f"FPS: {fps_display:.1f}",
                f"Faces: {num_detections}",
                f"Total: {total_faces}",
                f"Frame: {frame_count}",
                f"Conf: {conf_threshold:.2f}"
            ]
            
            # Draw semi-transparent background for text
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (250, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            
            # Draw text
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(annotated_frame, text, (20, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add instructions at bottom
            cv2.putText(annotated_frame, "Press 'q' to quit, 's' to save screenshot",
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save frame if recording
            if out is not None:
                out.write(annotated_frame)
            
            # Display frame
            cv2.imshow(window_name, annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✓ Quitting...")
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = os.path.join(os.path.dirname(output_video) if save_output else 'outputs',
                                               f'screenshot_{int(time.time())}.jpg')
                ensure_dir(os.path.dirname(screenshot_path))
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"✓ Screenshot saved: {screenshot_path}")
            elif key == ord('+') or key == ord('='):
                # Increase confidence threshold
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"Confidence threshold: {conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                # Decrease confidence threshold
                conf_threshold = max(0.05, conf_threshold - 0.05)
                print(f"Confidence threshold: {conf_threshold:.2f}")
    
    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # Calculate and print statistics
        total_time = time.time() - start_time
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        overall_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n{'='*60}")
        print("Real-Time Detection Summary")
        print(f"{'='*60}")
        print(f"  - Total frames processed: {frame_count}")
        print(f"  - Total faces detected: {total_faces}")
        if frame_count > 0:
            print(f"  - Average faces per frame: {total_faces / frame_count:.2f}")
        print(f"  - Average inference time: {avg_inference_time*1000:.2f} ms")
        print(f"  - Average FPS: {avg_fps:.2f}")
        print(f"  - Overall processing FPS: {overall_fps:.2f}")
        print(f"  - Total processing time: {total_time:.2f} seconds")
        if save_output and out is not None:
            print(f"  - Output video saved: {output_video}")
        print(f"{'='*60}")


def main():
    """Main function for real-time detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time face detection from camera')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save output video')
    parser.add_argument('--output', type=str, default='outputs/realtime_output.mp4',
                       help='Path to output video file')
    
    args = parser.parse_args()
    
    # Define paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model_path = args.model if os.path.isabs(args.model) else os.path.join(project_root, args.model)
    output_video = args.output if os.path.isabs(args.output) else os.path.join(project_root, args.output)
    
    detect_faces_realtime(
        model_path=model_path,
        camera_id=args.camera,
        conf_threshold=args.conf,
        save_output=args.save,
        output_video=output_video
    )


if __name__ == '__main__':
    main()

