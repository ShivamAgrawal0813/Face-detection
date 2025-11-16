"""
Dataset Conversion Script: WIDER FACE to YOLO Format

This script converts the WIDER FACE dataset format to YOLO format.
It reads label.txt files and converts bounding boxes to normalized YOLO format.
"""

import os
import sys
from pathlib import Path
import cv2
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import ensure_dir, convert_bbox_to_yolo, find_image_path


def parse_widerface_label_file(label_file_path: str) -> dict:
    """
    Parse WIDER FACE label.txt file.
    
    Format:
        # image_path
        x1 y1 width height blur expression illumination invalid occlusion pose
        ...
        # next_image_path
        ...
    
    Args:
        label_file_path: Path to the label.txt file
        
    Returns:
        Dictionary mapping image paths to list of bounding boxes
        Each bbox is [x1, y1, width, height]
    """
    annotations = {}
    current_image = None
    
    print(f"Reading label file: {label_file_path}")
    
    with open(label_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Lines starting with # are image paths
            if line.startswith('#'):
                # Extract image path (remove # and leading space)
                current_image = line[1:].strip()
                annotations[current_image] = []
            else:
                # Parse bounding box coordinates
                # Format: x1 y1 width height [additional attributes...]
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        x1 = float(parts[0])
                        y1 = float(parts[1])
                        width = float(parts[2])
                        height = float(parts[3])
                        
                        # Skip invalid bounding boxes (width or height <= 0)
                        if width > 0 and height > 0:
                            annotations[current_image].append([x1, y1, width, height])
                    except ValueError:
                        # Skip lines that can't be parsed
                        continue
    
    print(f"Found {len(annotations)} images with annotations")
    return annotations


def convert_dataset_to_yolo(widerface_dir: str, output_dir: str, split: str = 'train'):
    """
    Convert WIDER FACE dataset to YOLO format.
    
    Args:
        widerface_dir: Path to widerface directory (contains train/val folders)
        output_dir: Output directory for YOLO format data
        split: 'train' or 'val'
    """
    # Define paths
    split_dir = os.path.join(widerface_dir, split)
    label_file = os.path.join(split_dir, 'label.txt')
    images_dir = os.path.join(split_dir, 'images')
    
    # Output directories
    output_images_dir = os.path.join(output_dir, 'images', split)
    output_labels_dir = os.path.join(output_dir, 'labels', split)
    
    # Create output directories
    ensure_dir(output_images_dir)
    ensure_dir(output_labels_dir)
    
    # Check if label file exists
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Parse label file
    annotations = parse_widerface_label_file(label_file)
    
    # Process each image
    converted_count = 0
    skipped_count = 0
    
    print(f"\nConverting {split} dataset to YOLO format...")
    
    for image_path, bboxes in tqdm(annotations.items(), desc=f"Processing {split}"):
        if not bboxes:
            skipped_count += 1
            continue
        
        # Find the actual image file
        try:
            full_image_path = find_image_path(images_dir, image_path)
        except FileNotFoundError:
            skipped_count += 1
            continue
        
        # Get image dimensions
        try:
            img = cv2.imread(full_image_path)
            if img is None:
                skipped_count += 1
                continue
            img_height, img_width = img.shape[:2]
        except Exception as e:
            print(f"Error reading image {full_image_path}: {e}")
            skipped_count += 1
            continue
        
        # Create output image path (copy structure)
        # Extract relative path from image_path
        rel_path = image_path.replace('\\', '/')  # Normalize path separators
        output_image_path = os.path.join(output_images_dir, rel_path)
        output_image_dir = os.path.dirname(output_image_path)
        ensure_dir(output_image_dir)
        
        # Copy image to output directory
        import shutil
        shutil.copy2(full_image_path, output_image_path)
        
        # Convert bounding boxes to YOLO format
        yolo_labels = []
        for bbox in bboxes:
            x1, y1, width, height = bbox
            
            # Convert to YOLO format
            x_center, y_center, w_norm, h_norm = convert_bbox_to_yolo(
                x1, y1, width, height, img_width, img_height
            )
            
            # YOLO format: class_id x_center y_center width height
            # Class 0 for face
            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # Save YOLO label file
        # Label file has same name as image but with .txt extension
        # Maintain directory structure to avoid name collisions
        rel_path_no_ext = os.path.splitext(rel_path)[0]
        label_output_path = os.path.join(output_labels_dir, rel_path_no_ext + '.txt')
        label_output_dir = os.path.dirname(label_output_path)
        ensure_dir(label_output_dir)
        
        with open(label_output_path, 'w') as f:
            f.write('\n'.join(yolo_labels))
        
        converted_count += 1
    
    print(f"\n✓ Conversion complete for {split}!")
    print(f"  - Converted: {converted_count} images")
    print(f"  - Skipped: {skipped_count} images")


def create_data_yaml(output_dir: str):
    """
    Create data.yaml file for YOLO training.
    
    Args:
        output_dir: Output directory containing images and labels
    """
    yaml_path = os.path.join(output_dir, 'data.yaml')
    
    # Use absolute paths to avoid path resolution issues
    images_train_dir = os.path.join(output_dir, 'images', 'train')
    images_val_dir = os.path.join(output_dir, 'images', 'val')
    
    # Convert to absolute paths
    images_train_dir = os.path.abspath(images_train_dir)
    images_val_dir = os.path.abspath(images_val_dir)
    
    # Normalize path separators for Windows
    images_train_dir = images_train_dir.replace('\\', '/')
    images_val_dir = images_val_dir.replace('\\', '/')
    
    yaml_content = f"""# YOLO Dataset Configuration for Face Detection
# WIDER FACE Dataset converted to YOLO format

train: {images_train_dir}
val: {images_val_dir}

# Number of classes
nc: 1

# Class names
names: ['face']
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created data.yaml at: {yaml_path}")
    print(f"  Train: {images_train_dir}")
    print(f"  Val: {images_val_dir}")


def main():
    """Main function to convert WIDER FACE dataset to YOLO format."""
    # Define paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    widerface_dir = os.path.join(project_root, 'data', 'widerface')
    output_dir = os.path.join(project_root, 'data')
    
    print("=" * 60)
    print("WIDER FACE to YOLO Format Converter")
    print("=" * 60)
    
    # Convert train and val splits
    for split in ['train', 'val']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")
        convert_dataset_to_yolo(widerface_dir, output_dir, split)
    
    # Create data.yaml
    print(f"\n{'='*60}")
    print("Creating data.yaml")
    print(f"{'='*60}")
    create_data_yaml(output_dir)
    
    print(f"\n{'='*60}")
    print("✓ Dataset conversion completed successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

