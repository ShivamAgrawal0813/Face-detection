"""
Dataset Validation Script

This script validates the WIDER FACE to YOLO conversion process.
It tests label parsing, conversion, and verifies the dataset is ready for training.
"""

import os
import sys
import cv2
from pathlib import Path
from collections import defaultdict

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import ensure_dir, convert_bbox_to_yolo, find_image_path
from scripts.convert_to_yolo import parse_widerface_label_file


def validate_yolo_label(label_path: str, image_path: str = None) -> tuple:
    """
    Validate a YOLO format label file.
    
    Args:
        label_path: Path to YOLO label file
        image_path: Optional path to corresponding image (for dimension checking)
        
    Returns:
        Tuple of (is_valid, errors, warnings, stats)
    """
    errors = []
    warnings = []
    stats = {
        'total_boxes': 0,
        'valid_boxes': 0,
        'invalid_boxes': 0,
        'out_of_bounds': 0
    }
    
    if not os.path.exists(label_path):
        return False, [f"Label file not found: {label_path}"], [], stats
    
    # Get image dimensions if image path provided
    img_width, img_height = None, None
    if image_path and os.path.exists(image_path):
        try:
            img = cv2.imread(image_path)
            if img is not None:
                img_height, img_width = img.shape[:2]
        except Exception as e:
            warnings.append(f"Could not read image for validation: {e}")
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"Line {line_num}: Invalid format. Expected 5 values, got {len(parts)}")
                stats['invalid_boxes'] += 1
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                stats['total_boxes'] += 1
                
                # Validate class ID
                if class_id != 0:
                    warnings.append(f"Line {line_num}: Class ID is {class_id}, expected 0 for face")
                
                # Validate normalized coordinates (should be 0-1)
                if not (0.0 <= x_center <= 1.0):
                    errors.append(f"Line {line_num}: x_center {x_center} out of range [0, 1]")
                    stats['out_of_bounds'] += 1
                if not (0.0 <= y_center <= 1.0):
                    errors.append(f"Line {line_num}: y_center {y_center} out of range [0, 1]")
                    stats['out_of_bounds'] += 1
                if not (0.0 <= width <= 1.0):
                    errors.append(f"Line {line_num}: width {width} out of range [0, 1]")
                    stats['out_of_bounds'] += 1
                if not (0.0 <= height <= 1.0):
                    errors.append(f"Line {line_num}: height {height} out of range [0, 1]")
                    stats['out_of_bounds'] += 1
                
                # Check if box is too small (might be noise)
                if width < 0.01 or height < 0.01:
                    warnings.append(f"Line {line_num}: Very small bounding box (w={width:.4f}, h={height:.4f})")
                
                # Check if box extends beyond image bounds
                x_min = x_center - width / 2.0
                x_max = x_center + width / 2.0
                y_min = y_center - height / 2.0
                y_max = y_center + height / 2.0
                
                if x_min < 0 or x_max > 1.0 or y_min < 0 or y_max > 1.0:
                    warnings.append(f"Line {line_num}: Box extends beyond image bounds")
                
                # If we have image dimensions, validate against actual image
                if img_width and img_height:
                    # Convert back to pixel coordinates
                    x_center_px = x_center * img_width
                    y_center_px = y_center * img_height
                    width_px = width * img_width
                    height_px = height * img_height
                    
                    x1_px = x_center_px - width_px / 2.0
                    y1_px = y_center_px - height_px / 2.0
                    x2_px = x_center_px + width_px / 2.0
                    y2_px = y_center_px + height_px / 2.0
                    
                    if x1_px < 0 or y1_px < 0 or x2_px > img_width or y2_px > img_height:
                        warnings.append(f"Line {line_num}: Box extends beyond image dimensions")
                
                if not errors:
                    stats['valid_boxes'] += 1
                    
            except ValueError as e:
                errors.append(f"Line {line_num}: Could not parse values - {e}")
                stats['invalid_boxes'] += 1
    
    except Exception as e:
        return False, [f"Error reading label file: {e}"], [], stats
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings, stats


def test_conversion_sample(label_file_path: str, images_dir: str, num_samples: int = 10):
    """
    Test conversion on a sample of images from the label file.
    
    Args:
        label_file_path: Path to WIDER FACE label.txt
        images_dir: Directory containing images
        num_samples: Number of samples to test
    """
    print("=" * 60)
    print("Testing Label Conversion (Sample)")
    print("=" * 60)
    
    # Parse label file
    print(f"\nParsing label file: {label_file_path}")
    annotations = parse_widerface_label_file(label_file_path)
    
    if not annotations:
        print("❌ ERROR: No annotations found in label file!")
        return False
    
    print(f"✓ Found {len(annotations)} images with annotations")
    
    # Test first N samples
    test_samples = list(annotations.items())[:num_samples]
    print(f"\nTesting conversion on {len(test_samples)} samples...")
    
    conversion_errors = []
    conversion_warnings = []
    successful_conversions = 0
    
    for idx, (image_path, bboxes) in enumerate(test_samples, 1):
        print(f"\n  Sample {idx}/{len(test_samples)}: {image_path}")
        print(f"    - Bounding boxes: {len(bboxes)}")
        
        if not bboxes:
            conversion_warnings.append(f"{image_path}: No bounding boxes")
            continue
        
        # Find image file
        try:
            full_image_path = find_image_path(images_dir, image_path)
        except FileNotFoundError:
            conversion_errors.append(f"{image_path}: Image file not found")
            continue
        
        # Read image
        try:
            img = cv2.imread(full_image_path)
            if img is None:
                conversion_errors.append(f"{image_path}: Could not read image")
                continue
            img_height, img_width = img.shape[:2]
            print(f"    - Image size: {img_width}x{img_height}")
        except Exception as e:
            conversion_errors.append(f"{image_path}: Error reading image - {e}")
            continue
        
        # Convert bounding boxes
        yolo_labels = []
        for bbox_idx, bbox in enumerate(bboxes):
            x1, y1, width, height = bbox
            
            # Validate original bbox
            if width <= 0 or height <= 0:
                conversion_warnings.append(f"{image_path} bbox {bbox_idx}: Invalid dimensions")
                continue
            
            # Check if bbox is within image bounds
            if x1 < 0 or y1 < 0 or (x1 + width) > img_width or (y1 + height) > img_height:
                conversion_warnings.append(f"{image_path} bbox {bbox_idx}: Extends beyond image")
            
            # Convert to YOLO format
            try:
                x_center, y_center, w_norm, h_norm = convert_bbox_to_yolo(
                    x1, y1, width, height, img_width, img_height
                )
                
                # Validate converted coordinates
                if not (0.0 <= x_center <= 1.0) or not (0.0 <= y_center <= 1.0) or \
                   not (0.0 <= w_norm <= 1.0) or not (0.0 <= h_norm <= 1.0):
                    conversion_errors.append(
                        f"{image_path} bbox {bbox_idx}: Invalid normalized coordinates"
                    )
                    continue
                
                yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                print(f"    - Bbox {bbox_idx}: ({x1:.1f}, {y1:.1f}, {width:.1f}, {height:.1f}) -> "
                      f"({x_center:.4f}, {y_center:.4f}, {w_norm:.4f}, {h_norm:.4f})")
                
            except Exception as e:
                conversion_errors.append(f"{image_path} bbox {bbox_idx}: Conversion error - {e}")
        
        if yolo_labels:
            successful_conversions += 1
            print(f"    ✓ Successfully converted {len(yolo_labels)} bounding boxes")
        else:
            conversion_warnings.append(f"{image_path}: No valid bounding boxes after conversion")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Conversion Test Summary")
    print(f"{'='*60}")
    print(f"  Total samples tested: {len(test_samples)}")
    print(f"  Successful conversions: {successful_conversions}")
    print(f"  Errors: {len(conversion_errors)}")
    print(f"  Warnings: {len(conversion_warnings)}")
    
    if conversion_errors:
        print(f"\n❌ ERRORS:")
        for error in conversion_errors[:10]:  # Show first 10 errors
            print(f"    - {error}")
        if len(conversion_errors) > 10:
            print(f"    ... and {len(conversion_errors) - 10} more errors")
    
    if conversion_warnings:
        print(f"\n⚠ WARNINGS:")
        for warning in conversion_warnings[:10]:  # Show first 10 warnings
            print(f"    - {warning}")
        if len(conversion_warnings) > 10:
            print(f"    ... and {len(conversion_warnings) - 10} more warnings")
    
    return len(conversion_errors) == 0


def validate_yolo_dataset(data_yaml: str, split: str = 'train', num_samples: int = 20):
    """
    Validate converted YOLO dataset using Ultralytics validation.
    
    Args:
        data_yaml: Path to data.yaml
        split: 'train' or 'val'
        num_samples: Number of samples to validate
    """
    print("\n" + "=" * 60)
    print("Validating YOLO Dataset Format")
    print("=" * 60)
    
    if not os.path.exists(data_yaml):
        print(f"❌ ERROR: data.yaml not found: {data_yaml}")
        print("   Please run conversion first: python main.py --convert")
        return False
    
    # Read data.yaml to get paths
    import yaml
    try:
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ ERROR: Could not read data.yaml: {e}")
        return False
    
    images_dir = data_config.get('train' if split == 'train' else 'val', '')
    labels_dir = images_dir.replace('images', 'labels')
    
    if not os.path.exists(images_dir):
        print(f"❌ ERROR: Images directory not found: {images_dir}")
        return False
    
    if not os.path.exists(labels_dir):
        print(f"❌ ERROR: Labels directory not found: {labels_dir}")
        return False
    
    print(f"✓ Images directory: {images_dir}")
    print(f"✓ Labels directory: {labels_dir}")
    
    # Find image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(images_dir).rglob(f'*{ext}'))
        image_files.extend(Path(images_dir).rglob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ ERROR: No image files found in {images_dir}")
        return False
    
    print(f"✓ Found {len(image_files)} image files")
    
    # Validate sample of label files
    print(f"\nValidating {min(num_samples, len(image_files))} label files...")
    
    validation_stats = {
        'total': 0,
        'valid': 0,
        'invalid': 0,
        'missing_labels': 0,
        'total_boxes': 0,
        'valid_boxes': 0
    }
    
    sample_files = image_files[:num_samples]
    
    for image_file in sample_files:
        validation_stats['total'] += 1
        
        # Find corresponding label file
        rel_path = os.path.relpath(image_file, images_dir)
        label_file = os.path.join(labels_dir, os.path.splitext(rel_path)[0] + '.txt')
        
        if not os.path.exists(label_file):
            validation_stats['missing_labels'] += 1
            print(f"  ⚠ Missing label: {label_file}")
            continue
        
        # Validate label file
        is_valid, errors, warnings, stats = validate_yolo_label(str(label_file), str(image_file))
        
        validation_stats['total_boxes'] += stats['total_boxes']
        validation_stats['valid_boxes'] += stats['valid_boxes']
        
        if is_valid:
            validation_stats['valid'] += 1
        else:
            validation_stats['invalid'] += 1
            if errors:
                print(f"  ❌ {os.path.basename(label_file)}: {errors[0]}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")
    print(f"  Total files checked: {validation_stats['total']}")
    print(f"  Valid label files: {validation_stats['valid']}")
    print(f"  Invalid label files: {validation_stats['invalid']}")
    print(f"  Missing label files: {validation_stats['missing_labels']}")
    print(f"  Total bounding boxes: {validation_stats['total_boxes']}")
    print(f"  Valid bounding boxes: {validation_stats['valid_boxes']}")
    
    if validation_stats['invalid'] == 0 and validation_stats['missing_labels'] == 0:
        print(f"\n✓ Dataset validation PASSED!")
        return True
    else:
        print(f"\n❌ Dataset validation FAILED!")
        return False


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate WIDER FACE to YOLO conversion')
    parser.add_argument('--test-conversion', action='store_true', 
                       help='Test label conversion on samples')
    parser.add_argument('--validate-dataset', action='store_true',
                       help='Validate converted YOLO dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                       help='Dataset split to validate')
    parser.add_argument('--samples', type=int, default=20,
                       help='Number of samples to test')
    
    args = parser.parse_args()
    
    # If no specific action, run all validations
    if not args.test_conversion and not args.validate_dataset:
        args.test_conversion = True
        args.validate_dataset = True
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    widerface_dir = os.path.join(project_root, 'data', 'widerface')
    data_yaml = os.path.join(project_root, 'data', 'data.yaml')
    
    all_passed = True
    
    # Test conversion
    if args.test_conversion:
        label_file = os.path.join(widerface_dir, args.split, 'label.txt')
        images_dir = os.path.join(widerface_dir, args.split, 'images')
        
        if not os.path.exists(label_file):
            print(f"❌ ERROR: Label file not found: {label_file}")
            all_passed = False
        else:
            result = test_conversion_sample(label_file, images_dir, args.samples)
            all_passed = all_passed and result
    
    # Validate converted dataset
    if args.validate_dataset:
        if os.path.exists(data_yaml):
            result = validate_yolo_dataset(data_yaml, args.split, args.samples)
            all_passed = all_passed and result
        else:
            print(f"\n⚠ Skipping dataset validation: data.yaml not found")
            print(f"   Run conversion first: python main.py --convert")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED - Dataset is ready for training!")
    else:
        print("❌ VALIDATION FAILED - Please fix errors before training")
    print(f"{'='*60}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

