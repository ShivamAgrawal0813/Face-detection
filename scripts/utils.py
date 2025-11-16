"""
Utility functions for the Face Detection project.
Contains helper methods for file operations, directory creation, and common tasks.
"""

import os
from pathlib import Path
from typing import Tuple, List


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to the directory to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"✓ Directory ensured: {directory}")


def get_image_size(image_path: str) -> Tuple[int, int]:
    """
    Get image dimensions (width, height).
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (width, height)
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = img.shape[:2]
    return width, height


def convert_bbox_to_yolo(x1: float, y1: float, width: float, height: float, 
                         img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from WIDER FACE format (x1, y1, width, height) 
    to YOLO format (x_center, y_center, width, height) normalized.
    
    Args:
        x1: Top-left x coordinate
        y1: Top-left y coordinate
        width: Bounding box width
        height: Bounding box height
        img_width: Image width
        img_height: Image height
        
    Returns:
        Tuple of (x_center_norm, y_center_norm, width_norm, height_norm)
    """
    # Calculate center coordinates
    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0
    
    # Normalize to [0, 1]
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    # Clamp values to [0, 1] to ensure they're within valid range
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def find_image_path(base_dir: str, image_name: str) -> str:
    """
    Find the full path to an image file by searching in subdirectories.
    
    Args:
        base_dir: Base directory to search in
        image_name: Name of the image file (may include subdirectory path)
        
    Returns:
        Full path to the image file
    """
    # If image_name already contains path separators, try direct path first
    direct_path = os.path.join(base_dir, image_name)
    if os.path.exists(direct_path):
        return direct_path
    
    # Otherwise, search recursively
    for root, dirs, files in os.walk(base_dir):
        # Extract just the filename if path is included
        filename = os.path.basename(image_name)
        if filename in files:
            return os.path.join(root, filename)
    
    raise FileNotFoundError(f"Image not found: {image_name} in {base_dir}")


def count_files_in_directory(directory: str, extension: str = ".txt") -> int:
    """
    Count files with specific extension in a directory.
    
    Args:
        directory: Directory to search
        extension: File extension to count
        
    Returns:
        Number of files found
    """
    count = 0
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith(extension):
                count += 1
    return count

