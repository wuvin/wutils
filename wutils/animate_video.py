#!/usr/bin/env python3
"""
Script to create a video from a sequence of TIFF images.

Usage:
    python animate_video.py <image_directory> <output_video_path> [options]

Example:
    python animate_video.py data/raw/image/ data/processed/image_video.mp4 --fps 30
"""

import argparse
import os
import sys
import glob
import cv2
import numpy as np
from pathlib import Path


def natural_sort_key(s):
    """
    Natural sorting key function to sort filenames with numbers correctly.
    E.g., image1.tif, image2.tif, image10.tif instead of image1.tif, image10.tif, image2.tif
    """
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def create_video_from_images(image_dir, output_path, fps=30, quality='high'):
    """
    Create a video from TIFF images in a directory.
    
    Args:
        image_dir (str): Directory containing TIFF images
        output_path (str): Output video file path
        fps (int): Frames per second for the output video
        quality (str): Video quality ('high', 'medium', 'low')
    """
    # Convert to Path objects for easier handling
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    
    # Check if input directory exists
    if not image_dir.exists():
        print(f"Error: Input directory '{image_dir}' does not exist.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all TIFF files in the directory
    tiff_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    image_files = []
    
    for pattern in tiff_patterns:
        image_files.extend(glob.glob(str(image_dir / pattern)))
    
    if not image_files:
        print(f"Error: No TIFF files found in '{image_dir}'")
        sys.exit(1)
    
    # Sort files naturally (handles numbers correctly)
    image_files.sort(key=natural_sort_key)
    
    print(f"Found {len(image_files)} TIFF images")
    print(f"First image: {os.path.basename(image_files[0])}")
    print(f"Last image: {os.path.basename(image_files[-1])}")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Error: Cannot read the first image '{image_files[0]}'")
        sys.exit(1)
    
    height, width, channels = first_image.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Set up video codec and quality settings
    fourcc_options = {
        'high': cv2.VideoWriter_fourcc(*'mp4v'),    # High quality
        'medium': cv2.VideoWriter_fourcc(*'XVID'),  # Medium quality
        'low': cv2.VideoWriter_fourcc(*'MJPG')      # Lower quality, larger file
    }
    
    fourcc = fourcc_options.get(quality, cv2.VideoWriter_fourcc(*'mp4v'))
    
    # Create VideoWriter object
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )
    
    if not video_writer.isOpened():
        print("Error: Could not open video writer")
        sys.exit(1)
    
    # Process each image
    print("Processing images...")
    for i, image_file in enumerate(image_files):
        # Read image
        img = cv2.imread(image_file)
        
        if img is None:
            print(f"Warning: Could not read image '{image_file}', skipping...")
            continue
        
        # Resize image if dimensions don't match the first image
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        # Write frame to video
        video_writer.write(img)
        
        # Show progress
        if (i + 1) % 50 == 0 or i == len(image_files) - 1:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    # Release video writer
    video_writer.release()
    
    print(f"Video saved successfully to: {output_path}")
    print(f"Video properties: {width}x{height} @ {fps} FPS")


def main():
    parser = argparse.ArgumentParser(
        description="Create a video from a sequence of TIFF images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python animate_video.py data/raw/image/ data/processed/image_video.mp4
    python animate_video.py data/raw/image/ output.mp4 --fps 24 --quality medium
        """
    )
    
    parser.add_argument(
        'image_directory',
        help='Directory containing TIFF images'
    )
    
    parser.add_argument(
        'output_video',
        help='Output video file path (e.g., output.mp4)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for output video (default: 30)'
    )
    
    parser.add_argument(
        '--quality',
        choices=['high', 'medium', 'low'],
        default='high',
        help='Video quality setting (default: high)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Input directory: {args.image_directory}")
        print(f"Output video: {args.output_video}")
        print(f"FPS: {args.fps}")
        print(f"Quality: {args.quality}")
        print("-" * 50)
    
    try:
        create_video_from_images(
            args.image_directory,
            args.output_video,
            fps=args.fps,
            quality=args.quality
        )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()