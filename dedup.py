import argparse
import os
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path


from PIL import Image
from tqdm import tqdm
import imagehash
import torch
import pyiqa

# Define supported image extensions
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.webp']
# --- 1. Global Configuration and PyTorch Setup ---

# Initialize the BRISQUE metric once globally
# Use GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    # Create the BRISQUE metric object (lower_better=True for BRISQUE)
    # This automatically loads the necessary model and handles feature extraction.
    BRISQUE_METRIC = pyiqa.create_metric('brisque', device=DEVICE, as_loss=False)
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load BRISQUE model. {e}")
    BRISQUE_METRIC = None


class ImageRecord:
    """A dataclass-like object to hold image features and processing status."""
    def __init__(self, path):
        self.path = Path(path)
        self.pHash = None      # imagehash.ImageHash object
        self.brisque_score = None # float (lower is better)
        self.group_id = None     # ID of the similarity group
        self.action = 'UNKNOWN'  # 'KEEP' or 'MOVE'
        self.move_reason = ''    # Reason for moving (e.g., 'Duplicate', 'Low Quality')


def calculate_brisque_score(image_path: Path) -> float:
    """
    Calculates the genuine BRISQUE score using the pyiqa PyTorch library.
    Score ranges from 0 (best quality) to 100 (worst quality).
    """
    if BRISQUE_METRIC is None:
        return 999.0 # Fail if metric initialization failed

    try:
        # Load the image using PIL/Pillow and convert it to a PyTorch tensor
        # pyiqa usually expects a tensor, but it can also handle a path or numpy array.
        
        # We'll use the path directly for simplicity, letting pyiqa handle file I/O
        # Ensure the image is a valid path string for pyiqa's I/O handler
        score_tensor = BRISQUE_METRIC(str(image_path))
        
        # Extract the score as a standard Python float
        score = score_tensor.item()
        
        return score

    except Exception as e:
        print(f" - ⚠️ BRISQUE calculation error for {image_path.name}: {e}")
        # Return a very high score to ensure the file is culled (as low quality)
        return 999.0


# --- 2. Core Workflow Functions ---

def discover_images(input_dir: Path) -> list[ImageRecord]:
    """Recursively scans the directory and creates ImageRecord objects."""
    print(f"🔍 Discovering images in: {input_dir}")
    images = []
    # Include the Culled folder in the scan (Edge Case: Existing Subfolder)
    scan_paths = [input_dir] 

    for scan_path in scan_paths:
        for file_path in scan_path.rglob('*'):
            if file_path.suffix.lower() in VALID_EXTENSIONS:
                images.append(ImageRecord(file_path))
            elif file_path.is_file() and file_path.name != 'dedup_report.xml':
                # Gracefully handle non-image files by ignoring them
                pass
    
    print(f"Found {len(images)} potential image files.")
    return images

def extract_features(images: list[ImageRecord]):
    """Calculates pHash and BRISQUE scores for all images."""
    print("✨ Extracting features (pHash and BRISQUE)...")
    for i, record in enumerate(tqdm(images, desc="Feature Extraction")):
        try:
            img = Image.open(record.path)
            # Calculate pHash (8x8 default)
            record.pHash = imagehash.phash(img)
            
            # Calculate Quality Value (using actual BRISQUE)
            record.brisque_score = calculate_brisque_score(record.path)

            # Close image file
            img.close()
            
        except Exception as e:
            # We don't print the error here to avoid cluttering the progress bar.
            # You might want to log this to a file instead.
            record.action = 'MOVE'
            record.move_reason = f'Processing Error: {type(e).__name__}'
            record.brisque_score = 999.0 # Ensure it's moved

    print("✅ Feature extraction complete.")

def group_and_cull_similarity(images: list[ImageRecord], T_similarity: int):
    """
    Groups similar images by pHash Hamming Distance and culls the duplicates,
    keeping the one with the best (lowest) BRISQUE score.
    """
    print(f"🖼️ Grouping similar images (T_similarity={T_similarity})...")
    
    # Filter out images already marked for moving due to processing errors
    pending_images = [img for img in images if img.action == 'UNKNOWN']

    # O(N^2) comparison for simplicity, optimizing for smaller N is possible
    num_pending = len(pending_images)
    group_counter = 0

    for i in range(num_pending):
        img_i = pending_images[i]
        if img_i.group_id is not None:
            continue # Already part of a group

        # Start a new group
        group_counter += 1
        current_group = [img_i]
        img_i.group_id = group_counter

        for j in range(i + 1, num_pending):
            img_j = pending_images[j]
            if img_j.group_id is not None:
                continue # Already part of a group

            # Calculate Hamming Distance
            distance = img_i.pHash - img_j.pHash
            
            if distance <= T_similarity:
                current_group.append(img_j)
                img_j.group_id = group_counter
        
        # --- Quality Culling (Group-Level) ---
        if len(current_group) > 1:
            # Find the keeper: image with the lowest BRISQUE score
            keeper = min(current_group, key=lambda x: x.brisque_score)

            for img in current_group:
                if img is keeper:
                    img.action = 'KEEP'
                else:
                    img.action = 'MOVE'
                    img.move_reason = 'Group Duplicate'
    
    # Mark ungrouped images (unique content) as KEEP initially
    for img in pending_images:
        if img.action == 'UNKNOWN':
            img.action = 'KEEP'

def cull_overall_quality(images: list[ImageRecord], T_quality: float):
    """
    Culls images that are above the absolute quality threshold (low quality).
    This runs against all images that haven't been marked for MOVE yet.
    """
    print(f"🔪 Applying overall quality filter (T_quality={T_quality})...")
    
    for record in images:
        if record.action == 'KEEP':
            if record.brisque_score > T_quality:
                record.action = 'MOVE'
                record.move_reason = 'Absolute Low Quality'

def perform_file_operations(images: list[ImageRecord], input_dir: Path, culled_folder_name: str):
    """Moves files marked for MOVE and ensures KEEP files are in the main folder."""
    culled_dir = input_dir / culled_folder_name
    culled_dir.mkdir(exist_ok=True)
    
    moved_count = 0
    kept_count = 0
    
    print(f"📦 Executing file operations. Culled folder: {culled_dir.name}")

    for record in images:
        src = record.path
        
        if record.action == 'MOVE':
            dest = culled_dir / src.name
            
            # Prevent moving if source and destination are the same (shouldn't happen with .name)
            if src.resolve() != dest.resolve():
                try:
                    # shutil.move preserves metadata like timestamps
                    shutil.move(src, dest)
                    moved_count += 1
                except Exception as e:
                    print(f" - 🛑 Failed to MOVE {src.name}: {e}")

        elif record.action == 'KEEP':
            kept_count += 1
            # Edge Case: If a keeper was in the culled directory, move it back to the input root
            if src.parent == culled_dir:
                dest = input_dir / src.name
                if src.resolve() != dest.resolve():
                    try:
                        shutil.move(src, dest)
                        # Update the record path for the report
                        record.path = dest
                    except Exception as e:
                        print(f" - 🛑 Failed to MOVE keeper back {src.name}: {e}")

    print(f"✅ Operations complete. Kept: {kept_count}, Moved: {moved_count}")
    return kept_count, moved_count

def generate_report(images: list[ImageRecord], input_dir: Path, total_kept: int, total_moved: int, args: argparse.Namespace):
    """Generates an XML report summarizing the run and image status."""
    
    root = ET.Element('CullingReport')
    
    # --- Statistics and Variables ---
    stats = ET.SubElement(root, 'Statistics')
    ET.SubElement(stats, 'TotalScanned').text = str(len(images))
    ET.SubElement(stats, 'TotalKept').text = str(total_kept)
    ET.SubElement(stats, 'TotalMoved').text = str(total_moved)

    config = ET.SubElement(root, 'Configuration')
    ET.SubElement(config, 'InputDirectory').text = str(input_dir)
    ET.SubElement(config, 'SimilarityThreshold').text = str(args.T_similarity)
    ET.SubElement(config, 'QualityThreshold').text = str(args.T_quality)
    ET.SubElement(config, 'CulledFolderName').text = args.culled_folder
    
    # --- Image Details ---
    image_details = ET.SubElement(root, 'ImageDetails')
    for record in images:
        img_el = ET.SubElement(image_details, 'Image')
        ET.SubElement(img_el, 'Path').text = str(record.path)
        ET.SubElement(img_el, 'pHash').text = str(record.pHash)
        ET.SubElement(img_el, 'BRISQUEScore').text = f"{record.brisque_score:.2f}" if record.brisque_score else "N/A"
        ET.SubElement(img_el, 'Action').text = record.action
        ET.SubElement(img_el, 'MoveReason').text = record.move_reason
        ET.SubElement(img_el, 'GroupID').text = str(record.group_id) if record.group_id else 'N/A'

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0) # Pretty print
    report_path = input_dir / 'dedup_report.xml'
    tree.write(report_path, encoding='utf-8', xml_declaration=True)
    
    print(f"📝 Report saved to: {report_path}")

# --- 3. Main Execution Function ---

def main():
    parser = argparse.ArgumentParser(
        description="Automatically sort and cull images based on pHash similarity and BRISQUE quality."
    )
    parser.add_argument(
        '--input_dir', 
        type=Path, 
        required=True, 
        help="Path to the input folder containing images."
    )
    parser.add_argument(
        '--T_similarity', 
        type=int, 
        default=4, 
        help="Max Hamming Distance for two pHash values to be considered similar (Default: 4)."
    )
    parser.add_argument(
        '--T_quality', 
        type=float, 
        default=35.0, 
        help="Max BRISQUE score. Images above this are culled (Default: 35.0)."
    )
    parser.add_argument(
        '--culled_folder', 
        type=str, 
        default='Culled_Images', 
        help="Name of the subfolder to move culled images into (Default: Culled_Images)."
    )
    
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: Input directory not found at {args.input_dir}")
        return

    # --- Workflow Steps ---
    all_images = discover_images(args.input_dir)
    
    if not all_images:
        print("No images found. Exiting.")
        return

    extract_features(all_images)
    
    group_and_cull_similarity(all_images, args.T_similarity)
    
    cull_overall_quality(all_images, args.T_quality)

    total_kept, total_moved = perform_file_operations(
        all_images, args.input_dir, args.culled_folder
    )
    
    generate_report(all_images, args.input_dir, total_kept, total_moved, args)

if __name__ == '__main__':
    main()