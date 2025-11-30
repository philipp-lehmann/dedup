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

def extract_features(images: list[ImageRecord], input_dir: Path):
    """Calculates pHash and BRISQUE scores for all images, using cache if available."""
    print("✨ Extracting features (pHash and BRISQUE)...")
    
    # Define a higher resolution for the hash (e.g., 16x16 = 256 bits)
    HASH_RESOLUTION = 16 
    CULLED_FOLDER = "Culled_Images" # Must match the GUI default (Hardcoded here for feature extraction)

    # Load existing features: PASSING THE CULLED_FOLDER argument IS THE FIX
    cached_features = load_features_from_report(input_dir, CULLED_FOLDER) 

    for record in tqdm(images, desc="Feature Extraction"):
        
        cache_key = record.path.name
        
        if cache_key in cached_features:
            cached_data = cached_features[cache_key]
            
            # If a record exists in cache, check if it's the right file (by checking if the file exists at the path we expect from the cache)
            if 'current_path' in cached_data and cached_data['current_path'].exists():
                # Use cached features
                record.pHash = cached_data['pHash']
                record.brisque_score = cached_data['brisque_score']
                
                # Correct the ImageRecord path
                record.path = cached_data['current_path']
                
                continue

        # 2. If not cached or path is new/changed, calculate features
        try:
            img = Image.open(record.path)
            
            # Calculate pHash with increased resolution (16x16)
            record.pHash = imagehash.phash(img, hash_size=HASH_RESOLUTION) 
            
            # Calculate Quality Value (using actual BRISQUE)
            record.brisque_score = calculate_brisque_score(record.path)

            img.close()
                        
        except Exception as e:
            record.action = 'MOVE'
            record.move_reason = f'Processing Error: {type(e).__name__}'
            record.brisque_score = 999.0 

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


def load_features_from_report(input_dir: Path, culled_folder_name: str) -> dict: # <-- Added culled_folder_name
    """
    Loads pHash and BRISQUE scores from the existing dedup_report.xml.
    Returns a dictionary mapping absolute file names to cached data,
    updating the path based on the previous action.
    """
    report_path = input_dir / 'dedup_report.xml'
    culled_dir = input_dir / culled_folder_name # Define the culled directory
    cached_features = {}
    
    if not report_path.exists():
        return cached_features

    print("📄 Found previous report. Loading cached features...")
    
    try:
        tree = ET.parse(report_path)
        root = tree.getroot()

        for img_el in root.findall('.//ImageDetails/Image'):
            path_str = img_el.find('Path').text
            phash_hex = img_el.find('pHash').text
            brisque_score_str = img_el.find('BRISQUEScore').text
            action = img_el.find('Action').text # <-- READ THE ACTION
            
            # Only cache if both values are valid and available
            if path_str and phash_hex and brisque_score_str and brisque_score_str != 'N/A':
                original_path = Path(path_str)
                current_path = original_path

                # FIX: Adjust path if the file was moved to the culled folder
                # We check the action and assume the file is now in the culled directory.
                if action == 'MOVE' or action == 'move':
                    potential_culled_path = culled_dir / original_path.name
                    if potential_culled_path.exists():
                        current_path = potential_culled_path
                
                # We also check if the file was KEPT and moved back from a previous culled dir
                elif action == 'KEEP' or action == 'keep':
                    potential_kept_path = input_dir / original_path.name
                    if potential_kept_path.exists():
                        current_path = potential_kept_path
                        
                # Key the cache by the file's name (Path.name) for simple lookup
                cached_features[current_path.name] = {
                    'pHash': imagehash.hex_to_hash(phash_hex), 
                    'brisque_score': float(brisque_score_str),
                    'current_path': current_path # <-- Store the CURRENT path
                }
        
        print(f"   Loaded {len(cached_features)} feature records.")
        return cached_features

    except ET.ParseError as e:
        print(f" - ⚠️ Error parsing XML report: {e}. Ignoring cache.")
        return {}
    except Exception as e:
        print(f" - ⚠️ An unexpected error occurred while loading cache: {e}. Ignoring cache.")
        return {}


def run_culling_process(input_dir_path: str, t_similarity: int, t_quality: float, culled_folder_name: str) -> list:
    """
    Executes the full image culling workflow.
    
    Args:
        input_dir_path (str): Path to the input folder.
        t_similarity (int): Similarity threshold (pHash Hamming distance).
        t_quality (float): Quality threshold (Max BRISQUE score).
        culled_folder_name (str): Name of the subfolder for culled images.
        
    Returns:
        list[ImageRecord]: A list of all processed ImageRecord objects.
    """
    input_dir = Path(input_dir_path)
    
    if not input_dir.is_dir():
        # Raise an exception instead of printing an error message for the GUI to catch
        raise FileNotFoundError(f"Input directory not found at {input_dir_path}")

    # --- Workflow Steps ---
    all_images = discover_images(input_dir)
    
    if not all_images:
        return []

    # Note: extract_features now requires input_dir for caching
    extract_features(all_images, input_dir) 
    
    group_and_cull_similarity(all_images, t_similarity)
    
    cull_overall_quality(all_images, t_quality)

    # File operations are executed, updating the action and location of files
    total_kept, total_moved = perform_file_operations(
        all_images, input_dir, culled_folder_name
    )
    
    # The GUI will handle reporting, but we'll include report generation 
    # here to meet the original requirement (you may remove this if performance is critical)
    # NOTE: We need a dummy object for the report function since it expects 'args'
    class DummyArgs: pass
    args = DummyArgs()
    args.T_similarity = t_similarity
    args.T_quality = t_quality
    args.culled_folder = culled_folder_name
    
    generate_report(all_images, input_dir, total_kept, total_moved, args)
    
    # Return the final list of records for the GUI to display
    return all_images
    
# --- 3. Main Execution Function ---

def main():
    """
    Original CLI entry point, modified to call the new core function.
    """
    parser = argparse.ArgumentParser(
        description="CLI: Sort and cull images based on pHash similarity and BRISQUE quality."
    )
    # ... (all argparse definitions remain the same) ...
    
    args = parser.parse_args()

    try:
        # Call the core logic function
        run_culling_process(
            args.input_dir.resolve().as_posix(),  # Use absolute path string
            args.T_similarity, 
            args.T_quality, 
            args.culled_folder
        )
        print("\n✅ Culling process finished.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")


if __name__ == '__main__':
    main()