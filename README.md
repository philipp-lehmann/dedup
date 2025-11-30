# **🖼️ Image Culler: Duplicate & Low-Quality Image Removal Tool**

The Image Culler is a desktop application built with Python and PyQt6 designed to streamline image curation by identifying and separating duplicate or near-duplicate images, and flagging low-quality images using perceptual hashing and BRISQUE scoring.

This tool is optimized for real-time responsiveness using multi-threading and features persistent caching to make subsequent culling runs much faster.

## **✨ Features**

- **Duplicate Detection (pHash):** Uses perceptual hashing to find images that are visually similar, even if they have different file names, sizes, or minor compression artifacts.
- **Quality Assessment (BRISQUE):** Implements the Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) to assign a numerical quality score, helping identify blurry, noisy, or artifact-heavy images.
- **Persistent Caching:** Stores calculated features (pHash and BRISQUE scores) in a dedup_report.xml file, significantly speeding up feature extraction on subsequent runs of the same directory.
- **Non-Blocking GUI:** Uses a worker thread (QThread) for the intensive culling process, ensuring the graphical user interface remains responsive.
- **Responsive Results Display:** Shows thumbnails of both the "KEEPERS" (unique, high-quality images) and the "CULLED" (duplicates, low-quality images) in a dynamic, responsive grid layout.
- **Configurable Thresholds:** Allows users to adjust similarity and quality tolerances to fine-tune the culling aggressiveness.

## **⚙️ Requirements**

This project requires Python and several specific libraries, including PyQt6 for the GUI, and specialized libraries for image processing and quality analysis.

### **Python Dependencies**

You can install all necessary dependencies using pip:

pip install PyQt6 Pillow imagehash scikit-image numpy tqdm

**Note:** The BRISQUE calculation logic relies on NumPy and scikit-image, which must be installed correctly.

## **🚀 Usage**

### **1. File Structure**

Ensure you have two primary Python files in your project directory:

1. dedup_gui.py: Contains the PyQt6 application logic and the worker thread.
2. dedup.py: Contains the core image processing and culling functions (run_culling_process, extract_features, etc.).

### **2. Running the Application**

Execute the GUI script directly from your terminal:

python dedup_gui.py

### **3. Workflow**

1. **Select Folder:** Click the "Browse" button and choose the root directory containing the images you wish to cull.
2. **Adjust Parameters:**
- **Similarity Threshold (Tsim):** Controls how similar two images must be to be considered a duplicate. Higher values mean images need to be more different to be kept. (e.g., 80 is a good starting point for 256-bit hashes).
- **Quality Threshold (Tqual):** Images with a BRISQUE score *above* this value are flagged as low quality. BRISQUE scores range from 0 (best quality) to 100 (worst quality).
1. **Run Culling:** Click the "🏃 Run Culling" button. The process will run in the background, and the UI will update upon completion.
2. **Review Results:**
- The **KEEPERS** panel shows images deemed unique and of sufficient quality.
- The **CULLED** panel shows duplicates (the lower-quality copy of a pair) and images that failed the quality check.
1. **Data Persistence:** A dedup_report.xml file is created in your input folder. When you run the application again on the same folder, features for known images will be loaded instantly, only recalculating for new images.
2. **Output:** Culled images are moved to a subfolder named Culled_Images within your input directory.

## **💡 Technical Notes**

### **Caching and Re-runs**

To ensure correct thumbnail display after images have been moved, the culling script implements robust path reconstruction:

1. When loading the dedup_report.xml, the script checks the Action (KEEP or MOVE) from the previous run.
2. If an image was MOVEd, the script checks the Culled_Images subfolder for the file's current location before loading its cached features.
3. The GUI also includes a fallback mechanism in create_thumbnail_widget to check the alternate folder if the file is not found at the path stored in the ImageRecord.

### **Customizing BRISQUE**

The BRISQUE algorithm is computationally heavy. If you encounter performance issues, ensure your system has a fast NumPy installation. The default BRISQUE implementation is used to generate the quality score