import sys
import threading
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QScrollArea, QPushButton, QLineEdit, QLabel, QFileDialog, QGridLayout,
    QDockWidget, QSlider, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage, QIcon, QImage, QPixmap
from PIL import Image as PILImage
from dedup import run_culling_process, ImageRecord
from pathlib import Path
from tqdm import tqdm

# --- Worker Thread for Non-Blocking Execution ---
class DedupWorker(QThread):
    # Signals to communicate results back to the GUI thread
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, input_dir, t_similarity, t_quality, culled_folder):
        super().__init__()
        self.input_dir = input_dir
        self.t_similarity = t_similarity
        self.t_quality = t_quality
        self.culled_folder = culled_folder

    def run(self):
        try:
            # Call the core logic function using the provided arguments
            final_image_records = run_culling_process(
                self.input_dir, 
                self.t_similarity, 
                self.t_quality, 
                self.culled_folder
            )
            self.finished.emit(final_image_records)
        except Exception as e:
            self.error.emit(f"Culling failed: {type(e).__name__}: {str(e)}")


# --- Main Application Window ---
class DedupApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🖼️ Image Culler Application")
        self.setGeometry(100, 100, 1200, 800)
        self.worker = None

        self._create_sidebar()
        self._create_central_widget()
        self._setup_layout()

    def _create_sidebar(self):
        # 1. Folder Selection
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select Input Directory")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.select_folder)

        # 2. Parameters
        self.similarity_slider = QSlider(Qt.Orientation.Horizontal)
        self.similarity_slider.setRange(1, 200) # Assuming 256-bit hash, 100 is a safe max for now
        self.similarity_slider.setValue(80)
        self.similarity_slider.valueChanged.connect(self._update_params_display)
        self.sim_label = QLabel(f"Similarity Threshold (Tsim): 80")

        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(35)
        self.quality_slider.valueChanged.connect(self._update_params_display)
        self.qual_label = QLabel(f"Quality Threshold (Tqual): 35.0")
        
        # 3. Action Button
        self.run_button = QPushButton("🏃 Run Culling")
        self.run_button.clicked.connect(self.start_culling)

        # Layout Sidebar
        sidebar_widget = QWidget()
        layout = QGridLayout(sidebar_widget)
        
        layout.addWidget(QLabel("Input Folder:"), 0, 0)
        layout.addWidget(self.input_path_edit, 1, 0, 1, 2)
        layout.addWidget(self.browse_button, 1, 2)
        
        layout.addWidget(QLabel("--- Culling Parameters ---"), 2, 0, 1, 3)
        layout.addWidget(self.sim_label, 3, 0, 1, 3)
        layout.addWidget(self.similarity_slider, 4, 0, 1, 3)
        layout.addWidget(self.qual_label, 5, 0, 1, 3)
        layout.addWidget(self.quality_slider, 6, 0, 1, 3)
        
        layout.addWidget(self.run_button, 7, 0, 1, 3)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        dock = QDockWidget("Controls", self)
        dock.setWidget(sidebar_widget)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    
    def _update_params_display(self):
        self.sim_label.setText(f"Similarity Threshold (Tsim): {self.similarity_slider.value()}")
        self.qual_label.setText(f"Quality Threshold (Tqual): {self.quality_slider.value()}.0")

    def _create_central_widget(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Top Half: Kept Images
        self.kept_images_container = QWidget()
        self.kept_layout = QVBoxLayout(self.kept_images_container)
        self.kept_layout.addWidget(QLabel("✅ KEEPERS (Lowest BRISQUE / High Quality Unique)"))
        self.kept_grid = QWidget() # Grid or Flow layout goes here
        self.kept_scroll = QScrollArea()
        self.kept_scroll.setWidgetResizable(True)
        self.kept_scroll.setWidget(self.kept_grid)
        self.kept_layout.addWidget(self.kept_scroll)

        # Bottom Half: Culled Images
        self.culled_images_container = QWidget()
        self.culled_layout = QVBoxLayout(self.culled_images_container)
        self.culled_layout.addWidget(QLabel("🗑️ CULLED (Duplicates / Low Quality)"))
        self.culled_grid = QWidget() # Grid or Flow layout goes here
        self.culled_scroll = QScrollArea()
        self.culled_scroll.setWidgetResizable(True)
        self.culled_scroll.setWidget(self.culled_grid)
        self.culled_layout.addWidget(self.culled_scroll)
        
        # Split central area (50/50 split)
        main_layout.addWidget(self.kept_images_container, 1)
        main_layout.addWidget(self.culled_images_container, 1)
        
        self.setCentralWidget(central_widget)

    def _setup_layout(self):
        # Placeholder grid setup (replace with actual dynamic loading)
        self.kept_grid.setLayout(QGridLayout())
        self.culled_grid.setLayout(QGridLayout())
        self.kept_grid.layout().addWidget(QLabel("Run Culling to see results..."), 0, 0)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if folder:
            self.input_path_edit.setText(folder)

    def start_culling(self):
        input_dir = self.input_path_edit.text()
        if not input_dir:
            QMessageBox.warning(self, "Input Error", "Please select an input directory.")
            return

        self.run_button.setEnabled(False)
        self.run_button.setText("Processing... ⏳")
        
        # Create and start the worker thread
        self.worker = DedupWorker(
            input_dir, 
            self.similarity_slider.value(), 
            float(self.quality_slider.value()), # Convert slider int to float
            "Culled_Images" # Default folder name
        )
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.handle_error)
        self.worker.start()

    def create_thumbnail_widget(self, image_path: Path, input_dir: Path, culled_folder_name: str): 
        """
        Creates a QWidget containing an image thumbnail and its filename, 
        with a fallback check to the alternate location (culled/input).
        """
        
        THUMB_SIZE = 128
        
        # 1. Determine the actual file path with fallback
        final_path = image_path
        
        # Check if the file exists at the path stored in the record
        if not final_path.exists():
            # If not found, check the alternate location based on the path's expected root
            
            culled_dir = input_dir / culled_folder_name
            
            # Scenario 1: File expected in root but is in culled folder
            if final_path.parent == input_dir or final_path.parent.name == '': 
                potential_path = culled_dir / final_path.name
            
            # Scenario 2: File expected in culled folder but is in root
            elif final_path.parent == culled_dir:
                potential_path = input_dir / final_path.name
                
            else: # Unhandled/nested directory structure, stick to the stored path
                potential_path = None

            if potential_path and potential_path.exists():
                final_path = potential_path
                
            elif not final_path.exists():
                # If still not found after fallback, we raise the error for the fallback widget
                raise FileNotFoundError(f"Image not found at primary or fallback locations.")


        try:
            # Use the confirmed final_path
            img = PILImage.open(final_path) # <-- Use final_path here
            img.thumbnail((THUMB_SIZE, THUMB_SIZE)) # Resize in place, maintaining aspect ratio
            
            # 2. Convert PIL Image to QImage
            # Get image bytes
            data = img.tobytes("raw", img.mode)
            
            qimage = QImage(
                data, 
                img.size[0], 
                img.size[1], 
                img.size[0] * len(img.getbands()),
                QImage.Format.Format_RGB888 if img.mode == 'RGB' else QImage.Format.Format_Grayscale8 
            )

            # 3. Create QPixmap for display
            pixmap = QPixmap.fromImage(qimage)

            # 4. Create the composite Widget
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # Image Label and Filename Label logic remains the same
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            filename_label = QLabel(final_path.name) # Use final_path.name
            filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            filename_label.setWordWrap(True)

            layout.addWidget(image_label)
            layout.addWidget(filename_label)
            layout.setContentsMargins(5, 5, 5, 5)

            return widget

        except Exception as e:
            # Fallback widget if image loading fails (including the FileNotFoundError raised above)
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.addWidget(QLabel("🚫"))
            layout.addWidget(QLabel(f"{image_path.name} (Error)"))
            return widget
    
    def display_results(self, records: list[ImageRecord]):
        # Re-enable the button
        self.run_button.setEnabled(True)
        self.run_button.setText("🏃 Run Culling")
        
        # --- GET CURRENT FOLDER PATHS ---
        current_input_dir = Path(self.input_path_edit.text())
        CULLED_FOLDER_NAME = "Culled_Images" # Should match the default/constant used in dedup.py

        # 1. Clear previous results
        self._clear_grid(self.kept_grid.layout())
        self._clear_grid(self.culled_grid.layout())

        kept_list = [r for r in records if r.action == 'KEEP']
        culled_list = [r for r in records if r.action == 'MOVE']
        
        # --- Helper to populate the grids (using QGridLayout) ---
        def populate_grid(grid_widget: QWidget, image_list: list[ImageRecord]):
            layout = grid_widget.layout()
            if not image_list:
                layout.addWidget(QLabel("No images in this category."), 0, 0)
                return

            COLUMNS = 6 
            
            for index, record in enumerate(tqdm(image_list, desc="Loading Thumbnails", leave=False)):
                row = index // COLUMNS
                col = index % COLUMNS
                
                # Create the thumbnail widget, passing directory information
                thumb_widget = self.create_thumbnail_widget(
                    record.path, 
                    current_input_dir, 
                    CULLED_FOLDER_NAME
                ) 

                # Optionally add a tooltip with BRISQUE score and pHash
                tooltip_text = (
                    f"File: {record.path.name}\n"
                    f"BRISQUE: {record.brisque_score:.2f}\n"
                    f"pHash: {str(record.pHash)}\n"
                )
                if record.action == 'MOVE':
                    tooltip_text += f"Reason: {record.move_reason}"
                    
                thumb_widget.setToolTip(tooltip_text)
                layout.addWidget(thumb_widget, row, col)


        # 2. Populate Kept Images
        self.kept_images_container.layout().itemAt(0).widget().setText(
            f"✅ KEEPERS ({len(kept_list)} images)"
        )
        populate_grid(self.kept_grid, kept_list)
        
        # 3. Populate Culled Images
        self.culled_images_container.layout().itemAt(0).widget().setText(
            f"🗑️ CULLED ({len(culled_list)} images)"
        )
        populate_grid(self.culled_grid, culled_list)
        

    def handle_error(self, message):
        self.run_button.setEnabled(True)
        self.run_button.setText("🏃 Run Culling")
        QMessageBox.critical(self, "Runtime Error", message)

    def _clear_grid(self, layout):
        # Simple cleanup function for the grids
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

if __name__ == '__main__':
    # Ensure your dedup.py module is in the same directory for import
    app = QApplication(sys.argv)
    window = DedupApp()
    window.show()
    sys.exit(app.exec())