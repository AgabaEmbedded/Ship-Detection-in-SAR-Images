import os
import shutil
import random
import rasterio
import numpy as np
import pandas as pd
import logging
from rasterio.transform import rowcol
from PIL import Image

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
Data_dir = r"Input_Data"
train_image_dir = r"Data/train/images"
train_label_dir = r"Data/train/labels"
nopatch_image_dir = r"Data/train/images" # YOLO usually keeps background images in the same folder
nopatch_label_dir = r"Data/train/labels"

patch_size = 640
stride = 420
no_obj_keep_rate = 0.005  # Increased to 5% to give the model more "Land" to study

# Reset dataset folders
if os.path.exists("Data"):
    shutil.rmtree("Data")
    logger.info("Existing Data directory cleared.")

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)

# --- PROCESSING ---
csv_files = [p for p in os.listdir(Data_dir) if p.endswith(".csv")]
total_ship_patches = 0
total_bg_patches = 0

for csv_name in csv_files:
    image_name = os.path.splitext(csv_name)[0]
    image_path = os.path.join(Data_dir, image_name + ".tif")
    
    if not os.path.exists(image_path):
        logger.warning(f"Skipping {image_name}: TIFF not found.")
        continue

    logger.info(f"--- Processing: {image_name} ---")
    
    with rasterio.open(image_path) as src:
        transform = src.transform
        channels = src.count
        logger.info(f"Image Dimensions: {src.width}x{src.height}, Channels: {channels}")
        h, w = src.height, src.width
        
        csv_file = pd.read_csv(os.path.join(Data_dir, csv_name))

        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                
                labels = []
                for _, row in csv_file.iterrows():
                    # Convert Geo-coords to Pixel-coords
                    py_min, px_min = rowcol(transform, row["xmin"], row["ymin"])
                    py_max, px_max = rowcol(transform, row["xmax"], row["ymax"])

                    # Original Box Dimensions in Pixels
                    orig_xmin, orig_xmax = min(px_min, px_max), max(px_min, px_max)
                    orig_ymin, orig_ymax = min(py_min, py_max), max(py_min, py_max)
                    
                    orig_width = orig_xmax - orig_xmin
                    orig_height = orig_ymax - orig_ymin
                    orig_area = orig_width * orig_height

                    # Check if box overlaps with the current 640x640 window (j, i)
                    if not (orig_xmin > j + patch_size or orig_xmax < j or 
                            orig_ymin > i + patch_size or orig_ymax < i):
                        present_row = row.copy()
                        # Calculate Clipped Box (the part inside the patch)
                        c_xmin = max(orig_xmin, j)
                        c_xmax = min(orig_xmax, j + patch_size)
                        c_ymin = max(orig_ymin, i)
                        c_ymax = min(orig_ymax, i + patch_size)
                        
                        clipped_area = (c_xmax - c_xmin) * (c_ymax - c_ymin)

                        # --- 25% VISIBILITY LOGIC ---
                        if clipped_area >= (0.25 * orig_area):
                            # Convert to YOLO format (relative to patch)
                            box_w = (c_xmax - c_xmin) / patch_size
                            box_h = (c_ymax - c_ymin) / patch_size
                            cx = ((c_xmin + c_xmax) / 2 - j) / patch_size
                            cy = ((c_ymin + c_ymax) / 2 - i) / patch_size
                            
                            labels.append(f"0 {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}\n")

                # Prepare the Patch Image
                window = rasterio.windows.Window(j, i, patch_size, patch_size)
                patch_data = src.read(window=window).astype(np.float32)
                
                # Normalization
                if channels == 1:
                    patch_display = patch_data[0]
                else:
                    patch_display = np.moveaxis(patch_data, 0, -1)

                p_min, p_max = np.percentile(patch_display, 0), np.percentile(patch_display, 100)
                if p_max > p_min:
                    patch_display = np.clip(patch_display, p_min, p_max)
                    patch_display = (patch_display - p_min) / (p_max - p_min) * 255
                
                final_img = Image.fromarray(patch_display.astype(np.uint8))

                # Save Decision
                if labels:
                    save_name = f"{image_name}_{i}_{j}_{str(present_row['xmin']).replace(".", "-")}_{str(present_row['ymin']).replace('.', '-')}"
                    final_img.save(os.path.join(train_image_dir, f"{save_name}.png"))
                    with open(os.path.join(train_label_dir, f"{save_name}.txt"), "w") as f:
                        f.writelines(labels)
                    total_ship_patches += 1
                elif random.random() < no_obj_keep_rate:
                    # Save as Background Patch (Empty txt file)
                    save_name = f"patch_{image_name}_{i}_{j}"
                    final_img.save(os.path.join(nopatch_image_dir, f"{save_name}.png"))
                    with open(os.path.join(nopatch_label_dir, f"{save_name}.txt"), "w") as f:
                        pass
                    total_bg_patches += 1

logger.info(f"--- Processing Complete ---")
logger.info(f"Total Ship Patches: {total_ship_patches}")
logger.info(f"Total Background Patches: {total_bg_patches}")