import os
import shutil
import random
import rasterio
from rasterio import features
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
from rasterio.transform import rowcol
from PIL import Image

# --- LOGGING SETUP ---
# This setup creates a 'processing_log.log' file and also prints to the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing_log.log"), # Output to file
        logging.StreamHandler()                    # Output to console
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
Data_dir = r"Input_Data"
train_image_dir = r"Data/train/images"
train_label_dir = r"Data/train/labels"
nopatch_image_dir = r"Data/no_patch/images"
nopatch_label_dir = r"Data/no_patch/labels"

patch_size = 640
stride = 640
no_obj_keep_rate = 0.001

# Reset dataset folders
if os.path.exists("Data"):
    shutil.rmtree("Data", ignore_errors=True)
    logger.info("Cleared existing Data directory.")

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(nopatch_image_dir, exist_ok=True)
os.makedirs(nopatch_label_dir, exist_ok=True)

# --- PROCESSING ---
csv_files = [p for p in os.listdir(Data_dir) if p.endswith(".csv")]

GLOBAL_WATER_SHP = r".\water-polygons-split-4326\water_polygons.shp"

def get_local_mask(src, shp_path, b):
    # Added src to the signature to access src.height/width
    bbox = (b.left, b.bottom, b.right, b.top)
    
    try:
        # Load ONLY the polygons that intersect this image's bbox
        water_gdf = gpd.read_file(shp_path, bbox=bbox)
        
        if water_gdf.empty:
            logger.warning("No water bodies found for the current image extent.")
            return None
        
        # Land=1, Water=0
        land_mask = np.ones((src.height, src.width), dtype='uint8')
        
        if water_gdf.crs != src.crs:
            water_gdf = water_gdf.to_crs(src.crs)
            
        shapes = [(geom, 0) for geom in water_gdf.geometry if geom.is_valid]
        features.rasterize(shapes, out_shape=(src.height, src.width), 
                            transform=src.transform, out=land_mask)
        
        return land_mask
    except Exception as e:
        logger.error(f"Failed to create land mask: {e}")
        return None

for csv_name in csv_files:
    logger.info(f"--- Starting: {csv_name} ---")
    image_name = os.path.splitext(csv_name)[0]
    image_path = os.path.join(Data_dir, image_name + ".tif")
    
    if not os.path.exists(image_path):
        logger.warning(f"Skipping {image_name}: .tif file missing.")
        continue

    with rasterio.open(image_path) as src:
        transform = src.transform
        channels = src.count
        h, w = src.height, src.width
        bounds = src.bounds
        
        # --- 1. HANDLE LAND MASKING --- 
        logger.info(f"Attempting to generate land mask for {image_name}...")
        land_mask = get_local_mask(src, GLOBAL_WATER_SHP, bounds)

        if land_mask is not None:
            logger.info("Land mask applied successfully.")
        else:
            logger.info("No land mask available. Normalization will use full patch data.")

        # Load annotations
        csv_file = pd.read_csv(os.path.join(Data_dir, csv_name))
        present_row = csv_file.iloc[0]

        patch_count = 0
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                
                window = rasterio.windows.Window(j, i, patch_size, patch_size)
                patch_data = src.read(window=window)
                
                # Apply Mask if available (Set land areas to 0)
                if land_mask is not None:
                    local_mask = land_mask[i:i+patch_size, j:j+patch_size]
                    for c in range(channels):
                        patch_data[c][local_mask == 1] = 0

                labels = []
                for _, row in csv_file.iterrows():
                    py_min, px_min = rowcol(transform, row["xmin"], row["ymin"])
                    py_max, px_max = rowcol(transform, row["xmax"], row["ymax"])

                    xmin, xmax = min(px_min, px_max), max(px_min, px_max)
                    ymin, ymax = min(py_min, py_max), max(py_min, py_max)

                    if not (xmin > j + patch_size or xmax < j or ymin > i + patch_size or ymax < i):
                        present_row = row.copy()
                        c_xmin = max(xmin, j)
                        c_xmax = min(xmax, j + patch_size)
                        c_ymin = max(ymin, i)
                        c_ymax = min(ymax, i + patch_size)

                        box_w = (c_xmax - c_xmin) / patch_size
                        box_h = (c_ymax - c_ymin) / patch_size
                        center_x = ((c_xmin + c_xmax) / 2 - j) / patch_size
                        center_y = ((c_ymin + c_ymax) / 2 - i) / patch_size

                        if box_w > 0.001 and box_h > 0.001:
                            labels.append(f"0 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}\n")

                # --- 2. NORMALIZATION ---
                if channels == 1:
                    patch_display = patch_data[0].astype(np.float32)
                else:
                    patch_display = np.moveaxis(patch_data, 0, -1).astype(np.float32)

                # Min/Max (currently 0/100 percentile)
                p_min = np.percentile(patch_display, 0)
                p_max = np.percentile(patch_display, 100)

                if p_max > p_min:
                    patch_display = np.clip(patch_display, p_min, p_max)
                    patch_display = (patch_display - p_min) / (p_max - p_min) * 255
                
                patch_display = patch_display.astype(np.uint8)
                final_img = Image.fromarray(patch_display)

                if labels:
                    save_name = f"{image_name}_{i}_{j}_{present_row['xmin']}_{present_row['ymin']}"
                    final_img.save(os.path.join(train_image_dir, f"{save_name}.png"))
                    with open(os.path.join(train_label_dir, f"{save_name}.txt"), "w") as f:
                        f.writelines(labels)
                else:
                    save_name = f"{image_name}_{i}_{j}"
                    if random.random() < no_obj_keep_rate:
                        final_img.save(os.path.join(nopatch_image_dir, f"{save_name}.png"))
                        with open(os.path.join(nopatch_label_dir, f"{save_name}.txt"), "w") as f:
                            pass 

                patch_count += 1
        logger.info(f"Finished {image_name}: Generated {patch_count} patches.")

logger.info("--- ALL DONE ---")