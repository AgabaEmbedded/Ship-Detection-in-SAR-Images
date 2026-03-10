import os
import shutil
import random
import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from PIL import Image

# --- CONFIGURATION ---
Data_dir = r"Input_Data"
train_image_dir = r"Data/Image/train"
train_label_dir = r"Data/labels/train"
nopatch_image_dir = r"Data/Image/no_patch"
nopatch_label_dir = r"Data/labels/no_patch"

patch_size = 640
stride = 420
no_obj_keep_rate = 0.01  # Keep 1% of empty patches

# Reset dataset folders
shutil.rmtree("Data", ignore_errors=True)
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(nopatch_image_dir, exist_ok=True)
os.makedirs(nopatch_label_dir, exist_ok=True)

# --- PROCESSING ---
csv_files = [p for p in os.listdir(Data_dir) if p.endswith(".csv")]

for csv_name in csv_files:
    print(f"--- Processing: {csv_name} ---")
    image_name = os.path.splitext(csv_name)[0]
    image_path = os.path.join(Data_dir, image_name + ".tif")
    
    if not os.path.exists(image_path):
        print(f"Skipping {image_name}: TIFF file not found.")
        continue

    with rasterio.open(image_path) as src:
        # Read image metadata
        transform = src.transform
        channels = src.count  # 1 or 3
        h, w = src.height, src.width
        
        # Load bounding box annotations (Expected in Degrees/EPSG:4326)
        csv_file = pd.read_csv(os.path.join(Data_dir, csv_name))

        patch_count = 0
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                
                # 1. Read the patch window (faster than reading whole image)
                window = rasterio.windows.Window(j, i, patch_size, patch_size)
                patch_data = src.read(window=window) # Shape: (Channels, 640, 640)
                
                labels = []

                # 2. Process Annotations for this Patch
                for _, row in csv_file.iterrows():
                    # Convert Degree Coordinates to Pixel Coordinates
                    # rowcol returns (row, col) which is (y, x)
                    py_min, px_min = rowcol(transform, row["xmin"], row["ymin"])
                    py_max, px_max = rowcol(transform, row["xmax"], row["ymax"])

                    # Standardize order (top-left to bottom-right)
                    xmin, xmax = min(px_min, px_max), max(px_min, px_max)
                    ymin, ymax = min(py_min, py_max), max(py_min, py_max)

                    # Check if box overlaps with the current patch window (j=x_offset, i=y_offset)
                    if not (xmin > j + patch_size or xmax < j or ymin > i + patch_size or ymax < i):
                        
                        # Clip coordinates to patch boundaries
                        c_xmin = max(xmin, j)
                        c_xmax = min(xmax, j + patch_size)
                        c_ymin = max(ymin, i)
                        c_ymax = min(ymax, i + patch_size)

                        # Convert to Patch-Relative YOLO format (0.0 to 1.0)
                        box_w = (c_xmax - c_xmin) / patch_size
                        box_h = (c_ymax - c_ymin) / patch_size
                        center_x = ((c_xmin + c_xmax) / 2 - j) / patch_size
                        center_y = ((c_ymin + c_ymax) / 2 - i) / patch_size

                        # Filter out tiny slivers (less than 10% of width/height surviving)
                        if box_w > 0.001 and box_h > 0.001:
                            labels.append(f"0 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}\n")

                # 3. Handle Image Normalization & Saving
                # Move channels to last dim for PIL: (H, W, C)
                if channels == 1:
                    patch_display = patch_data[0] # (640, 640)
                else:
                    patch_display = np.moveaxis(patch_data, 0, -1) # (640, 640, 3)

                # Normalize to 0-255 uint8
                p_min, p_max = patch_display.min(), patch_display.max()
                if p_max > p_min:
                    patch_display = (patch_display - p_min) / (p_max - p_min) * 255
                patch_display = patch_display.astype(np.uint8)

                # Convert to PIL Image
                final_img = Image.fromarray(patch_display)
                save_name = f"patch_{image_name}_{i}_{j}"

                if labels:
                    final_img.save(os.path.join(train_image_dir, f"{save_name}.png"))
                    with open(os.path.join(train_label_dir, f"{save_name}.txt"), "w") as f:
                        f.writelines(labels)
                else:
                    # Save a small percentage of background patches
                    if random.random() < no_obj_keep_rate:
                        final_img.save(os.path.join(nopatch_image_dir, f"{save_name}.png"))
                        with open(os.path.join(nopatch_label_dir, f"{save_name}.txt"), "w") as f:
                            pass 

                patch_count += 1
        print(f"Finished {image_name}: Generated {patch_count} patches.")

print("\n--- ALL DONE ---")