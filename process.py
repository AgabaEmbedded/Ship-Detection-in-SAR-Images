import os
#import cv2
import shutil
import random
#import zipfile
import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from PIL import Image


# Define directories
Data_dir = r"Input_Data"
train_image_dir = r"Data/Image/train"
train_label_dir = r"Data/labels/train"
nopatch_image_dir = r"Data/Image/no_patch"
nopatch_label_dir = r"Data/labels/no_patch"
#zipfile_folder = r"/content/drive/MyDrive/Road Pattern"

#working_dir = r"/content/working/"

# Reset dataset folders
shutil.rmtree("Data", ignore_errors=True)
#shutil.rmtree("working", ignore_errors=True)
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(nopatch_image_dir, exist_ok=True)
os.makedirs(nopatch_label_dir, exist_ok=True)
#os.makedirs(working_dir, exist_ok=True)

# Patch parameters
patch_size = 640
stride = 420
patch_id = 0

# Process each TIFF image
#print([x for x in os.listdir() if x.endswith(".tif")])
for csv_path in [p for p in os.listdir(Data_dir) if p.endswith(".csv")]:
    print(csv_path)
    image_name = os.path.splitext(csv_path)[0]
    image_path = os.path.join(Data_dir, csv_path.replace(".csv", ".tif"))

    #print(f"copying {image_name}...")
    #shutil.copy(os.path.join(zipfile_folder, image_name+".zip"), working_dir)


    #print(f"extracting {image_name}...")
    #with zipfile.ZipFile(os.path.join(working_dir, image_name+".zip"), "r") as zip_ref:
    #    zip_ref.extractall(working_dir)

    #for file_name in os.listdir(working_dir):
    #    dir1 = os.path.join(working_dir, file_name)
    #    if os.path.isdir(dir1):
    #        print(f"fetchin {image_name} directory...")
    #        image_path = os.path.join(dir1, image_filename_dict[image_name]+".tif")
            #shutil.copy(os.path.join(image_filename_dict[image_name]+".tif", dir1), working)


    print(f"loading {image_name} image...")
    print(image_path)
    with rasterio.open(image_path) as src:
        image = src.read()
        transform = src.transform
        _, h, w = image.shape  # Image dimensions

        # Convert resolution to proper format
        x_res, y_res = src.res
        if y_res > 0:
            y_res = -y_res  # Ensure it's negative

        print(f"Processing {image_name} with resolution ({x_res}, {y_res})")

        print(f"loading {csv_path}...")
        # Load bounding box annotations
        csv_file = pd.read_csv(os.path.join(Data_dir, csv_path))

        # Iterate through patches
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = image[:, i:i+patch_size, j:j+patch_size].copy()

                #im = Image.fromarray(np.moveaxis(((patch - patch.min()) / (patch.max() - patch.min()) * 255).astype(np.uint8), 0, -1))
                labels = []

                # Convert object bounding box coordinates to pixel space
                for _, row in csv_file.iterrows():
                    # Convert world coordinates to pixel coordinates
                    ymin, xmin = rowcol(transform, row["xmin"], row["ymin"])
                    ymax, xmax = rowcol(transform, row["xmax"], row["ymax"])

                    # Ensure values are in correct order
                    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
                    ymin, ymax = min(ymin, ymax), max(ymin, ymax)

                    # Check if the object is inside the patch
                    if not (xmin > j + patch_size or xmax < j or ymin > i + patch_size or ymax < i):
                        index = row["id"]
                        # Clip bounding box to fit within the patch
                        xmin, xmax = max(xmin, j), min(xmax, j + patch_size)
                        ymin, ymax = max(ymin, i), min(ymax, i + patch_size)

                        # Convert to YOLO format
                        box_w = (xmax - xmin) / patch_size
                        box_h = (ymax - ymin) / patch_size
                        center_x = ((xmin + xmax) / 2 - j) / patch_size
                        center_y = ((ymin + ymax) / 2 - i) / patch_size

                        labels.append(f"0 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}\n")

                # Save the patch and labels
                for b in range(patch.shape[0]):
                    diff = patch[b].max() - patch[b].min()
                    #patch[b] = (patch[b] - patch[b].min()) / (patch[b].max() - patch[b].min())*255
                    if diff > 0:
                        patch[b] = (patch[b] - patch[b].min()) / diff * 255
                    else:
                        patch[b] = 0

                patch = patch.astype(np.uint8)
                patch_to_save = patch.squeeze()
                #print(patch.shape)
                #patch = np.moveaxis(patch, 0, -1)
                patch_image = Image.fromarray(patch[0])

                if labels:
                    patch_image.save(f"{train_image_dir}/patch_{image_name}_{int(index)}_{patch_id}.png")
                    with open(f"{train_label_dir}/patch_{image_name}_{int(index)}_{patch_id}.txt", "w") as file:
                        file.writelines(labels)

                    # Augmentation: Rotate Left (90° Counterclockwise)
                    #rotated_left = patch_image.rotate(90, expand=True)
                    #rotated_left_labels = [f"{i} {y} {1 - float(x)} {h} {w}\n" for i, x, y, w, h in
                    #                        [line.split() for line in labels]]

                    #rotated_left.save(f"{train_image_dir}/image_{image_name}_{patch_id}_rl.png")
                    #with open(f"{train_label_dir}/label_{image_name}_{patch_id}_rl.txt", "w") as file:
                    #    file.writelines(rotated_left_labels)

                    # Augmentation: Rotate Right (90° Clockwise)
                    #rotated_right = patch_image.rotate(-90, expand=True)
                    #rotated_right_labels = [f"{i} {1 - float(y)} {x} {h} {w}\n" for i, x, y, w, h in
                    #                       [line.split() for line in labels]]
                    #rotated_right.save(f"{train_image_dir}/image_{image_name}_{patch_id}_rr.png")
                    #with open(f"{train_label_dir}/label_{image_name}_{patch_id}_rr.txt", "w") as file:
                    #    file.writelines(rotated_right_labels)


                    # Augmentation: Flip Up-Down
                    #flipped_ud = patch_image.transpose(Image.FLIP_TOP_BOTTOM)
                    #flipped_ud_labels = [f"{i} {x} {1 - float(y)} {w} {h}\n" for i, x, y, w, h in
                    #                     [line.split() for line in labels]]
                    #flipped_ud.save(f"{train_image_dir}/image_{image_name}_{patch_id}_flip.png")
                    #with open(f"{train_label_dir}/label_{image_name}_{patch_id}_flip.txt", "w") as file:
                    #    file.writelines(flipped_ud_labels)
                else:
                    if random.random() < 0.01:  # Save only 10% of no-object patches to reduce dataset size
                        patch_image.save(f"{nopatch_image_dir}/patch_{image_name}_{patch_id}.png")
                        with open(f"{nopatch_label_dir}/patch_{image_name}_{patch_id}.txt", "w") as file:
                            pass  # Empty label file for no-object patches
                #print(f"patch {patch_id} of image {image_name}")
                patch_id += 1
                #if len(os.listdir(r"/content/Data/Image/train")) == 50:

    #shutil.rmtree(working_dir)
    del image
    #os.makedirs(working_dir, exist_ok=True)

print(f"Total patches created: {patch_id}")
