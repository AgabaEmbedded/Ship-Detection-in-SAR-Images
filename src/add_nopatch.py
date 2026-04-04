import os
import shutil

for i in range(0, len(os.listdir(r"Data/Image/no_patch")), 10):
    image_path = os.path.join(r"Data/Image/no_patch", os.listdir(r"Data/Image/no_patch")[i])
    label_path = os.path.join(r"Data/labels/no_patch", os.listdir(r"Data/Image/no_patch")[i].replace("png", "txt"))
    shutil.copy(image_path, r"Data/Image/train")
    shutil.copy(label_path, r"Data/labels/train")
    print(f"Done {image_path}")