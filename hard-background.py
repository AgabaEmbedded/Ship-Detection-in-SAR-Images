import os
import shutil
from ultralytics import YOLO

# 1. Load your current model
model = YOLO('path/to/your/best.pt')

# 2. Paths
background_folder = 'path/to/raw_background_tiles'
hard_negatives_out = 'path/to/hard_negatives'
os.makedirs(hard_negatives_out, exist_ok=True)

# 3. Run Inference
# We use a low stream buffer to handle large folders
results = model.predict(source=background_folder, conf=0.25, save=False)

for result in results:
    # If the model found ANY boxes, it's a "Hard Negative" (False Positive)
    if len(result.boxes) > 0:
        file_path = result.path
        file_name = os.path.basename(file_path)
        print(f"False Positive detected in: {file_name}. Moving to hard negatives.")
        shutil.copy(file_path, os.path.join(hard_negatives_out, file_name))

print(f"Done! Check {hard_negatives_out} for images to re-upload as Null.")