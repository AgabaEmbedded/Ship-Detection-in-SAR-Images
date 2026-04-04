import roboflow

rf = roboflow.Roboflow(api_key="9VCfGRMmzvi61B4c9qu5")

# get a workspace
workspace = rf.workspace("agabaembedded")

# Upload data set to a new/existing project
workspace.upload_dataset(
    r"./Data", # This is your dataset path
    "Ship-Detection-SAR-Full", # This will either create or get a dataset with the given ID
    num_workers=10,
    project_license="MIT",
    project_type="object-detection",
    batch_name=None,
    num_retries=0
)

"""
import os
import roboflow
import logging

# --- CONFIGURATION ---
DATASET_PATH = r"./Data/train/images"
TRACKER_FILE = "uploaded_tracker.txt"
API_KEY = "9VCfGRMmzvi61B4c9qu5" # Remember to rotate this key later for security!

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Initialize Roboflow
rf = roboflow.Roboflow(api_key=API_KEY)
workspace = rf.workspace("agabaembedded")
project = workspace.project("SAR-Ship-Detection")

# 2. Load the list of already uploaded files
if os.path.exists(TRACKER_FILE):
    with open(TRACKER_FILE, "r") as f:
        uploaded_files = set(line.strip() for line in f)
else:
    uploaded_files = set()

# 3. Get list of local images
all_images = [f for f in os.listdir(DATASET_PATH) if f.endswith('.png')]
new_images = [img for img in all_images if img not in uploaded_files]

logger.info(f"Total images found: {len(all_images)}")
logger.info(f"Already uploaded: {len(uploaded_files)}")
logger.info(f"New images to upload: {len(new_images)}")

# 4. Upload one by one (or in small batches) to track progress
if not new_images:
    logger.info("Everything is already up to date!")
else:
    for img_name in new_images:
        image_full_path = os.path.join(DATASET_PATH, img_name)
        
        # We use project.upload instead of upload_dataset for fine-grained control
        try:
            project.upload(
                image_path=image_full_path,
                num_retry=2
            )
            # Update the tracker file immediately after a successful upload
            with open(TRACKER_FILE, "a") as f:
                f.write(f"{img_name}\n")
            logger.info(f"Successfully uploaded: {img_name}")
            
        except Exception as e:
            logger.error(f"Failed to upload {img_name}: {e}")

logger.info("Update process finished.")"""