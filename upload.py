import roboflow

rf = roboflow.Roboflow(api_key="9VCfGRMmzvi61B4c9qu5")

# get a workspace
workspace = rf.workspace("agabaembedded")

# Upload data set to a new/existing project
workspace.upload_dataset(
    r"./Backup/pushed data", # This is your dataset path
    "Ship-Detection-in-SAR-Full", # This will either create or get a dataset with the given ID
    num_workers=10,
    project_license="MIT",
    project_type="object-detection",
    batch_name=None,
    num_retries=0
)