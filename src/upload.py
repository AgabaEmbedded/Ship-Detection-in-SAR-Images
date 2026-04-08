import roboflow

rf = roboflow.Roboflow(api_key="9VCfGRMmzvi61B4c9qu5")

# get a workspace
workspace = rf.workspace("agabaembedded")

# Upload data set to a new/existing project
workspace.upload_dataset(
    r"./Data", 
    "Ship-Detection-SAR-Full",
    num_workers=10,
    project_license="MIT",
    project_type="object-detection",
    batch_name=None,
    num_retries=0
)
