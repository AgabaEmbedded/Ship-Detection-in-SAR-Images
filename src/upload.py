import roboflow
rf = roboflow.Roboflow(api_key="9VCfGRMmzvi61B4c9qu5")

# get a workspace
workspace = rf.workspace("agabaembedded")

# Upload data set to a new/existing project
workspace.upload_dataset(
    r"C:\Users\Agaba_Embedded4\Desktop\Ship Detection\Additional data\second test\Data", 
    "ship-detection-sar-full",
    num_workers=10,
    project_license="MIT",
    project_type="object-detection",
    batch_name=None,
    num_retries=0
)
