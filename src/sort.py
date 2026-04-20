import os
import shutil



for path in os.listdir(r"./Additional data/second test/Data/train/labels"):
    if path not in os.listdir(r"C:\Users\Agaba_Embedded4\Downloads\processed labels") and len(path.split("_"))>4:
        os.remove(os.path.join(r"Additional data/second test/Data/train/labels", path))
        os.remove(os.path.join(r"Additional data/second test/Data/train/images", path.replace(".txt", ".png")))
    