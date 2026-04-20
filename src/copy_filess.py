import os
import shutil


def copy_files(name_src, src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for filename in os.listdir(name_src):
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)
        
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)

copy_files(r"Additional data/first test/dataset/train/images", r"Additional data/second test/Data/train/images", r"Additional data/second test/dataset/train/images")
copy_files(r"Additional data/first test/dataset/train/labels", r"Additional data/second test/Data/train/labels", r"Additional data/second test/dataset/train/labels")
copy_files(r"Additional data/first test/dataset/valid/images", r"Additional data/second test/Data/train/images", r"Additional data/second test/dataset/valid/images")
copy_files(r"Additional data/first test/dataset/valid/labels", r"Additional data/second test/Data/train/labels", r"Additional data/second test/dataset/valid/labels")
copy_files(r"Additional data/first test/dataset/test/images", r"Additional data/second test/Data/train/images", r"Additional data/second test/dataset/test/images")
copy_files(r"Additional data/first test/dataset/test/labels", r"Additional data/second test/Data/train/labels", r"Additional data/second test/dataset/test/labels")