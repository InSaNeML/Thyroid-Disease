import os, shutil
import pandas as pd
import numpy as np

csv_data = pd.read_csv("csv_data/train.csv", usecols=['row_id', 'view_position', 'image_name', 'detected'])

for i in range(int(len(csv_data)*0.7)):
    image_name = csv_data.loc[i,"image_name"]
    detected = csv_data.loc[i, "detected"]
    curr_dir = os.getcwd()
    src = os.path.join(curr_dir, "data", image_name)
    dest = os.path.join(curr_dir, "img_data/train", detected, image_name)
    shutil.copy(src, dest)
    if(i%300 == 0):
        print(i," files have been copied to train folder.")

for i in range(int(len(csv_data)*0.7), int(len(csv_data))):
    image_name = csv_data.loc[i,"image_name"]
    detected = csv_data.loc[i, "detected"]
    curr_dir = os.getcwd()
    src = os.path.join(curr_dir, "data", image_name)
    dest = os.path.join(curr_dir, "img_data/validation", detected, image_name)
    shutil.copy(src, dest)
    if(i%300 == 0):
        print(i - int(len(csv_data)*0.7)," files have been copied to validation folder.")