import os, shutil
import pandas as pd
import numpy as np

csv_data = pd.read_csv("csv_data/train.csv", usecols=['row_id', 'view_position', 'image_name', 'detected'])

for i in range(len(csv_data)):
    image_name = csv_data.loc[i,"image_name"]
    detected = csv_data.loc[i, "detected"]
    curr_dir = os.getcwd()
    src = os.path.join(curr_dir, "data", image_name)
    dest = os.path.join(curr_dir, "img_data/train", detected, image_name)
    shutil.copy(src, dest)
    if(i%30 == 0):
        print(i," files have been copied.")