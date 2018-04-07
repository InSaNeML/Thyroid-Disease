#import operating system modules to transfer files
import os, shutil
#import pandas and numpy for reading csv files
import pandas as pd
import numpy as np

csv_data = pd.read_csv("csv_data/train.csv", usecols=['row_id', 'view_position', 'image_name', 'detected'])

for i in range(int(len(csv_data)*0.5)):
    image_name = csv_data.loc[i,"image_name"]
    detected = csv_data.loc[i, "detected"]
    view_position = str(csv_data.loc[i, "view_position"])
    curr_dir = os.getcwd()
    curr_dir = os.path.normpath(curr_dir + os.sep + os.pardir)
    src = os.path.join(curr_dir, "data", image_name)
    dest = os.path.join(curr_dir, "img_data/train", view_position+detected, image_name)
    shutil.copy(src, dest)
    if(i%300 == 0):
        print(i," files have been copied to train folder.")

for i in range(int(len(csv_data)*0.5), int(len(csv_data)*0.8)):
    image_name = csv_data.loc[i,"image_name"]
    detected = csv_data.loc[i, "detected"]
    view_position = str(csv_data.loc[i, "view_position"])
    curr_dir = os.getcwd()
    curr_dir = os.path.normpath(curr_dir + os.sep + os.pardir)
    src = os.path.join(curr_dir, "data", image_name)
    dest = os.path.join(curr_dir, "img_data/validation", view_position+detected, image_name)
    shutil.copy(src, dest)
    if(i%300 == 0):
        print(i - int(len(csv_data)*0.5)," files have been copied to validation folder.")

shutil.rmtree("../data")