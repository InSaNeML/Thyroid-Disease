import os, shutil
try:
    if os.path.exists("../train.zip"): os.remove("../train.zip")
    if os.path.exists("thyroid classification.zip"): os.remove("thyroid classification.zip")

    shutil.rmtree("../img_data", ignore_errors=True)
    shutil.rmtree("../data", ignore_errors=True)
except:
    print("Some error might have come up. Please check the validity of the folders.")