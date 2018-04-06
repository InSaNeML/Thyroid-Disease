import os, shutil

os.mkdir("../img_data")
os.chdir("../img_data")
print(os.getcwd())

os.mkdir("train")
os.mkdir("validation")
os.mkdir("test")

def create_folders():
	os.mkdir("class_1")
	os.mkdir("class_2")
	os.mkdir("class_3")
	os.mkdir("class_4")
	os.mkdir("class_5")
	os.mkdir("class_6")
	os.mkdir("class_7")
	os.mkdir("class_8")
	os.mkdir("class_9")
	os.mkdir("class_10")
	os.mkdir("class_11")
	os.mkdir("class_12")
	os.mkdir("class_13")
	os.mkdir("class_14")

os.chdir("train")
create_folders()
os.chdir("../validation")
create_folders()
os.chdir("../test")
create_folders()