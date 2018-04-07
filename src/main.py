import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from model import create_model

print("Importing data.")
train_dir = "../img_data/train"
validation_dir = "../img_data/validation"

print("Creating Conv2D model.")
model = create_model()

print("Generating image models.This may take some time.")
train_datagen = ImageDataGenerator(rescale = 1.0/255,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.3,
	horizontal_flip=True,
	fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale = 1.0/255)

train_generator = train_datagen.flow_from_directory(train_dir,
	target_size = (128,128),
	color_mode = "grayscale",
	shuffle = "True",
	batch_size = 32,
	class_mode = "categorical")

validation_generator = validation_datagen.flow_from_directory(validation_dir,
	target_size = (128,128),
	color_mode = "grayscale",
	shuffle = "True",
	batch_size = 32,
	class_mode = "categorical")

#epochs = input("Enter number of epochs you want to train the model on:")

epochs = 1
epochs = int(epochs)
print("Fitting data to Conv2d D model.")
history = model.fit_generator(train_generator,
	steps_per_epoch= 50,
	epochs = epochs,
	validation_data = validation_generator)

#Ploting the curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.plot()
plt.savefig("accuracy plot.png")

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.plot()
plt.savefig("loss plot.png")