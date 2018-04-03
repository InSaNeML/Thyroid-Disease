import pandas as pd
import numpy as np
from keras.layers import Dense, Conv2D
from model import create_model
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

train_dir = "../img_data/train"
validation_dir = "../img_data/validation"

model = create_model()

train_datagen = ImageDataGenerator(rescale = 1.0/255,
	rotation_range=45,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')

validation_datagen = ImageDataGenerator(rscale = 1.0/255)

train_generator = train_datagen.flow_from_directory(train_dir,
	target_size = (128,128),
	batch_size = 10,
	class_mode = "categorical")

validation_generator = validation_datagen.flow_from_directory(validation_dir,
	target_size = (128,128),
	batch_size = 10,
	class_mode = "categorical")

history = model.fit_generator(train_generator,
	steps_per_epoch= 200,
	epochs = 10,
	validation_data = validation_generator,
	validation_steps = 50)


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

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()