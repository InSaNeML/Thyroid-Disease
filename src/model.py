import keras
from keras import models,layers

def create_model():

    """This function can be called to create new convolution models.
    The function can be modified as per need avoiding the requirement to incorporate the code in the main program.
    The parameters can be changed and the number of layers are variable as per user requirement.
    """
    
    #building a convolution network model for image data
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    #allowing maxpooling of image data
    model.add(layers.MaxPooling2D((2, 2)))


    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #allowing maxpooling of image data
    model.add(layers.MaxPooling2D((2, 2)))


    model.add(layers.Conv2D(64, (3, 3), activation='relu'))


    #we need to flatten the image pixels to process them further
    model.add(layers.Flatten())

    
    #the dense neural network model
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(13, activation='softmax')

    #compile the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model