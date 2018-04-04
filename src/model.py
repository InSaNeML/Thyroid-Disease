import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras import regularizers

def create_model():

    """This function can be called to create new convolution models.
                The function can be modified as per need avoiding the requirement to incorporate the code in the main program.
                The parameters can be changed and the number of layers are variable as per user requirement."""
        
    #building a convolution network model for image data
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    #allowing maxpooling of image data
    model.add(MaxPooling2D((4, 4)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    #allowing maxpooling of image data
    model.add(MaxPooling2D((4, 4)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    #allowing maxpooling of image data
    model.add(MaxPooling2D((4, 4)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    #allowing maxpooling of image data
    model.add(MaxPooling2D((4, 4)))

    model.add(Conv2D(128, (3, 3), activation='relu'))

    #we need to flatten the image pixels to process them further
    model.add(Flatten())
    
    #the dense neural network model
    model.add(Dense(64, activation='relu'))
    model.add(Dense(14, activation='softmax'))

    #compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    print(model.summary())
    return model