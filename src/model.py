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
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(300, 300, 1)))
    #allowing maxpooling of image data
    model.add(MaxPooling2D(4, 4))

    model.add(Conv2D(64, (4, 4), kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    #allowing maxpooling of image data
    #model.add(MaxPooling2D((4, 4)))

    #model.add(Conv2D(128, (3, 3), activation='relu'))
    #allowing maxpooling of image data
    #model.add(MaxPooling2D((4, 4)))

    #model.add(Conv2D(32, (4, 4), kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    #model.add(MaxPooling2D((4, 4)))
    #we need to flatten the image pixels to process them further
    model.add(Flatten())
    
    #the dense neural network model
    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(28, activation='softmax'))

    #compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    print(model.summary())
    return model