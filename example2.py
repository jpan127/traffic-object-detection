import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

#========================#
BATCH_SIZE      = 0
NUM_CLASSES     = 0
EPOCHS          = 0
HIDDEN_UNITS    = 0
LEARNING_RATE   = 0
#========================#

# Load data
images = load_img()             # Load images
x = img_to_array(images[0])
x = x.reshape((1,) + x.shape)

# Class 1
# Class 2

# Preprocess data
x_train = 0
y_train = 0
x_test  = 0
y_test  = 0

# Randomly transform data
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
data_augmenter = ImageDataGenerator(    rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')
# Generate 20 randomly augmented versions of x
for i, batch in enumerate(data_augmenter.flow(x, batch_size=1, save_to_dir='augmented', save_prefix='class_name', save_format='jpeg')):
    if i > 20:
        break


rmsprop = RMSprop(lr=LEARNING_RATE)

# Create model
model = Sequential()
model.add(  SimpleRNN(HIDDEN_UNITS, 
            kernel_initializer=initializers.RandomNormal(stddev=0.001),
            recurrent_initializer=initializers.Identity(gain=1.0),
            activation='relu',
            input_shape=x_train.shape[1:]))
model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))
model.compile(loss='categoriacal_crossentropy', optimizer=rmspropr, metrics=['accuracy'])

# Train the model
model.fit(  x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(x_test, y_test))