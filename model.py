from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Lambda, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
import keras.callbacks
from keras import backend as K
import tensorflow as tf

import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import os
import csv

### function defining the model architecture
def model_architecture():
# Build the Final Test Neural Network in Keras Here
# Model design NV


# Build the Final Test Neural Network in Keras Here
    model = Sequential()
    # model.add(Cropping2D(cropping=((70,24),(60,60)), input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,24),(0,0)), input_shape=(160, 320, 3)))

    # Preprocess incoming data, centered around zero with small standard deviation 0.5 
    model.add(Lambda(lambda x: x /255 -0.5 ))
    #conv1
    model.add(Convolution2D(3, 5, 5, subsample=(2,2), border_mode="valid"))
    model.add(Activation('relu'))
    # model.add(ELU())

    #conv2
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode="valid"))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    #model.add(MaxPooling2D((2, 2)))

    #conv3
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode="valid"))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    #model.add(MaxPooling2D((2, 2)))

    #conv4
    model.add(Convolution2D(48, 3, 3, border_mode="valid"))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    #model.add(MaxPooling2D((2, 2)))

    #conv5
    model.add(Convolution2D(64, 3, 3,  border_mode="valid"))
    # model.add(ELU())
    model.add(Activation('relu'))
    # model.add(Dropout(0.4))
    #model.add(MaxPooling2D((2, 2)))

    #Flatten and FC
    model.add(Flatten())
    model.add(Dense(100))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Dense(10))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.summary()
    return model


    
# function to read images for filesystem
def process_image(img_path):
    source_path = img_path
    filename = source_path.split('/')[-1]
    current_path = "data/IMG/" + filename
#     print(current_path)
    img = cv2.imread(current_path)
    #flipping for color channels
    img = np.flip(img, 2)
    return img

## generator function    
def generator(dataset, batch_size=32):
    num_samples = len(dataset)
    while 1: # Loop forever so the generator never terminates
        shuffle(dataset)
        for offset in range(0, num_samples, batch_size):
            batch_samples = dataset[offset:offset+batch_size]

            images_all  = []   
            steering_angles = []
            for batch_sample in batch_samples:
                
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                # read in images from center, left and right cameras
                img_center = process_image(batch_sample[0])
                img_left = process_image(batch_sample[1])
                img_right = process_image(batch_sample[2])
                
                #print(img_center)
                ### flipping images
                img_center_flipped = np.flip(img_center,1)
                img_left_flipped =  np.flip(img_left,1)
                img_right_flipped =  np.flip(img_right,1)
                
                steering_center_flipped = -steering_center
                steering_left_flipped = -steering_left
                steering_right_flipped = -steering_right
                
                # add images and angles to data set
                images_all.extend([img_center, img_left, img_right, \
                img_center_flipped, img_left_flipped, img_right_flipped] )
                steering_angles.extend([steering_center, steering_left, steering_right, \
                steering_center_flipped, steering_left_flipped, steering_right_flipped] )

            # trim image to only see section with road
            X_train = np.array(images_all)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)
    

    
    
if __name__ == "__main__":
    
    # Data generation
    CSV_records = []
    with open('data/driving_log.csv', 'r') as csvfile:
        reader_obj = csv.reader(csvfile)
    for row in reader_obj:
        CSV_records.append(row)
    print("Read driving log completed")
    # spliting training and validation data
    training_data1, validation_data1 = train_test_split(CSV_records[1:], test_size=0.2)
    print("Data Records Split")
    # calling generator function for training and validation data generation    
    train_generator = generator(training_data1, batch_size=32)
    validation_generator = generator(validation_data1, batch_size=32)
    
    ## creating keras model
    model = model_architecture()
    print("Model architecture created")
    
    ## Setting up callbacks
    filename = "model_chkpt.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # EarlyStopping callback
    early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=1, mode='max') 
    # tensorboard callback
    # tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    # callbacks_list = [checkpoint, early_stop, tensorboard_callback]
    callbacks_list = [checkpoint, early_stop]
    
    # compile and train the model using the generator function
    # Calling Compile function on keras model for setting up loss, optimizer and metrics
    model.compile(loss ='mse', optimizer= 'adam', metrics=['accuracy'])

    #run training using generator
    history_data = model.fit_generator(train_generator, samples_per_epoch= \
                len(training_data1)*6, validation_data=validation_generator, \
                nb_val_samples=len(validation_data1)*6, nb_epoch=3 \
                       , callbacks=callbacks_list)


    #Saving the model
    model.save('model.h5')