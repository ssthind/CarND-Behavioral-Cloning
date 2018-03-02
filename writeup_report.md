# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The steps followed in this project were :
* Used the udacity provided data
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"

[video1]: ./input_data/center_imgs.mp4 "Input Data from Center camera video"
[video2]: ./input_data/left_imgs.mp4 "Input Data from Left camera video"
[video3]: ./input_data/right_imgs.mp4 "Input Data from Right camera video"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 run_best_model_folder
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Model is defined using Keras library(tensorflow backend) in function "model_architecture" consists of a initial cropping the image to extract the Region of interest (ROI). Then the data is normalized in the model using a Keras lambda layer. Next convolution neural network layers with 5x5, 3x3 filter sizes and depths/'no. of filters' as 5, 24, 32, 48, 64. Followed by Flattening and fully connected layers. 
The model includes RELU layers to introduce nonlinearity after every layer and stride 2 in convolution layer for compacting the information passed to the next layer.
The ELU activation and MAX pooling was also tried, but it appeared that RELU layers with strides in convolution layer performs better.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting after mutiple convolution and Fully connected layer. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


38400/38568 [============================>.] - ETA: 0s - loss: 0.0228 - acc: 0.1814
Epoch 00000: val_acc improved from -inf to 0.17931, saving model to model_chkpt.h5
38568/38568 [==============================] - 99s - loss: 0.0229 - acc: 0.1813 - val_loss: 0.0183 - val_acc: 0.1793
Epoch 2/3
38400/38568 [============================>.] - ETA: 0s - loss: 0.0183 - acc: 0.1813
Epoch 00001: val_acc did not improve
38568/38568 [==============================] - 53s - loss: 0.0183 - acc: 0.1813 - val_loss: 0.0172 - val_acc: 0.1793
Epoch 3/3
38400/38568 [============================>.] - ETA: 0s - loss: 0.0172 - acc: 0.1813
Epoch 00002: val_acc did not improve
38568/38568 [==============================] - 53s - loss: 0.0172 - acc: 0.1813 - val_loss: 0.0165 - val_acc: 0.1793


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
Batch used was 32, with generating mutiply by factor of 6, resulting in 32 X 6 = 192 images
Dropout of 0.4 and 0.3 have been used.

#### 4. Appropriate training data

Training data used with the [udacity data](https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip)

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try various models on the existing data set.

My first step was to use simple single layer convolution neural network model. Then next model tested was Similar to [Nvidia architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I thought a similar model to this might be appropriate because it, was used for the similar task. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding drop out layers after convolutions layer 2, 3, 4 and after first dense/fully connected layer. and to reduce the number of parameter MAX Pooling layer were added to convolutions layers(with ELU activation) 2,3,4,

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle more very closer to lane in certains turns. So to improve the driving behavior in these cases, I tried experimenting the the RELU activation(instead of ELU) and with Stride(subsample) of 2 in convolution layer instead of MAX pooling. Which improved the model further making the car move more in center compared to earlier models tried.

At the end of the process, the vehicle is able to drive autonomously around the track much better on the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a 5 convolution neural network with the followed with 4 Dense layers. More details listed below

Here is a visualization of the architecture 

___________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 66, 320, 3)    0           cropping2d_input_1[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 66, 320, 3)    0           cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 3)    228         lambda_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 31, 158, 3)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 24)    1824        activation_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 14, 77, 24)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 14, 77, 24)    0           activation_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 36)     21636       dropout_1[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 5, 37, 36)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 5, 37, 36)     0           activation_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 48)     15600       dropout_2[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 3, 35, 48)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 3, 35, 48)     0           activation_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     27712       dropout_3[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 1, 33, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           activation_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 100)           0           activation_6[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_4[0][0]
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         activation_7[0][0]
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 10)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          activation_8[0][0]
===================================================
Total params: 283,871
Trainable params: 283,871
Non-trainable params: 0
____________________________________________________________________________________________________



#### 3. Creation of the Training Set & Training Process

Udacity data set was used for training, which consist 5 laps in one direction and 4 laps in opposite direction for 3 cameras: center, left, right.
Center camera data video: ![alt text][video1]

For vehicle recovering from the left side and right sides of the road back to center I used the left and right camera image with stiring correction.

For the image from left camera a stiring angle was calculate by adding correction of 0.2 to stiring angle for center image.
![alt text][video2]

For the image from right camera a stiring angle was calculate by adding correction of -0.2 to stiring angle for center image.
![alt text][video3]


To augment the data set, I also flipped images using the numpy function `np.flip(img_XXXX,1)`(lines 121 to 123) and negated the stiring angles (lines 125 to 127)

The process of fetching the data for file was done using function `process_image` , which was called from `generator` function. `generator` function was further used to feed the data to the model by randomly shuffling the for each of data sets(training and validation set) for epoch. During training and validation in function call `model.fit_generator`. Total of 8035*5 = 48210 number of data points where generated and split in 80:20 ratio for training(38568) and validation for each Epoch. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary. Further keras callbacks functions ensured the best model was saved
