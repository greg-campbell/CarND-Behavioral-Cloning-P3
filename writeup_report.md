#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/model.png "Model Visualization"
[image2]: ./img/original.png "Original"
[image3]: ./img/preprocessed.png "Preprocessed (resize only)"
[image4]: ./img/preprocessed2.png "Preprocessed (resize and crop)"
[image5]: ./img/loss.png "Loss"
[image6]: ./img/steering_angle_hist2.png "Histogram of steering angles"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with a 64x2x2 convolution layer, a 2x2 max pooling layer, a 32x3x3 convolution layer, another 2x2 max pooling layer, and 5 dense layers with depths from 1024 to 1 (model.py, lines 113-127).

The model includes RELU activation on the convolution and dense layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py line 114). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 121 and 126). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 130). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 164).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The left and right cameras had their steering angles adjusted by +/- 0.28, respectively. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I originally attempted to replicate the model in the nVidia paper, as the course notes and discussion forums indicated that it was a good starting point. However, I found that it took a long time to train due to the relatively large image sizes.

[Yousef Rahal on the course discussion forum](https://carnd-forums.udacity.com/questions/26218583/answers/36052305) indicated that he had success with resizing the images to 32x16 and using only two convolution layers, and that this significantly reduced the amount of time required to train the network. I attempted to replicate this model, and had some success, but after collecting enough recovery data, I found that the car was able to handle all of the turns fine, but would veer to the right on the bridge. I was able to determine that this was happening due to the low resolution, causing it to mistake the bridge for a turn. Increasing the resolution of the images to 64x32 fixed this.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with a 75%/25% split. I used dropout layers to combat overfitting. I found that the loss and mean squared error provided a poor indication of how well the car would be able to handle the track, so I relied mostly on actually testing the model in the simulator in order to evaluate its performance.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with a 64x2x2 convolution layer, a 2x2 max pooling layer, a 32x3x3 convolution layer, another 2x2 max pooling layer, and 5 dense layers with depths from 1024 to 1 (model.py, lines 113-127).

Here is a visualization of the architecture:

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded several laps of the vehicle driving in the center of the lane. I also recorded multiple runs of the vehicle traversing the track backwards, in order to reduce bias caused by the fact that the track has more left turns than right turns.

I found that driving slowly while collecting data was more useful than letting the simulator default to the maximum speed of 30.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to stay close to the center of the track.

I did not touch track 2 because I have a finite amount of time and value my sanity.

To augment the data sat, I also flipped images and angles thinking that this would help the network generalize and handle both left and right turns.

After the collection process, I had 25490 number of data points. I then preprocessed this data by resizing each image to 64x32 and cropping it to 64x18.

![alt text][image2]
![alt text][image3]
![alt text][image4]

I finally randomly shuffled the data set and put 25% of the data into a validation set. 

While training, I used a method to stratify the training data based on steering angle, and select data more uniformly with different steering angles. This is because a large percentage of the data has a very small steering angle, and would normally cause the car to prefer driving straight. Without doing this, the car would occasionally forget to turn and just drive off the track.

The following plot shows the distribution of the steering angles in the data set:

![alt text][image6]

During training, I augmented the data with 33% probability. Augmentation involved using the left, right, and center cameras with equal probability. The left and right cameras had their steering angles adjusted by +/- 0.28, respectively. Augmentation also had a 50% chance of flipping the image horizontally and reversing the steering angle.
 
The validation set helped determine if the model was over or under fitting. As I trained my model on an EC2 instance, I used 20 epochs for the final version of the model. This was probably overkill, as the plot of the loss levels out after roughly 5 epochs.

![alt text][image5]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
