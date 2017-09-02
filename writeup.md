# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center.jpg "Center"
[image2]: ./images/left.jpg "Left"
[image3]: ./images/right.jpg "Right"
[image4]: ./images/curve.jpg "Curve"
[image5]: ./images/straight.jpg "Straight"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode. **Note this file is modified**
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video_9.mp4 The video of the car driving at 9mph without brake at turning
* video_25.mp4 The video of the car driving at 25mph with brake (to 15mph) at turning

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
**Please modify `drive.py` according to the way I stated in the code in order to see two versions of results**

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model uses the Nvidia pipeline as the starting point. ( code Line 88 - 93 for convolution layers, 95 - 98 for fully-connected layers). The code is from the video in Lesson 14.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 86). 

In order to let the model focus on important aspects of the images, like the road itself and the surroundings of the road, and reduce the distraction of unimportant aspects, like sky and clouds, I also introduced a cropping layer to select only the rows that are important. ( code line 85)

#### 2. Attempts to reduce overfitting in the model

The model contains 2 dropout layers in order to reduce overfitting (model.py lines 87 for convolution layers and 94 for fully-connected layers). 

The model was trained and validated on an augmented data set to ensure that the model was not overfitting (code line 65-70). The data is augmented by adding a y-axis (left-right) flipped counterpart for each image with inversed turning angle. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).
```python
model.compile(loss='mse', optimizer='adam')
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from a well-known and simple architecture that proves to be useful in behavioral cloning, then tune the model slightly to combat overfitting and reduce the error.

My first step was to use a convolution neural network model similar to the Nvidia pipeline. https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/. I thought this model might be appropriate because according to their paper, they achieved an autonomy value of 90%, which means human only need to explicitly guide the car 10% of the time. They also stated that an autonomous car using that model can drive 10 miles on a regular highway without intervention.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it contains two dropout layers. One is a 50% layer before all the convolution layers, the other one is a 20% layer before all the fully-connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track ( see section 3 below ) to improve the driving behavior in these cases, I augmented the dataset to make sure the model can predict correct results on hard parts of the track.  Although this has nothing to do with deriving model architecture, this is an important step for the success of the project.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 84-98) consisted of a convolution neural network with the following layers and layer sizes:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Cropping        		| Crops out top 70 and bottom 25 rows							| 
| Normalization        		| uses Lambda layer from Keras						|
| Dropout       		| 0.5						|
| Convolution 24x5x5     	| 1x1 stride, valid padding, 2x2 subsampling ( perhaps max pooling? ), RELU	|
| Convolution 36x5x5     	| 1x1 stride, valid padding, 2x2 subsampling, RELU |
| Convolution 48x5x5     	| 1x1 stride, valid padding, 2x2 subsampling, RELU	|
| Convolution 64x3x3     	| 1x1 stride, valid padding, RELU 	|
| Convolution 64x3x3     	| 1x1 stride, valid padding, RELU 	|
| FLATTEN				|			
| Dropout       		| 0.2						|
| Fully connected		| 1164 -> 100 									|	
| Fully connected		| 100 -> 50   									|		
| Fully connected		| 50 -> 10       									|
| Fully connected		| 10 -> 1       									|
		





#### 3. Creation of the Training Set & Training Process

The simulator can generate three images for each frame. One is a center image that is shoot at the center of the car, the other two are left image that shows the left side of the car and right image that shows the right side of the car.

Center:

![alt text][image1]

Left:

![alt text][image2]

Right:

![alt text][image3]


I started with the provided sample data. I used `video.py` on that data and turns that into a video, which proves to be much helpful for analyzing image series.

According to my observation, the sample contains at least 4 laps of driving in both directions, clockwisely and counter-clockwisely. I also discovered the video is sometimes shaky, so I realized that the sample data also contains enough data for recovery driving. However, I did not find anything that is focus on driving smoothly on curves.

I then put this dataset into the model stated above. The car could navigate correctly for most of the time, except at the position shown in the image below. The car will drive into the ramp every time the car passes here, and forced to stop due to terrain. 

![alt text][image4]

As you can see, this curve looks different to other curves. There is no red and white pavement on some parts of the curve, which I believe this tricks the car to think it is a straight segment instead of a curve.

In order to solve this problem, I recorded another dataset that focused just on driving through this segment of the track smoothly even in a very slow speed. 

I then put this dataset along with the provided sample dataset into the model. The car can now navigate this part of the track correctly, but a new problem arised. The car falls off this part of the track. It thinks this is a left-turning curve.

![alt text][image5]

I realized my augmented dataset generated in last might step might be a overkill. In addition, the grass on the left side of the road act like a wall, and the no grass part looks very similarly to a valid road, which tricks the car to think it is a left turn instead of going straight.  I then recorded another dataset that focused on driving straightly on the part above. 

I then put all the 3 datasets into the network, and the result is good. The car can stay on track all the time for 10 minutes at 9mph. I then changed `drive.py` to let the car drive at 25mph and slow down on curves, the result is good too, although the car is shaking on some portion of straight tracks.

```python
 # Uncomment this for stress testing of the model at 25mph, or the car will drive at 9mph
        # # if the car is turning significantly, reduce the speed
        # if abs(self.steering) > 0.2:
        #     self.set_point = 15
        #     self.turning_step = self.step
        # # if the car is about 3 seconds after a significant turn, go back to cruising speed
        # elif self.step - self.turning_step > 30:
        #     self.set_point = 25
```

After the collection process, I had 8692 data points:
- Provided Sample data : 8036
- Data of that curve: 481
- Balancing data for straight track: 175

Due to the tight memory constraint on AWS, I implemented a generator that fetches 32 data points for each batch.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
