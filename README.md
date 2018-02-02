# **Behavioral Cloning** 

---


#### The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Collect more data to be used for the car to avoid collusion. 
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model_structure.001.png
[image2]: ./images/before_flipping.png "Before flipping"
[image3]: ./images/after_flipping.png "Model Structure"
[image4]: ./images/low_gamma.png "Image with low gamma value"
[image5]: ./images/high_gamma.png  "Image with high gamma value"
[image6]: ./images/shadowed_img.png "Shadowed image"
[image7]: ./images/cropped_img.png "Cropped image"
[image8]: ./images/video_img.jpg "video image"

---
### Files Submitted 

#### My project includes the following files:
* model.py: &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; File containing the script to create and train the model
* model_helper.py: File containg function to help generate augmented data on model.py 
* drive.py:&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;File for driving the car in autonomous mode
* model.h5:&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;Model data containing a trained convolution neural network 
* model_summary.ipynb: Jupyter notebook file to Summrize the Total process of project. 
* ml_deploy.sh:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;File to async local data to linux server data help change in local file to quickly reflect on server data.
* image_deploy.sh:     File to async local image data to linux server. 

---
### Data collection 
#### I have collected data with following strategies. 
* 1.Make 3 round labs witn normal direction. 
* 2.Make 1 round reverse direction lab.
* 3.Try to turn smoothly as possible especially on the curve track. 
* 4.Make random driving to catch the car recovering movement on the car to avoid collusion. 
    
---
### Data augmentation & Preprocessing

#### Image from and right and left camera

* Image from right and left Camera was added  by adding and subtracting 0.2 from its original steering angle

#### Flipping image. 
* Flipping image horizontally to augment data. </br>
![alt text][image2]   ![alt text][image3]

#### Adjust Brightness. 
* I produced function to return random gamma adjusted image to augment data. 
 ![alt text][image4]  ![alt text][image5]

#### Make Shadow. 
* Make random shadow on image to augment more data. </br>
![alt text][image6]

#### Cropping image.
* Cropped unnecessary part of the image. 
* The function was implemented by using cv2.resize function rather than cropping2D layer of keras since it show us more competence in training velocity of model. </br>
![alt text][image7]

#### Normalization. 
* The image was normalized by dividing 255.0 and subtract 0.5 to every pixel . ** (x/255.0) - 0.5 ** 

---
### Model Structure
#### Nvidia's end to end self driving car model. 
* Model was inspried by nvidia's self-driving car end to end model. 
* 5 Convolutional layer and 5 fully connected layer. 
* 3 5x5 filter and 2 3x3 convolutional filter was implemented. 
* Unlike my former project I tried to add fully connected layer on the model without activation function. 
* At each epoch 20000 data was used with 50 samples on each batch. 
![alt text][image1]

#### Model performance. 
* Model's performance after 5 epoches. 
    loss: 0.0169 - val_loss: 0.0168
    
---
### Final Video. 

[![alt text][image8]](https://youtu.be/9f5mRL6Jrjw)
