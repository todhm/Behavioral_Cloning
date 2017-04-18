import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import pandas as pd
import skimage.transform as sktransform
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
import shutil
from sklearn.utils import shuffle

#Crop function to cut the unncessairy area of the image.
def crop_process(image):

    image = image[50:-20, :]
    image = cv2.resize(image,(64,64),cv2.INTER_AREA)

    return image

#flipping image to get more data
def flip_img(image,steering):
    image_flipped = np.fliplr(image)
    steering_flipped = - steering
    return image_flipped, steering_flipped



def random_shear(image,steering):
    rows,cols,ch = image.shape
    dx = np.random.randint(-50,50+1)
    #    print('dx',dx)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 10.0
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering

    return image,steering



"""
Function to change brightness to add more data
http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

"""
def random_gamma(image):

    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


#Making a shadow on existing image.
def making_shadow(image):
    h, w = image.shape[0], image.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = - k * x1
    for i in range(h):
        c = int((i - b) / k)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
    return image



#Function to make a batch of images.
def get_next_image_files(samples,batch_size):
    total_len = len(samples)
    rnd_indices = np.random.randint(0,total_len,batch_size)
    batch_imgs = []
    for index in rnd_indices:
        img = samples.iloc[index]['center_path']
        steering_angle = samples.iloc[index]['steering_angle']
        batch_imgs.append((img,steering_angle))
    return batch_imgs


#Function to make return randomly adjusted image to augment data. 
def return_processing(img_path,steering,training = True):

    img = cv2.imread(img_path)
    coin = np.random.randint(0,2)
    if bool(coin) and training:
        img,steering = flip_img(img,steering)

    if training:
        img = random_gamma(img)
    coin2 = np.random.randint(0,2)
    if bool(coin2) and training:
        img = making_shadow(img)
    img = crop_process(img)
    return img,steering



def generator(samples,training = True,batch_size = 50):
    while 1:
        if training:
            samples = shuffle(samples)
        image_data = get_next_image_files(samples,batch_size)
        images = []
        angles = []
        for image_path,steering_angle in image_data:
            img,steering = return_processing(image_path,steering_angle)
            images.append(img)
            angles.append(steering)
        X_train = np.array(images)
        y_train = np.array(angles)
        yield (X_train,y_train)
