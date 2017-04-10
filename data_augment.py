'''
Augment the data
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

#NUM_NEW_IMAGES = 100000
NUM_NEW_IMAGES = 1000

def flip_img(image,measurement):
    #flipping image to get more data
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    return image_flipped, measurement_flipped




#Sheering technique inspired by ksakmann's reposiotry



def transform_image(image,steering, shear_range):
    height, width, channels = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    random_point = [ width/2 + dx, height/2]
    src_triangle = np.float32([[0, height], [width, height], [width/2, height/2]])
    dst_triangle = np.float32([[0,height],[width,height],random_point])
    warp_mat = cv2.getAffineTransform(src_triangle, dst_triangle)
    dsteering = dx/(height/2) * 360/(2*np.pi*25.0) / 6.0
    image = cv2.warpAffine(image, warp_mat, (width, height),borderMode=1)
    steering += dsteering

    return image,steering




def random_gamma(image):
    """
    way to change brightness to add more data
    http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

    """
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)



def making_shadow(image):
    h, w = image.shape[0], image.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = - k * x1
    for i in range(h):
        c = int((i - b) / k)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
    return image







    ########################################################
    # Main function
    ########################################################
def data_aug(path_X,orig_y, num_new_images):

    orig_X = []
    for x in path_X:
        orig_X.append(cv2.imread(x))
    orig_X = np.asarray(orig_X)
    orig_y = np.asarray(orig_y)

    # Create NUM_NEW_IMAGES new images, via image transform on random original image
    for i in range(NUM_NEW_IMAGES):
    	# Pick a random image from original dataset to transform
    	rand_idx = np.random.randint(orig_X.shape[0])

    	# Create new image
    	image,steering = transform_image(orig_X[rand_idx], orig_y[rand_idx],200)

    	# Add new data to augmented dataset
    	if i == 0:
            new_X = np.expand_dims(image, axis=0)
            new_y = steering
    	else:
    		new_X = np.concatenate((new_X, np.expand_dims(image, axis=0)))
    		new_y = np.append(new_y, steering)

    	if (i+1) % 1000 == 0:
    		print('%d sheered images generated' % (i+1,))
    new_X = np.concatenate((orig_X, new_X))
    new_y = np.concatenate((orig_y, new_y))

    for i in range(len(orig_X)):
    	# Pick a random image from original dataset to transform

    	# Create new image
    	image,steering = flip_img(orig_X[i], orig_y[i])

    	# Add new data to augmented dataset
    	if i == 0:
    		flip_X = np.expand_dims(image, axis=0)
    		flip_y = np.array([steering])
    	else:
    		flip_X = np.concatenate((flip_X, np.expand_dims(image, axis=0)))
    		flip_y = np.append(flip_y, steering)

    print('flipped images generated')
    new_X = np.concatenate((new_X, flip_X))
    new_y = np.concatenate((new_X, flip_y))
    # Create dict of new data, and write it to disk via pickle file
    new_data = {'features': new_X, 'labels': new_y}


    for i in range(NUM_NEW_IMAGES):
    	# Pick a random image from original dataset to transform
    	rand_idx = np.random.randint(orig_X.shape[0])

    	# Create new image
    	image = random_gamma(orig_X[rand_idx])

    	# Add new data to augmented dataset
    	if i == 0:
    		bright_img = np.expand_dims(image, axis=0)
    		bright_y = np.array([orig_y[rand_idx]])
    	else:
    		bright_img = np.concatenate((bright_img, np.expand_dims(image, axis=0)))
    		bright_y = np.append(bright_y, orig_y[rand_idx])

    	if (i+1) % 1000 == 0:
    		print('%d gamma adjusted_img generated' % (i+1,))

    new_X = np.concatenate((new_X, bright_img))
    new_y = np.concatenate((new_X, bright_y))
    with open(new_file, mode='wb') as f:
    	pickle.dump(new_data, f)

    return new_data
df = pd.read_csv('../video_collection/driving_log.csv',names =['center_path','left_path','right_path','steering_angle','throttle','break','speed'] )
df = pd.read_csv('../video_collection/driving_log.csv',names =['center_path','left_path','right_path','steering_angle','throttle','break','speed'] )
train_samples, validation_samples = train_test_split(df[['center_path','steering_angle']],test_size = 0.2)
data_aug(train_samples['center_path'],train_samples['steering_angle'],1000)
