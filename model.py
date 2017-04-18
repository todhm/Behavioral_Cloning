import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout,Convolution2D,MaxPooling2D,Flatten,Lambda
import tensorflow as tf
import keras.backend as K
from model_helper import *
from sklearn.utils import shuffle
from keras.models import load_model


samples = pd.read_csv('../videos3/driving_log.csv',skiprows = 0)
samples.columns = ['center_path','left_path','right_path','steering_angle','throttle','break','speed']
samples[['center_path','left_path','right_path']] = samples[['center_path','left_path','right_path']].applymap(lambda x : x.replace('/Users/hm/Desktop/homemaster/week13','/home/ml/'))

#Code to add left and right side of camera
left_df = samples[['left_path','steering_angle']]
left_df['steering_angle'] = left_df['steering_angle'].apply(lambda x: x + 0.2)
left_df.rename(columns={'left_path':'center_path'},inplace = True)
right_df = samples[['right_path','steering_angle']]
right_df['steering_angle'] = samples['steering_angle'].apply(lambda x: x- 0.2)
right_df.rename(columns={'right_path':'center_path'},inplace = True)
total_df = pd.concat([samples[['center_path','steering_angle']],
                      left_df,right_df])

#Split the dataset
train_samples,validation_samples = train_test_split(total_df ,test_size= 0.2)
train_samples = shuffle(train_samples)
train_samples = train_samples.reset_index()
validation_samples = validation_samples.reset_index()
print(len(validation_samples))


model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64, 64, 3)))

# starts with five convolutional and maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


adam = Adam(lr=1e-4)

model.compile(optimizer=adam, loss='mean_squared_error')
model_name = 'model.h5'
#model = load_model(model_name)
history = model.fit_generator(
    generator(train_samples),
    samples_per_epoch=20000,
    nb_epoch=5,
    validation_data=generator(validation_samples,False),
    nb_val_samples = len(validation_samples))

model.save(model_name)

print('Done',model_name)
