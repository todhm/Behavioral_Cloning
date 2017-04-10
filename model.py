import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Flatten
import tensorflow as tf
import keras.backend as K





def generator(session,samples,batch_size = 32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+ batch_size]
            images = []
            angles = []
            for batch_sample in zip(batch_samples['center_path'],batch_samples['steering_angle']):
                center_image = cv2.imread(batch_sample[0])
                center_angle = float(batch_sample[1])
                images.append(center_image)
                angles.append(center_angle)


            X_train = np.array(images)
            y_train = np.array(angles)
            sklearn.utils.shuffle(X_train, y_train)
            X_batch = session.run(resize_op, {img_placeholder:X_train})
            X_batch = preprocess_input(X_batch)
            y_batch = y_train.reshape(batch_size,1,1,1)
            yield (X_batch,y_batch)


ch, row, col = 3, 80, 320  # Trimmed image format
with tf.Session() as sess:
    img_placeholder = tf.placeholder("uint8", (None, 160, 320, 3),name = 'img_placeholder')
    resize_op = tf.image.resize_images(img_placeholder, (299, 299), method=0)
    train_generator = generator(sess,train_samples, batch_size=32)
    validation_generator = generator(sess,validation_samples, batch_size=32)
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    predictions = Dense(1)(x)
    print(predictions.shape)
    for layer in base_model.layers:
        layer.trainable = False
    # Preprocess incoming data, centered around zero with small standard deviation
    model = Model(input = base_model.input,output = predictions)
    model.compile(loss='mse', optimizer='adam')

    model.fit_generator(train_generator, samples_per_epoch=
    len(train_samples), validation_data=validation_generator, \
                nb_val_samples=len(validation_samples), nb_epoch=1)
