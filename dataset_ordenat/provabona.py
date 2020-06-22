# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:49:51 2020

@author: Caterina
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import keras

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 100, 100

base_model = applications.VGG16(include_top=False,input_shape=(100,100,3), weights='imagenet')

base_model.trainable = False

inputs = keras.Input(shape=(100, 100, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with 6 units bc we have 6 classes
outputs = keras.layers.Dense(6,activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


datagen = ImageDataGenerator(rescale=1. / 255)

train_set = datagen.flow_from_directory(
    'data/train',
    target_size=(img_width, img_height,3),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

test_set = datagen.flow_from_directory(
    'data/test',
    target_size=(img_width, img_height,3),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

model.fit_generator(training_set,
                         steps_per_epoch = 7500,
                         epochs = 14,
                         validation_data = test_set,
                         validation_steps = 2000)

