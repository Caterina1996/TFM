# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:30:11 2020

@author: mmunt
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras import applications
from keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 143, 181

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'Train'
validation_data_dir = 'Val'

# nb_train_samples = 3926
# nb_validation_samples = 982

epochs = 50
batch_size = 16

n_anger_tr=178
n_disgust_tr=52
n_fear_tr=78
n_happiness_tr=184
n_neutral_tr=144
n_sadness_tr=161
n_surprise_tr=94

n_anger_te=77
n_disgust_te=23
n_fear_te=46
n_happiness_te=72
n_neutral_te=84
n_sadness_te=73
n_surprise_te=56

num_classes=7


datagen = ImageDataGenerator(rescale=1. / 255)

train_set = datagen.flow_from_directory(
    'Train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

test_set = datagen.flow_from_directory(
    'Val',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)



#new_input = Input(shape=(640, 480, 3))
# build the VGG16 network
base_model = applications.VGG16(include_top=False, weights='imagenet')

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(num_classes,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train=train_set.n//train_set.batch_size


model.fit_generator(generator=train_set,
                   steps_per_epoch=step_size_train,
                   validation_data = test_set,
                   epochs=15)

model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])


predictions=model.predict(test_set)

print('Confusion Matrix')
ypred=[round(list(i).index(max(i))) for i in predictions]
print(confusion_matrix(test_set.classes, ypred))




