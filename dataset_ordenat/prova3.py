# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:33:30 2020

@author: Caterina
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:15:37 2020

@author: cmunt
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 100, 100

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 3926
nb_validation_samples = 982
epochs = 50
batch_size = 16

n_anger_tr=751
n_disgust_tr=904
n_fear_tr=313
n_happiness_tr=499
n_sadness_tr=610
n_surprise_tr=849 

n_anger_te=190
n_disgust_te=221
n_fear_te=65 
n_happiness_te=102
n_sadness_te=171 
n_surprise_te=233 



def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        'data/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size +1)
    
    #print("bottleneck_features_train ",bottleneck_features_train)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)
    
    #print("bottleneck_features_train ",bottleneck_features_train)
    

    generator = datagen.flow_from_directory(
        'data/test',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    
    print(generator.class_indices)
    
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size+1)
    
    #print("bottleneck_features_val ",bottleneck_features_validation)
     
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)
    
    #print("bottleneck_features_val ",bottleneck_features_validation)
    
save_bottlebeck_features()

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', mode="rb"))
    train_labels = np.array(
        [0] * n_anger_tr + [1] *n_disgust_tr + [2]*n_fear_tr + [3]*n_happiness_tr
        +[4]*n_sadness_tr +[5]*n_surprise_tr)
    

    print("train_labels",train_labels)
    validation_data = np.load(open('bottleneck_features_validation.npy', mode="rb"))
    validation_labels = np.array(
        [0] * n_anger_te + [1] *n_disgust_te + [2]*n_fear_te + [3]*n_happiness_te
        +[4]*n_sadness_te +[5]*n_surprise_te)
    
    
#    print("validation_labels",validation_labels)
    #print(train_data)
#    print("train data shape:" ,train_data.shape)
    model = Sequential()
    #model.add(Flatten())
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    print("vl data shape: ",validation_data.shape)
    print("vl labels shape: ",validation_labels.shape)
    
    
    print("train data shape: ",train_data.shape)
    print("train labels shape: ",train_labels.shape)
    
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)



train_top_model()