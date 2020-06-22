# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:31:23 2020

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

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

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



datagen = ImageDataGenerator(rescale=1. / 255)

# build the VGG16 network
model = applications.VGG16(include_top=False, weights='imagenet')

train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

bottleneck_features_train = model.predict_generator(
    train_generator, nb_train_samples // batch_size +1)

#print("bottleneck_features_train ",bottleneck_features_train)
np.save(open('bottleneck_features_train.npy', 'wb'),
        bottleneck_features_train)

#print("bottleneck_features_train ",bottleneck_features_train)


test_generator = datagen.flow_from_directory(
    'data/test',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

bottleneck_features_validation = model.predict_generator(
    test_generator, nb_validation_samples // batch_size+1)

#print("bottleneck_features_val ",bottleneck_features_validation)
 
np.save(open('bottleneck_features_validation.npy', 'wb'),
        bottleneck_features_validation)
    
    #print("bottleneck_features_val ",bottleneck_features_validation)
    
#-----------------------------------------------------------------------

b=np.array([[0,0,0,0,0,1] for i in range(n_anger_tr)]
        + [[0,0,0,0,1,0] for i in range(n_disgust_tr)]
        + [[0,0,0,1,0,0] for i in range(n_fear_tr)]
        + [[0,0,1,0,0,0] for i in range(n_happiness_tr)]
        + [[0,1,0,0,0,0] for i in range(n_sadness_tr)] 
        + [[1,0,0,0,0,0] for i in range(n_surprise_tr)])




train_data = np.load(open('bottleneck_features_train.npy', mode="rb"))

train_labels=b
print("train_labels",train_labels)
validation_data = np.load(open('bottleneck_features_validation.npy', mode="rb"))


validation_labels=np.array([[0,0,0,0,0,1] for i in range(n_anger_te)]
    + [[0,0,0,0,1,0] for i in range(n_disgust_te)]
    + [[0,0,0,1,0,0] for i in range(n_fear_te)]
    + [[0,0,1,0,0,0] for i in range(n_happiness_te)]
    + [[0,1,0,0,0,0] for i in range(n_sadness_te)] 
    + [[1,0,0,0,0,0] for i in range(n_surprise_te)])

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

history=model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels))
model.save_weights(top_model_weights_path)

model.save('prueba2.h5')



plt.figure(1)

## summarize history for accuracy
#
#plt.subplot(211)
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#
## summarize history for loss
#
#plt.subplot(212)
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

predictions=model.predict(validation_data)

print('Confusion Matrix')
ypred=[round(list(i).index(max(i))) for i in predictions]
print(confusion_matrix(test_generator.classes, ypred))

cnn=confusion_matrix(test_labels, rounded_predictions)

