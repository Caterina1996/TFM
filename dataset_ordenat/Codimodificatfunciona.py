
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math


# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/test'

# number of epochs to train top model
epochs = 50
# batch size used by flow_from_directory and predict_generator
batch_size = 16

    # build the VGG16 network
model = applications.VGG16(include_top=False, weights='imagenet')

datagen = ImageDataGenerator(rescale=1. / 255)

generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

print(len(generator.filenames))
print(generator.class_indices)
print(len(generator.class_indices))

train_labels = generator.classes

nb_train_samples = len(generator.filenames)
num_classes = len(generator.class_indices)

predict_size_train = int(math.ceil(nb_train_samples / batch_size))

bottleneck_features_train = model.predict_generator(
    generator, nb_train_samples // batch_size +1)

np.save('bottleneck_features_train.npy', bottleneck_features_train)

generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

#train_labels = generator_top.classes

validation_labels = generator.classes

nb_validation_samples = len(generator.filenames)

predict_size_validation = int(
    math.ceil(nb_validation_samples / batch_size))

predict_size_validation=nb_validation_samples // batch_size+1#...

bottleneck_features_validation = model.predict_generator(
    generator, predict_size_validation)

np.save('bottleneck_features_validation.npy',
        bottleneck_features_validation)



#datagen_top = ImageDataGenerator(rescale=1. / 255)
#generator_top = datagen_top.flow_from_directory(
#    train_data_dir,
#    target_size=(img_width, img_height),
#    batch_size=batch_size,
#    class_mode='categorical',
#    shuffle=False)
#
#nb_train_samples = len(generator_top.filenames)
#num_classes = len(generator_top.class_indices)
#
## save the class indices to use use later in predictions
#np.save('class_indices.npy', generator_top.class_indices)

# load the bottleneck features saved earlier
train_data = np.load('bottleneck_features_train.npy')

# get the class lebels for the training data, in the original order
#train_labels = generator_top.classes

# https://github.com/fchollet/keras/issues/3467
# convert the training labels to categorical vectors
train_labels = to_categorical(train_labels, num_classes=num_classes)

#generator_top = datagen_top.flow_from_directory(
#    validation_data_dir,
#    target_size=(img_width, img_height),
#    batch_size=batch_size,
#    class_mode=None,
#    shuffle=False)

#nb_validation_samples = len(generator_top.filenames)

validation_data = np.load('bottleneck_features_validation.npy')

#validation_labels = generator_top.classes
validation_labels = to_categorical(
    validation_labels, num_classes=num_classes)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(validation_data, validation_labels))

model.save_weights(top_model_weights_path)

(eval_loss, eval_accuracy) = model.evaluate(
    validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))

plt.figure(1)



save_bottlebeck_features()
train_top_model()