'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data

In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs

So that we have 1000 training examples for each class, and 400 validation examples for each class.

In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from PIL import Image
from keras.preprocessing import image
import keras.backend as K

#from imagenet_utils import decode_predictions, preprocess_input


# path to the model weights files.
weights_path = '/root/vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '/root/kaggleData/train'
validation_data_dir = '/root/kaggleData/validation'
kaggle_data_dir = '~/.keras/datasets/kaggle/validation/dogs/'
fine-tune_weights_path = 'models/fine-tune_model.h5'

nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 50

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))


# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning


# add the model on top of the convolutional base
model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)

#model.load_weights(weights_path)
model.load_weights(fine-tune_weights_path)


def read_img(img_path):
    '''This function returns a preprocessed image
    '''
    dim_ordering = K.image_dim_ordering()
    mean = (103.939, 116.779, 123.68)
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img = image.img_to_array(img, dim_ordering=dim_ordering)

    if dim_ordering == 'th':
        img[0, :, :] -= mean[0]
        img[1, :, :] -= mean[1]
        img[2, :, :] -= mean[2]
        # 'RGB'->'BGR'
        img = img[::-1, :, :]
    else:
        img[:, :, 0] -= mean[0]
        img[:, :, 1] -= mean[1]
        img[:, :, 2] -= mean[2]
        img = img[:, :, ::-1]

    img = np.expand_dims(img, axis=0)
    return img





#img_width, img_height = 150, 150
cat_img_path ="/root/.keras/datasets/kaggle/train/cats/cat.5783.jpg"
test_img1 = read_img(cat_img_path)

#print(test_img1)
classes = model.predict(test_img1)[0]
print(classes)
dog_img_path ="/root/.keras/datasets/kaggle/train/dogs/dog.5783.jpg"
test_img2 = read_img(dog_img_path)

#print(test_img2)
classes = model.predict(test_img2)[0]
print(classes)


