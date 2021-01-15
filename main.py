import numpy as np # linear algebra
import random
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import cv2
import shutil
from glob import glob
# Helper libraries
import matplotlib.pyplot as plt
import math

import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import keras as ks

path_positive_cases = os.path.join('input/CT_COVID/')
path_negative_cases = os.path.join('input/CT_NonCOVID/')

positive_images_ls = glob(os.path.join(path_positive_cases,"*.png"))

negative_images_ls = glob(os.path.join(path_negative_cases,"*.png"))
negative_images_ls.extend(glob(os.path.join(path_negative_cases,"*.jpg")))

covid = {'class': 'CT_COVID',
         'path': path_positive_cases,
         'images': positive_images_ls}

non_covid = {'class': 'CT_NonCOVID',
             'path': path_negative_cases,
             'images': negative_images_ls}

total_positive_covid = len(positive_images_ls)
total_negative_covid = len(negative_images_ls)
print("Total Positive Cases Covid19 images: {}".format(total_positive_covid))
print("Total Negative Cases Covid19 images: {}".format(total_negative_covid))

image_positive = cv2.imread(os.path.join(positive_images_ls[0]))
image_negative = cv2.imread(os.path.join(negative_images_ls[0]))

fig = plt.figure(figsize=(8, 8))
fig.add_subplot(1, 2, 1)
plt.imshow(image_negative)
fig.add_subplot(1,2, 2)
plt.imshow(image_positive)
plt.show()

print("Image POS Shape {}".format(image_positive.shape))
print("Image NEG Shape {}".format(image_negative.shape))

# Create Train-Test Directory
subdirs  = ['train/', 'test/']
for subdir in subdirs:
    labeldirs = ['CT_COVID', 'CT_NonCOVID']
    for labldir in labeldirs:
        newdir = subdir + labldir
        os.makedirs(newdir, exist_ok=True)

# Copy Images to test set

random.seed(12)
test_ratio = 0.1
########## yahan change

for cases in [covid, non_covid]:
    total_cases = len(cases['images'])  # number of total images
    num_to_select = int(test_ratio * total_cases)  # number of images to copy to test set

    print(cases['class'], num_to_select)

    list_of_random_files = random.sample(cases['images'], num_to_select)  # random files selected

    for files in list_of_random_files:
        shutil.copy2(files, 'test/' + cases['class'])

# Copy Images to train set
for cases in [covid, non_covid]:
    image_test_files = os.listdir('test/' + cases['class']) # list test files
    for images in cases['images']:
        if images.split('/')[-1] not in (image_test_files): #exclude test files from shutil.copy
            shutil.copy2(images, 'train/' + cases['class'])


total_train_covid = len(os.listdir('train/CT_COVID'))
total_train_noncovid = len(os.listdir('train/CT_NonCOVID'))
total_test_covid = len(os.listdir('test/CT_COVID'))
total_test_noncovid = len(os.listdir('test/CT_NonCOVID'))

print("Train sets images COVID: {}".format(total_train_covid))
print("Train sets images Non COVID: {}".format(total_train_noncovid))
print("Test sets images COVID: {}".format(total_test_covid))
print("Test sets images Non COVID: {}".format(total_test_noncovid))

batch_size = 32
epochs = 50
IMG_HEIGHT = 224
IMG_WIDTH = 224

train_image_generator = ImageDataGenerator(rescale=1./255,
                                           horizontal_flip = True) # Generator for our training data
test_image_generator = ImageDataGenerator(rescale=1./255,
                                          horizontal_flip = True) # Generator for our validation

# re-size all the images to this
IMAGE_SIZE = [224, 224]

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False

folders = glob('input/*/')
folders
vgg.summary()

x = Flatten()(vgg.output)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

train_dir = os.path.join('train')
test_dir = os.path.join('test')


total_train = total_train_covid + total_train_noncovid
total_test = total_test_covid + total_test_noncovid

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

# fit the model
r = model.fit_generator(
  train_data_gen,
  validation_data=test_data_gen,
  epochs=26,
  steps_per_epoch=len(train_data_gen),
  validation_steps=len(test_data_gen)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')