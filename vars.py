#!/usr/bin/python3
"""
Made by:  martijnvwezel@muino.nl and rens@rens.nu

"""
from keras import backend as K
# * hyper parameters

DEBUG = True

path = 'E:/roeiboot-v3.1'
train_dir = 'train'
val_dir = 'validation'
test_dir = 'test'

training_size = 919
validation_size = 229

num_classes = 10  # * number of classes
class_names = ["Argo", "Gyas", "Laga", "Nereus", "Njord", "Orca", "Proteus", "Skadi", "Skoll", "Triton"]  # TODO make automatic

img_width = 32
img_height = 32

# *  Network part
epochs = 100
batch_size = 32
lr_schedule = 0.0001



# ! Don't change below
train_data_dir = path + '/' + train_dir
validation_data_dir = path + '/'+ val_dir
test_data_dir = path + '/' + test_dir

input_shape = (img_width, img_height, 3)

# * fix input shape
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)

    