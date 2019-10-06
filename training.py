#!/usr/bin/python3
"""
Made by:  martijnvwezel@muino.nl and rens@rens.nu


Data set structure: 
        [path_to_dataset]/train/[class_directorys]/[files]
        [path_to_dataset]/validation/[class_directorys]/[files]
        [path_to_dataset]/test/[class_directorys]/[files]
"""
import os
import pickle


from keras.utils import to_categorical
import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from network import AttentionResNet18, AttentionResNetCifar10, AttentionResNet56, AttentionResNet92

from vars import path, class_names, num_classes, img_height, img_width, training_size, validation_size, input_shape
from vars import epochs, batch_size, lr_schedule, DEBUG
from load_data import load_custom_data


# * Enable GPU POWER
K.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 8}) # ! Need a change
config.gpu_options.per_process_gpu_memory_fraction = 0.4   # ! Need a change
sess = tf.Session(config=config)
keras.backend.set_session(sess)

#*#######################################################################
# *
# * Loading custom data set, with cifar as example
# *
#*#######################################################################


# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
(x_train, y_train), (x_test, y_test) = load_custom_data(path, training_size, validation_size)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#*#######################################################################

# * check if image size can be used

model = AttentionResNet18(input_shape, n_classes=num_classes)

# model = AttentionResNetCifar10(input_shape, n_classes=num_classes)
# model = AttentionResNet56(input_shape, n_classes=num_classes)
# model = AttentionResNet92(input_shape, n_classes=num_classes)



########################################################################
# * If the earlier model/weights should be loaded into the source code with
# TODO implement that the model knows that the weight are already learned
if(0):
    print("Load model")
    model = loadTypeModel('model_weights/'+'model.json')
if(0):
    # * load weights into new model
    print("Load weights")
    model.load_weights('model_weights/'+"deep_learning_trained_model.h5")

########################################################################

# * define loss, metrics, optimizer
model.compile(keras.optimizers.Adam(lr=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])


if DEBUG :
    model.summary() # * Print model to screen

# * save model to file
model_json = model.to_json()
with open('model_weights/'+'model.json', 'w') as json_file:
    json_file.write(model_json)


train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)


# * Prepare callbacks for model saving and for learning rate adjustment.
data_dir = "logs/"
# save model if its performing better # "model") 
filepath = os.path.join(data_dir, "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)


lr_scheduler = LearningRateScheduler(lr_schedule)  # not better performance

lr_reducer = ReduceLROnPlateau(
    monitor='val_acc',
    factor=0.2,
    patience=7,
    min_lr=10e-7,
    epsilon=0.01,
    verbose=1
)

early_stopper = EarlyStopping(
    monitor='val_acc',
    min_delta=0,
    patience=15,
    verbose=1
)


# * Prepare usefull callbacks
callbacks = [checkpoint, lr_reducer, early_stopper] 



# * Learning part
model.fit_generator(
    train_datagen.flow(
        x_train, 
        y_train,
        batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_datagen.flow(
        x_test, 
        y_test, 
        batch_size=batch_size),
    validation_steps=len(x_test) // batch_size,
    initial_epoch=0
)


# * Serialize model to JSON
model_json = model.to_json()
with open('model_weights/'+'model.json', 'w') as json_file:
    json_file.write(model_json)

# * Save the learned model weights to disk
model.save_weights('model_weights/'+'deep_learning_trained_model_with_attention.h5')