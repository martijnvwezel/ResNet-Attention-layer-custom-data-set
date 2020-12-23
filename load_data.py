"""
Made by:  martijnvwezel@muino.nl and rens@rens.nu

"""
from pathlib import Path
import numpy as np
from PIL import Image
from keras.datasets import cifar100
from tensorflow.python.keras import backend as K
from vars import class_names, num_classes, img_height, img_width , train_dir, val_dir


def shuffle_in_unison_scary(a, b):
    """
    Random shuffle array
    """
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def read_pil_image(img_path, height, width):
    """
        Read the image with PIL library
    """
    with open(img_path, 'rb') as f:
        return np.array(Image.open(f).convert('RGB').resize((height, width)))

def load_class(path, idxx):
    """
        Load a class into array
    """
    print( path )
    x_train = np.array([read_pil_image(str(p), img_height, img_width) for p in Path(path + '/' + train_dir + '/' + class_names[idxx]).rglob("*.jpg")])
    x_test  = np.array([read_pil_image(str(p), img_height, img_width) for p in Path(path + '/' + val_dir   + '/' + class_names[idxx]).rglob("*.jpg")])

    if K.image_data_format() == 'channels_first':
        x_train = np.rollaxis(x_train, 3, 1)
        x_test = np.rollaxis(x_test, 3, 1)

    return (x_train, np.ones(len(x_train)) * idxx), (x_test, np.ones(len(x_test)) * idxx)



def load_custom_data(path, training_size, validation_size):
    # TODO implement automatic detection of the sizes
    num_images_class = training_size # 919
    num_images_class_val = validation_size #  100
    x_train = np.empty((num_images_class * num_classes, img_width, img_height, 3), dtype='uint8')
    y_train = np.empty((num_images_class * num_classes), dtype='uint8')

    x_test = np.empty((num_images_class_val * num_classes, img_width, img_height, 3), dtype='uint8')
    y_test = np.empty((num_images_class_val * num_classes,), dtype='uint8')

    for idx in range(0, num_classes):
        (
            x_train[idx * num_images_class:(idx + 1) * num_images_class, :, :, :],
            y_train[idx * num_images_class:(idx + 1) * num_images_class]
        ), \
        (
            x_test[idx * num_images_class_val:(idx + 1) * num_images_class_val, :, :, :],
            y_test[idx * num_images_class_val:(idx + 1) * num_images_class_val]
        ) \
            = load_class(path, idx)

    shuffle_in_unison_scary(x_train, y_train)
    return (x_train, y_train), (x_test, y_test)
