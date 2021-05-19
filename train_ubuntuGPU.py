import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, Dropout
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
#import cv2
from glob import glob

#----------------GPU------------------#
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
#-------------------------------------#
train_dir = ("train")

test_dir = ("test")

train_pic_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.5, horizontal_flip=True, fill_mode='nearest')

test_pic_gen = ImageDataGenerator(rescale=1. / 255)

train_flow = train_pic_gen.flow_from_directory(train_dir, (128, 128), batch_size=32, class_mode='categorical', color_mode='grayscale')

test_flow = test_pic_gen.flow_from_directory(test_dir, (128, 128), batch_size=32, class_mode='categorical', color_mode='grayscale')


def get_num_of_classes():
    return len(glob('train/*'))

def show_train_history(train_history, train, validation):
	plt.plot(train_history.history[train])
	plt.plot(train_history.history[validation])
	plt.title('Train History')
	plt.ylabel(train)
	plt.xlabel('Epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()


# 訓練
model = Sequential([
    Convolution2D(64, 7, 7, input_shape=(128, 128, 1), activation='relu'),#輸入層 filter=64 kernel=7x7 
    MaxPool2D(pool_size=(2, 2)),#池化層
    Convolution2D(32, 5, 5, input_shape=(64, 64, 1), activation='relu'),#捲積層 filter=32 kernel=5x5
    MaxPool2D(pool_size=(2, 2)),#池化層
    Convolution2D(16, 3, 3, input_shape=(32, 32, 1), activation='relu'),#捲積層 filter=16 kernel=3x3
    MaxPool2D(pool_size=(2, 2)),#池化層
    Flatten(),#平坦層
    Dense(56, activation='relu'),#隱藏層 56個神經元
    Dropout(0.3),#原本是0.5 但是我沒遇到overfitting 所以往下調
    # Dense(64,activation='relu'),
    Dense(6, activation='softmax')#輸出層
])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

#model = models.load_model('cnn_model_gpu_v2.h5')

#train_history=model.fit_generator(train_flow, steps_per_epoch=100, epochs=100, verbose=1, validation_data=test_flow, validation_steps=100)
train_history=model.fit_generator(train_flow, steps_per_epoch=100, epochs=100, verbose=1, validation_data=test_flow,validation_steps=100)

model.save('cnn_model_gpu_v3.h5')
show_train_history(train_history,'accuracy','val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')
