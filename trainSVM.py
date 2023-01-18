import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras import layers
import cv2, os
from tqdm import tqdm
from random import shuffle
import shutil

train_dir= "./training_set/training_set/"
test_dir= "./test_set/test_set/"
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range = 0.2,
                                   zoom_range = 0.2,
                                   preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
                                   horizontal_flip = True)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input,rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(200,200), batch_size=64, class_mode='binary')
val_generator = test_datagen.flow_from_directory(test_dir, target_size=(200,200), batch_size=64, class_mode='binary')

# model = tf.keras.models.load_model('catsvsdogs.h5')
# acc = model.evaluate(val_generator, steps=len(val_generator),verbose=0)
# print(acc[1])

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters = 32, padding = "same",activation = "relu",kernel_size=3, strides = 2,input_shape=(200,200,3)))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides = 2))

model.add(tf.keras.layers.Conv2D(filters = 32, padding = "same",activation = "relu",kernel_size=3))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides = 2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation="relu"))

model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation = "linear"))
model.summary()
# Model Compilation
model.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])
history = model.fit(train_generator, epochs=50, validation_data=val_generator)
acc = model.evaluate(val_generator,verbose=0)
print(acc[1])

model.save('./models/catsvsdogsSVM.h5')

 # plot loss
plt.subplot(211)
plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], color='blue', label='train')
plt.plot(history.history['val_loss'], color='orange', label='test')
plt.show()
# plot accuracy
plt.subplot(212)
plt.title('Classification Accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train')
plt.plot(history.history['val_accuracy'], color='orange', label='test')
plt.show()