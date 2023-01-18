import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

def evaluateModel(test_dir,modelPath):
    test_dir
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input,rescale=1.0/255.0)
    val_generator = test_datagen.flow_from_directory(test_dir, target_size=(200,200), batch_size=64, class_mode='binary')

    model = tf.keras.models.load_model(modelPath)

    acc = model.evaluate(val_generator, steps=len(val_generator),verbose=0)
    print(acc[1])

def Single_Image_Prediction(file):
    image = cv2.imread(file)
    # plt.imshow(image)
    # plt.show()
    image = cv2.resize(image,[200,200])
    img_arr = tf.keras.preprocessing.image.img_to_array(image)
    img_arr = img_arr/255.
    np_image = np.expand_dims(img_arr, axis=0)
    return np_image

def predictImage(file, modelPath):
    model = tf.keras.models.load_model(modelPath)
    image = Single_Image_Prediction(file)
    pred_value = model.predict(image)

    return pred_value
