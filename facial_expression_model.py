# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:44:07 2019

@author: jaydeep thik
"""

import tensorflow as tf
from keras.utils import to_categorical
from tensorflow import keras
from keras import layers
from keras import models

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend


# backend.set_image_data_format('channels_last')

# ---------------------------------------------------------------------------------------------------------------------------------
def generate_dataset():
    """generate dataset from csv"""

    df = pd.read_csv("./resources/fer2013/fer2013.csv")
    indexNames = df[(df['emotion'] != 4) & (df['emotion'] != 3) & (df['emotion'] != 6)].index
    df.drop(indexNames, inplace=True)
    df = df.replace(4, 0)
    df = df.replace(3, 1)
    df = df.replace(6, 2)

    print(df.head)

    train_samples = df[df['Usage'] == "Training"]
    validation_samples = df[df["Usage"] == "PublicTest"]
    test_samples = df[df["Usage"] == "PrivateTest"]

    y_train = train_samples.emotion.astype(np.int32).values
    y_valid = validation_samples.emotion.astype(np.int32).values
    y_test = test_samples.emotion.astype(np.int32).values
    # buscar que sea 50/50 entre happy y sad

    sad = 0
    happy = 0
    neutral = 0
    for elem in y_test:
        if elem == 0:
            sad += 1
        if elem == 1:
            happy += 1
        if elem == 2:
            neutral += 1
    print("sad", sad / len(y_test))
    print("Happy", happy / len(y_test))
    print("Neutral", neutral / len(y_test))

    X_train = np.array([np.fromstring(image, np.uint8, sep=" ").reshape((48, 48)) for image in train_samples.pixels])
    X_valid = np.array(
        [np.fromstring(image, np.uint8, sep=" ").reshape((48, 48)) for image in validation_samples.pixels])
    X_test = np.array([np.fromstring(image, np.uint8, sep=" ").reshape((48, 48)) for image in test_samples.pixels])

    return X_train, y_train, X_valid, y_valid, X_test, y_test


# ---------------------------------------------------------------------------------------------------------------------------------

def generate_model(lr=0.001):
    """training model"""

    model_n = models.Sequential()
    model_n.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
    model_n.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model_n.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model_n.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model_n.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model_n.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model_n.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model_n.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model_n.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model_n.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model_n.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model_n.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model_n.add(layers.Flatten())
    model_n.add(layers.Dense(64, activation='relu'))
    model_n.add(layers.Dense(64, activation='relu'))
    model_n.add(layers.Dense(3, activation='softmax'))
    model_n.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('Training....')
    print(model_n.summary())
    return model_n


# ---------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # df = pd.read_csv("./fer2013/fer2013.csv")
    X_train, y_train, X_valid, y_valid, X_test, y_test = generate_dataset()

    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    y_test = to_categorical(y_test)

    X_train = X_train.reshape((X_train.shape[0], 48, 48, 1)).astype(np.float32)
    X_valid = X_valid.reshape((X_valid.shape[0], 48, 48, 1)).astype(np.float32)
    X_test = X_test.reshape((X_test.shape[0], 48, 48, 1)).astype(np.float32)

    X_train /= 255
    X_valid /= 255
    X_test /= 255

    model = generate_model(0.01)
    history = model.fit(X_train, y_train, batch_size=128, epochs=15, validation_data=(X_valid, y_valid),
                        shuffle=True)
    model.save("my_model.h5")
