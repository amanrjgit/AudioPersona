# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:41:38 2024

@author: Aman Jaiswar
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras import regularizers
from spela.spectrogram import Spectrogram
from spela.melspectrogram import Melspectrogram
import warnings
warnings.filterwarnings("ignore")


def BaseModel(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
    return model

def BaseModelMulti(input_shape,num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=input_shape),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer for multiclass classification
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def CNNModel1(input_shape,num_classes,speech_feature=None):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape),  # Reshape input for Conv1D
        Conv1D(128, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(3),
        Conv1D(128, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(3),
        Conv1D(64, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(3),
        Flatten(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer for multiclass classification
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def CNNModel2(input_shape,num_classes,speech_feature=None):
    initial_learning_rate = 0.0003
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,  # Adjust the decay steps according to your dataset size and training duration
        decay_rate=0.96,   # Adjust the decay rate as needed
        staircase=True)
    
    model = tf.keras.Sequential()
    if speech_feature == "spectrogram":
        model.add(Spectrogram(input_shape=(1, 66150)))
    elif speech_feature == "melspectrogram":
        model.add(Melspectrogram(input_shape=(1, 66150), name='melgram'))
        
    model.add(tf.keras.layers.Conv2D(64, 3, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
        
    model.add(tf.keras.layers.Conv2D(64, 3, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    
    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizers.l2(0.01)))
    
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    return model