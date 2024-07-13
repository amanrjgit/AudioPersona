# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:44:16 2024

@author: Aman Jaiswar
"""
from myutils import *
import os
from sklearn.model_selection import train_test_split
from models import BaseModel
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from spela.spectrogram import Spectrogram 
from spela.melspectrogram import Melspectrogram
from statistics import mode
import warnings
warnings.filterwarnings("ignore")
    
training_dir = r"C:\Users\Aman Jaiswar\Desktop\Python\Speech Recognizer" 

if not os.path.exists(training_dir+"\\base"):
    os.mkdir(training_dir+"\\base")
if not os.path.exists(training_dir+"\\base\\audios"):
    os.mkdir(training_dir+"\\base\\audios")
if not os.path.exists(training_dir+"\\base\\train"):
    os.mkdir(training_dir+"\\base\\train")
if not os.path.exists(training_dir+"\\base\\models"):
    os.mkdir(training_dir+"\\base\\models")

def get_num_classes():
    return os.listdir(training_dir+"\\base\\train")

def add_new_speaker(name):
    name = name.lower()
    counter = 0
    for file in os.listdir(training_dir+"\\base\\audios"):
        if file.split("_")[0] == name:
            counter += 1
    new_name = name + "_" + str(counter)
    user_audio_path = f'{training_dir}\\base\\audios\\{new_name}.wav'
    record_audio(file_name=user_audio_path, duration=45)
    if not os.path.exists(training_dir+f"\\base\\train\\{new_name}"):
        os.mkdir(training_dir+f"\\base\\train\\{new_name}")
    split_mp3_to_segments(user_audio_path, training_dir+f"\\base\\train\\{new_name}")
    return training_dir+f"\\base\\train\\{new_name}"

def train_model(filepath,Model):
    x,y = prepare_training_data(filepath)
    input_shape = x[0].shape
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    model = Model(input_shape)
    model.fit(x_train,y_train,epochs=20,batch_size=16)
    model_name = os.path.basename(filepath)
    model.save(training_dir+f"\\base\\models\\{model_name}.keras")
    pass

def train_multiclass_model(filepath,Model):
    x,y,num_classes = prepare_multiclass_training_data(filepath,model=2)
    print(num_classes)
    input_shape = x[0].shape
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model = Model(input_shape,num_classes=num_classes,speech_feature="spectrogram")
    model.fit(x_train, y_train, epochs=10, batch_size= 20, validation_data=(x_test, y_test), callbacks=[early_stopping])
    model.save(training_dir+"\\base\\models\\multiclass.keras")
    
    pass

class Model2Pickle:
    def __init__(self, model_folder=r"C:\Users\Aman Jaiswar\Desktop\Python\Speech Recognizer\base\models"):
        self.model_folder = model_folder
        self.models = self._load_models()

    def _load_models(self):
        models_ = {}
        for file in os.listdir(self.model_folder):
            if file.endswith('.keras'):
                model_path = os.path.join(self.model_folder, file)
                speaker_name = file.split('.')[0]
                model = tf.keras.models.load_model(model_path)
                models_[speaker_name] = model
        return models_

    def predict(self, voice_data):
        predictions = {}
        for speaker_name, model in self.models.items():
            speaker_prediction = model.predict(voice_data)
            predictions[speaker_name] = round(speaker_prediction[0][0],5)
        return predictions
    
class MulticlassModel:
    def __init__(self, model_folder=r"C:\Users\Aman Jaiswar\Desktop\Python\Speech Recognizer\base\models\multiclass.keras"):
        self.model_folder = model_folder
        self.model = tf.keras.models.load_model(self.model_folder,custom_objects={'Spectrogram': Spectrogram, 'Melspectrogram': Melspectrogram})
        
    def predict(self, voice_data):
        predictions = []
        audio_splits = split_mp3(voice_data)
        for i in audio_splits:
            extracted_features = load_audio(i,file=False)# extract_audio_features(i,file=False) 
            #extracted_features = extracted_features.reshape(1,-1)
            extracted_features = np.expand_dims(extracted_features, axis=1)
            pred = self.model.predict(extracted_features)
            predictions.append(pred)
        return predictions
