# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:35:06 2024

@author: Aman Jaiswar
"""

import librosa
import os
import numpy as np
import pyaudio
import wave
import soundfile as sf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random
import warnings
warnings.filterwarnings("ignore")

def time_stretch(audio, rate):
    y, sr = librosa.load(audio, sr=22050)
    y, _ = librosa.effects.trim(y)
    
    y = librosa.effects.time_stretch(y=y,rate=rate)
    
    reshaped_array = np.reshape(y, [1, -1])

    # Check if audio needs padding or truncation
    if reshaped_array.shape[1] < 66150:
        # If the audio is shorter than the target shape, pad it with zeros
        padded_audio = np.pad(reshaped_array, ((0, 0), (0, 66150 - reshaped_array.shape[1])), mode='constant')
        return padded_audio
    elif reshaped_array.shape[1] > 66150:
        # If the audio is longer than the target shape, truncate it
        truncated_audio = reshaped_array[:, :66150]
        return truncated_audio
    else:
        return reshaped_array

def pitch_shift(audio, n_steps, sr=22050):
    y, sr = librosa.load(audio, sr=sr)
    y, _ = librosa.effects.trim(y)
    
    y = librosa.effects.pitch_shift(y=y,sr=sr,n_steps=n_steps)
    
    reshaped_array = np.reshape(y, [1, -1])
    
    # Check if audio needs padding or truncation
    if reshaped_array.shape[1] < 66150:
        # If the audio is shorter than the target shape, pad it with zeros
        padded_audio = np.pad(reshaped_array, ((0, 0), (0, 66150 - reshaped_array.shape[1])), mode='constant')
        return padded_audio
    elif reshaped_array.shape[1] > 66150:
        # If the audio is longer than the target shape, truncate it
        truncated_audio = reshaped_array[:, :66150]
        return truncated_audio
    else:
        return reshaped_array

def time_mask(audio, max_time_mask):
    y, sr = librosa.load(audio, sr=22050)
    y, _ = librosa.effects.trim(y)
    
    t = len(y)
    t_mask = random.randint(0, max_time_mask)
    start = random.randint(0, t - t_mask)
    masked_audio = y.copy()
    masked_audio[start:start + t_mask] = 0
    
    reshaped_array = np.reshape(masked_audio, [1, -1])
    
    # Check if audio needs padding or truncation
    if reshaped_array.shape[1] < 66150:
        # If the audio is shorter than the target shape, pad it with zeros
        padded_audio = np.pad(reshaped_array, ((0, 0), (0, 66150 - reshaped_array.shape[1])), mode='constant')
        return padded_audio
    elif reshaped_array.shape[1] > 66150:
        # If the audio is longer than the target shape, truncate it
        truncated_audio = reshaped_array[:, :66150]
        return truncated_audio
    else:
        return reshaped_array

def frequency_mask(audio, max_freq_mask):
    y, sr = librosa.load(audio, sr=22050)
    y, _ = librosa.effects.trim(y)
    
    f = y.shape[0]
    f_mask = random.randint(0, max_freq_mask)
    start = random.randint(0, f - f_mask)
    masked_audio = y.copy()
    masked_audio[start:start + f_mask] = 0
    
    reshaped_array = np.reshape(masked_audio, [1, -1])
    
    # Check if audio needs padding or truncation
    if reshaped_array.shape[1] < 66150:
        # If the audio is shorter than the target shape, pad it with zeros
        padded_audio = np.pad(reshaped_array, ((0, 0), (0, 66150 - reshaped_array.shape[1])), mode='constant')
        return padded_audio
    elif reshaped_array.shape[1] > 66150:
        # If the audio is longer than the target shape, truncate it
        truncated_audio = reshaped_array[:, :66150]
        return truncated_audio
    else:
        return reshaped_array

def add_awgn(audio_path, snr_dB=15):
    signal, sr = librosa.load(audio_path, sr=22050)
    signal, _ = librosa.effects.trim(signal)
    signal = librosa.util.normalize(signal)
    # Calculate noise power
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_dB / 10.0))

    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    # Add noise to the signal
    noisy_signal = signal + noise
    reshaped_array = np.reshape(noisy_signal, [1, -1])

    # Check if audio needs padding or truncation
    if reshaped_array.shape[1] < 66150:
        # If the audio is shorter than the target shape, pad it with zeros
        padded_audio = np.pad(reshaped_array, ((0, 0), (0, 66150 - reshaped_array.shape[1])), mode='constant')
        return padded_audio
    elif reshaped_array.shape[1] > 66150:
        # If the audio is longer than the target shape, truncate it
        truncated_audio = reshaped_array[:, :66150]
        return truncated_audio
    else:
        return reshaped_array



def extract_audio_features(audio_file, sample_rate=22050,file=True):
    # Load audio file
    if file:
        y, sr = librosa.load(audio_file, sr=sample_rate)
        y, _ = librosa.effects.trim(y)
    else:
        y = audio_file
        sr = sample_rate
    
    # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = mfccs.flatten()
    desired_length = 2808
    # Check if the length of mfccs exceeds the desired length
    if len(mfccs) > desired_length:
        # Truncate the MFCC sequence
        mfccs = mfccs[:desired_length]
    elif len(mfccs) < desired_length:
        # Pad the MFCC sequence with zeros
        mfccs = np.pad(mfccs, (0, desired_length - len(mfccs)), mode='constant')
    
    # Extract prosodic features
    rms = librosa.feature.rms(y=y)
    intensity = np.mean(rms)
    
    # Extract VTLN (Vocal Tract Length Normalization)
    # Example: Pitch shifting
    y_shifted = np.array(librosa.effects.pitch_shift(y, sr=sr, n_steps=4))
    y_shifted = y_shifted.flatten()
    if len(y_shifted) > 66150:
        y_shifted = y_shifted[:66150]
    elif len(y_shifted) < 66150:
        y_shifted = np.pad(y_shifted, ((0, 0), 66150 - len(y_shifted)), mode='constant')
    
    # Extract chroma_stft (Chromagram)
    chroma = np.array(librosa.feature.chroma_stft(y=y, sr=sr))
    chroma = chroma.flatten()
    if len(chroma) > 2592:
        # Truncate the MFCC sequence
        chroma = chroma[:2592]
    elif len(chroma) < 2592:
        # Pad the MFCC sequence with zeros
        chroma = np.pad(chroma, (0, 2592 - len(chroma)), mode='constant')
        
    # Combine all features
    audio_features = np.concatenate([
        mfccs.flatten(), 
        np.array([intensity]),  # Ensure intensity is a 1D array
        y_shifted.flatten(), 
        chroma.flatten()
    ])
    
    return audio_features

def load_audio(audio_path,sample_rate=22050,file=True):
    if file:
        y, sr = librosa.load(audio_path, sr=sample_rate)
        y, _ = librosa.effects.trim(y)
    else:
        y = audio_path
        sr = sample_rate
    y = librosa.util.normalize(y)
    reshaped_array = tf.reshape(y, [1, -1])
    if reshaped_array.shape[1] < 66150:
        # If the audio is shorter than the target shape, pad it with zeros
        padded_audio = np.pad(reshaped_array, ((0, 0), (0, 66150 - reshaped_array.shape[1])), mode='constant')
        return padded_audio
    elif reshaped_array.shape[1] > 66150:
        # If the audio is longer than the target shape, truncate it
        truncated_audio = reshaped_array[:, :66150]
        return truncated_audio
    else:
        return reshaped_array


def prepare_training_data(files):
    dummy = "C:\\Users\\Aman Jaiswar\\Desktop\\Python\\Speech Recognizer\\base\\dummy"
    x = []
    y = []
    for file in os.listdir(files):
        path = os.path.join(files,file)
        x.append(extract_audio_features(path,sample_rate=22050))
        y.append(1)
    for file in os.listdir(dummy):
        path = os.path.join(dummy,file)
        x.append(extract_audio_features(path,sample_rate=22050))
        y.append(0)
    return np.array(x),np.array(y)

def prepare_multiclass_training_data(filepath, model):
    x = []
    y = []
    num_classes = np.arange(len(os.listdir(filepath)))
    print("classes:",num_classes)
    one_hot_encoded_classes = to_categorical(num_classes)
    print("encoded:", one_hot_encoded_classes)
    
    for Class, file in enumerate(os.listdir(filepath)):
        dir_path = os.path.join(filepath, file)
        for f in os.listdir(dir_path):
            if model == 1:
                # Extract features and add AWGN separately
                features = extract_audio_features(os.path.join(dir_path, f), sample_rate=22050)
                noisy_signal = add_awgn(os.path.join(dir_path, f))
                x.append(features)
                x.append(noisy_signal)
            elif model == 2:
                # Load audio and add AWGN separately
                signal = load_audio(os.path.join(dir_path, f), sample_rate=22050)
                noisy_signal = add_awgn(os.path.join(dir_path, f))
                time_stretch_audio = time_stretch(os.path.join(dir_path, f),rate=0.7)
                pitch_shift_audio = pitch_shift(os.path.join(dir_path, f),n_steps=4)
                time_mask_audio = time_mask(os.path.join(dir_path, f),max_time_mask=300)
                frequency_mask_audio = frequency_mask(os.path.join(dir_path, f),max_freq_mask=30)                
                
                x.append(signal)
                x.append(noisy_signal)
                x.append(time_stretch_audio)
                x.append(pitch_shift_audio)
                x.append(time_mask_audio)
                x.append(frequency_mask_audio)
            
            y.append(np.array(one_hot_encoded_classes[Class]))
            y.append(np.array(one_hot_encoded_classes[Class]))
            y.append(np.array(one_hot_encoded_classes[Class]))
            y.append(np.array(one_hot_encoded_classes[Class]))
            y.append(np.array(one_hot_encoded_classes[Class]))
            y.append(np.array(one_hot_encoded_classes[Class]))
            
    return np.array(x), np.array(y), len(num_classes)

        
def record_audio(file_name=None, duration=30, sample_rate=22050, chunk_size=1024, channels=1):
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open a new stream for recording
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Recording...")

    frames = []

    # Record audio in chunks
    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    
    
    # Save the recorded audio to a WAV file
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as {file_name}")
        
def record_audio_without_saving(duration=10,sample_rate=22050, chunk_size=1024):
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open a new stream for recording
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Recording without saving...")

    frames = []

    # Record audio in chunks
    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    audio_data = b''.join(frames)

    # Convert audio data to NumPy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    audio_array = audio_array.astype(np.float32) / np.iinfo(np.int16).max

    
    return audio_array


def split_mp3_to_segments(mp3_file_path, output_directory,save=True):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the MP3 file
    audio, sample_rate = librosa.load(mp3_file_path, sr=None)

    # Calculate the duration of each segment in samples (2 seconds)
    segment_duration = sample_rate * 3

    # Split the audio into 2-second segments
    for i, start_sample in enumerate(range(0, len(audio), segment_duration)):
        # Extract the segment
        segment = audio[start_sample:start_sample + segment_duration]

        # Define the output file name
        output_file_name = os.path.join(output_directory, f"{os.path.splitext(os.path.basename(mp3_file_path))[0]}_{i}.wav")

        # Save the segment as a new WAV file
        sf.write(output_file_name, segment, sample_rate)
        
    return None

def split_mp3(audio_array):
    segment_duration = 22050 * 3
    audios = []
    for i, start_sample in enumerate(range(0, len(audio_array), segment_duration)):
        # Extract the segment
        segment = audio_array[start_sample:start_sample + segment_duration]
        audios.append(np.array(segment))
    return audios
        