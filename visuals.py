# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:03:43 2024

@author: Aman Jaiswar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display
import tkinter as tk


def plot_waveform(y,sr,root):
    plt.figure(figsize=(8, 4))
    if len(y) != 0:
        librosa.display.waveshow(y, sr=sr,color='b')
    plt.title('Waveform Display')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def plot_spectrogram(y,sr,root):
    plt.figure(figsize=(8, 4))
    if len(y) != 0:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def plot_harmonic_content(y,sr,root):
    plt.figure(figsize=(8, 4))
    if len(y) != 0:
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        librosa.display.waveshow(y_harmonic, sr=sr, alpha=0.5, color='b')
        librosa.display.waveshow(y_percussive, sr=sr, alpha=0.5, color='r')
    plt.title('Harmonic Content')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(['Harmonic', 'Percussive'])
    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def plot_mfcc(y, sr, root):
    plt.figure(figsize=(8, 4))
    if len(y) != 0:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar()
    plt.title('MFCC')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficients')
    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)