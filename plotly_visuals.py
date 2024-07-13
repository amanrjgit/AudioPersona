# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:22:20 2024

@author: Aman Jaiswar
"""

from plotly.subplots import make_subplots
import tkinter as tk
from tkinter import ttk
import numpy as np
import plotly.graph_objs as go
import librosa
import librosa.display
from tkinterweb import WebView

def plot_waveform(y, sr, root):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(y)) / sr, y=y, mode='lines', name='Waveform', line=dict(color='blue')))
    fig.update_layout(title='Waveform Display', xaxis_title='Time (s)', yaxis_title='Amplitude')
    waveform_html = fig.to_html(full_html=False)
    waveform_widget = WebView(root, html=waveform_html)
    waveform_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=10, pady=10)

def plot_spectrogram(y, sr, root):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig = go.Figure(data=go.Heatmap(z=D, x=np.arange(D.shape[1]) / sr, y=np.arange(D.shape[0])))
    fig.update_layout(title='Spectrogram', xaxis_title='Time (s)', yaxis_title='Frequency (Hz)')
    spectrogram_html = fig.to_html(full_html=False)
    spectrogram_widget = WebView(root, html=spectrogram_html)
    spectrogram_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=10, pady=10)

def plot_harmonic_content(y, sr, root):
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=np.arange(len(y_harmonic)) / sr, y=y_harmonic, mode='lines', name='Harmonic', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(y_percussive)) / sr, y=y_percussive, mode='lines', name='Percussive', line=dict(color='red')), row=2, col=1)
    fig.update_layout(title='Harmonic Content', xaxis_title='Time (s)', yaxis_title='Amplitude')
    harmonic_html = fig.to_html(full_html=False)
    harmonic_widget = WebView(root, html=harmonic_html)
    harmonic_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=10, pady=10)

def plot_sonogram(y, sr, root):
    D = np.abs(librosa.stft(y))
    fig = go.Figure(data=go.Heatmap(z=librosa.amplitude_to_db(D, ref=np.max), x=np.arange(D.shape[1]) / sr, y=np.arange(D.shape[0])))
    fig.update_layout(title='Sonogram', xaxis_title='Time (s)', yaxis_title='Frequency (Hz)')
    sonogram_html = fig.to_html(full_html=False)
    sonogram_widget = WebView(root, html=sonogram_html)
    sonogram_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=10, pady=10)
