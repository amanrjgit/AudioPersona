# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 11:51:26 2024

@author: Aman Jaiswar
"""

from main import add_new_speaker,train_model,Model2Pickle,train_multiclass_model,MulticlassModel,get_num_classes
from models import CNNModel1,CNNModel2
from myutils import record_audio, record_audio_without_saving, extract_audio_features
import tkinter as tk
from tkinter import ttk
import threading
import time
import numpy as np
from visuals import plot_waveform,plot_spectrogram,plot_mfcc,plot_harmonic_content
from statistics import mode
import warnings
warnings.filterwarnings("ignore")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Speaker Recognizer")
        self.geometry("600x500")
        self.tab_control = ttk.Notebook(self)

        self.home_tab = ttk.Frame(self.tab_control)
        self.train_tab = ttk.Frame(self.tab_control)

        self.tab_control.add(self.home_tab, text='Home')
        self.tab_control.add(self.train_tab, text='Train')

        self.create_home_tab()
        self.create_train_tab()

        self.tab_control.pack(expand=1, fill="both")
        
        self.model_instance = MulticlassModel()
        

    def create_home_tab(self):
        self.home_frame = ttk.Frame(self.home_tab)
        self.home_frame.pack(fill="both", expand=True)
        
        record_button = ttk.Button(self.home_frame, text="Record Audio", command=self.record_without_save)
        record_button.place(x=20,y=15)
        
        predict_speaker = ttk.Button(self.home_frame, text="Predict Speaker", command=self.predict_speaker)
        predict_speaker.place(x=110,y=15)
        
        self.predict_label = tk.Label(self.home_frame,text="Predicted Speaker: ",bg="#d9d9d9",fg="#8C8989",font=("Arial", 16, "bold"))
        self.predict_label.place(relx=0.015,rely=0.87)
        
        self.notebook = ttk.Notebook(self.home_frame)
        self.notebook.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.waveform_page = ttk.Frame(self.notebook)
        self.waveform_page.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        plot_waveform(y=[], sr=22050, root=self.waveform_page)
        self.notebook.add(self.waveform_page, text="Waveform")
        
        self.spectrogram_page = ttk.Frame(self.notebook)
        self.spectrogram_page.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        plot_spectrogram(y=[], sr=22050, root=self.spectrogram_page)
        self.notebook.add(self.spectrogram_page, text="Spectogram")
        
        self.harmonic_page = ttk.Frame(self.notebook)
        self.harmonic_page.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        plot_harmonic_content(y=[], sr=22050, root=self.harmonic_page)
        self.notebook.add(self.harmonic_page, text="Harmonic Content")
        
        self.mfcc_page = ttk.Frame(self.notebook)
        self.mfcc_page.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        plot_mfcc(y=[], sr=22050, root=self.mfcc_page)
        self.notebook.add(self.mfcc_page, text="MFCC")
        
    def record_without_save(self):
        # Clear existing plots
        try:
            self.notebook.forget(self.waveform_page)
            self.notebook.forget(self.spectrogram_page)
            self.notebook.forget(self.harmonic_page)
            self.notebook.forget(self.mfcc_page)
        except AttributeError:
            pass
        # Create new frames for each plot
        self.waveform_page = ttk.Frame(self.notebook)
        self.waveform_page.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.spectrogram_page = ttk.Frame(self.notebook)
        self.spectrogram_page.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.harmonic_page = ttk.Frame(self.notebook)
        self.harmonic_page.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.mfcc_page = ttk.Frame(self.notebook)
        self.mfcc_page.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.audio_array = record_audio_without_saving(duration=10)
            
        plot_waveform(y=self.audio_array, sr=22050, root=self.waveform_page)
        self.notebook.add(self.waveform_page, text="Waveform")
        
        plot_spectrogram(y=self.audio_array, sr=22050, root=self.spectrogram_page)
        self.notebook.add(self.spectrogram_page, text="Spectogram")
        
        plot_harmonic_content(y=self.audio_array, sr=22050, root=self.harmonic_page)
        self.notebook.add(self.harmonic_page, text="Harmonic Content")
        
        plot_mfcc(y=self.audio_array, sr=22050, root=self.mfcc_page)
        self.notebook.add(self.mfcc_page, text="MFCC")
        
    def predict_speaker(self):
        classes = get_num_classes()
        print(classes)
        if self.audio_array is not None:
            predictions = self.model_instance.predict(voice_data=self.audio_array)
            print(predictions)
            predicted_classes = []
            for i in predictions:
                max_index = np.argmax(i)
                # Get the actual class corresponding to the maximum probability
                actual_class = classes[max_index]
                predicted_classes.append(str(actual_class.split("_")[0]).title())
            print("Prediction:", mode(predicted_classes))
            self.predict_label.config(text=f"Predicted Speaker: {mode(predicted_classes)}")
        else:
            print("No audio data recorded yet.")
        

    def create_train_tab(self):
        train_frame = ttk.Frame(self.train_tab)
        
        # Label and entry for speaker name
        speaker_label = ttk.Label(train_frame, text="Speaker Name:")
        speaker_label.place(x=35,y=52)

        self.speaker_var = tk.StringVar()
        self.speaker_entry = ttk.Entry(train_frame, textvariable=self.speaker_var, validate="key", validatecommand=self.validate_speaker_name,width=58)
        self.speaker_entry.place(x=120,y=50)

        # Button to add speaker
        self.add_speaker_button = ttk.Button(train_frame, text="Add Speaker", state="disabled", command=self.add_speaker)
        self.add_speaker_button.place(x=482,y=48)
        
        # Create a custom style for the progress bar
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure("Custom.Horizontal.TProgressbar", troughcolor='#b2dfdb', bordercolor='#00796b', background='#00796b', foreground='black')
        
        collection_frame = tk.Frame(train_frame, bg="lightblue")
        collection_frame.place(relx=0.5, rely=0.57, anchor=tk.CENTER)
        
        # Label for voice data collection instructions
        collection_instructions = "Once upon a time, in a town nestled between the verdant mountains and the azure sea, there lived a skilled carpenter named Arjun. Arjun was known far and wide for his dedication to his craft. Every morning, as the sun painted the sky with hues of orange and pink, Arjun would begin his work, carving intricate designs into blocks of wood. The melodious sound of his tools against the wood was like music to the townsfolk's ears. Each piece he created was a testament to his skill and passion, embodying the perfect blend of functionality and art. His creations were not just objects, but stories captured in wood, waiting to be told. Evenings in the town were a spectacle to behold. As Arjun put away his tools, the setting sun would cast long shadows, dancing on his masterpieces. The aroma of freshly carved wood lingered in the air, a signature of another day spent in creation."
        collection_label = tk.Label(collection_frame, text=collection_instructions, wraplength=500, justify='left', font=('Arial', 12),bg="lightblue")
        collection_label.pack(padx=10, pady=10)
            
        self.recording_progress = ttk.Progressbar(train_frame, mode='indeterminate',style="Custom.Horizontal.TProgressbar",length=522)
        self.recording_progress.place(x=38,y=107)
        train_frame.pack(fill="both", expand=True)
        
        # Button to train model
        self.train_model_button = ttk.Button(train_frame, text="Train", state="normal", command=self.train)
        self.train_model_button.place(relx=0.45,rely=0.9)

    def validate_speaker_name(self):
        if len(self.speaker_var.get())>1:
            self.add_speaker_button.config(state="normal")
        else:
            self.add_speaker_button.config(state="disabled")
        return True
    
    def record(self):
        speaker_name = self.speaker_var.get()
        self.output = add_new_speaker(name = speaker_name)
        self.stop_recording()


    def start_recording(self):
        self.recording_thread = threading.Thread(target=self.record)
        self.recording_thread.start()
        self.recording_progress.start(10)
        
        
    def stop_recording(self):
        # Stop the recording thread
        self.recording_progress.stop()
        self.add_speaker_button.config(state="normal")
        self.train_model_button.config(state="normal")
    
    def add_speaker(self):
        print("Speaker added:", self.speaker_var.get())
        self.start_recording()
        
        
    def train(self):
        #train_model(filepath=self.output,Model=CNNModel)
        tk.messagebox.showinfo("Training","Training Started. \nPlease wait...")
        train_multiclass_model(filepath=r"C:\Users\Aman Jaiswar\Desktop\Python\Speech Recognizer\base\train", Model=CNNModel2)
        tk.messagebox.showinfo("Training","Training Completed")
        
        

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
