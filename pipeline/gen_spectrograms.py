import os
import numpy as np
import wave
import matplotlib.pyplot as plt
from shutil import copy2 as cp
import librosa 
import librosa.display
from PIL import Image, ImageChops

from constants import *

n_mels = 128
height = 100 
width = 100
    
def trim_image(name):
    im = Image.open(name)
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.convert('RGB').getbbox()
    if not bbox: 
        exit
    return im.crop(bbox)

def resize_image(name): 
    im = Image.open(name)
    return im.resize((height, width))
        
def process_wav(wavname):
    wav = wave.open(wavname, 'r')
    frames = wav.readframes(-1)
    si= np.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return si, frame_rate
    
class SpectrogramLoader: 
    def __init__(self, wavname): 
        self.wavname = wavname 
        self.y, self.sr = librosa.load(wavname)
        self.name = os.path.basename(wavname).strip(".wav")

    def get_spectrogram(self): 
        fig = plt.figure(num=None, figsize=(1, 1), dpi=100, frameon=False)
        
        si, fr = process_wav(self.wavname)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        plt.specgram(si, Fs=fr, cmap='gnuplot')
        
        self.save_spectrogram(fig, "spectrogram", librosa=False)
    
    def get_mel_spectrogram(self): 
        fig, ax = plt.subplots()
        
        S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=self.sr, fmax=8000, ax=ax)
        
        self.save_spectrogram(fig, "mel", librosa=True)

    def get_sfft_spectrogram(self,): 
        fig, ax = plt.subplots()
        
        S_2 = np.abs(librosa.stft(self.y))
        s_db = librosa.amplitude_to_db(S_2, ref=np.max)
        img = librosa.display.specshow(s_db, y_axis='log', x_axis='time', ax=ax)
        
        self.save_spectrogram(fig, "stft", librosa=True)
    
    def get_sfft_chromagram(self): 
        fig, ax = plt.subplots()

        S = np.abs(librosa.stft(self.y))
        chroma = librosa.feature.chroma_stft(S=S, sr=self.sr)
        img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
        
        self.save_spectrogram(fig, "stftchroma", librosa=True)
    
    def get_mfcc(self):
        fig, ax = plt.subplots()
        
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=40)
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
        
        self.save_spectrogram(fig, "mfcc", librosa=True)
        
    def get_chromagram(self): 
        fig, ax = plt.subplots()

        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
        
        self.save_spectrogram(fig, "chroma", librosa=True)
    
        
    def save_spectrogram(self, fig, kind, librosa): 
        plt.axis('off')
        
        label = self.name.split("_")
        gesture_name, n = label[0], label[1]
        
        path_components = ["./data/images", kind, "all", gesture_name +"/"]
        base_path = "/".join(path_components)
        full_name =  base_path + n + ".png"
        
        # resize correctly
        if librosa:
            fig.set_size_inches(1.29, 1.3)

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        plt.savefig(full_name, dpi=100)
        
        if librosa: 
            trim_image(full_name)
        
        im = resize_image(full_name)
        im.save(full_name)

    def all_spectrograms(self): 
        self.get_mel_spectrogram()
        self.get_sfft_spectrogram()
        self.get_chromagram()
        self.get_sfft_chromagram()
        self.get_mfcc()
        self.get_spectrogram()

def main(): 
    
    for entry in os.scandir(param_wav_directory):
        if (entry.path.endswith(".wav") and entry.is_file()):
            print(entry.path)
            loader = SpectrogramLoader(entry.path)
            loader.all_spectrograms()

main()