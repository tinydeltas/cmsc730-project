import os
import wave
from shutil import copy2 as cp

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops

import params

n_mels = 128
    
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
    def __init__(self, gesture_label, datestamp, wavname=None): 
        self.gesture_label = gesture_label
        self.datestamp = datestamp 
        self.wavname = wavname 
        if wavname is not None: 
            self.y, self.sr = librosa.load(wavname)
        self.id= os.path.basename(wavname).strip(".wav")
    
    def sample_rate(self): 
        return self.sr

    def get_spectrogram(self, path=None): 
        # fig = plt.figure(num=None, figsize=(1, 1), dpi=100, frameon=False)
        
        si, fr = process_wav(self.wavname)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        plt.axis(ymin=0, ymax=8000)
        Pxx, freqs, bins, im = plt.specgram(si, Fs=fr, NFFT=params.nfft, noverlap=params.noverlap, cmap='gnuplot')
        # print(Pxx.shape)
        Pxx = np.abs(Pxx)
        f = Pxx[:800,:]
        f = np.expand_dims(f, axis=2)
        
        # self.save_spectrogram(fig, "spectrogram", False, path)
        self.save_spectrogram_array(f, "spectrogram", path)

        
    def save_spectrogram_array(self, pxx, kind, path): 
        if path is None: 
            path, base_path = self.get_path(kind)
            if not os.path.exists(base_path):
                print("creating directory: ", base_path)
                os.makedirs(base_path)
        np.save(path, pxx)
        
    def get_mel_spectrogram(self, path=None): 
        fig, ax = plt.subplots()
        
        S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=self.sr, fmax=8000, ax=ax)
        
        self.save_spectrogram(fig, "mel", True, path)

    def get_sfft_spectrogram(self, path=None): 
        fig, ax = plt.subplots()
        
        S_2 = np.abs(librosa.stft(self.y))
        s_db = librosa.amplitude_to_db(S_2, ref=np.max)
        img = librosa.display.specshow(s_db, y_axis='log', x_axis='time', ax=ax)
        
        self.save_spectrogram(fig, "stft", True, path)
    
    def get_sfft_chromagram(self, path=None): 
        fig, ax = plt.subplots()

        S = np.abs(librosa.stft(self.y))
        chroma = librosa.feature.chroma_stft(S=S, sr=self.sr)
        img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
        
        self.save_spectrogram(fig, "stftchroma", True, path)
    
    def get_mfcc(self, path=None):
        fig, ax = plt.subplots()
        
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=40)
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)

        
        self.save_spectrogram(fig, "mfcc", True, path)
        
    def get_chromagram(self, path=None): 
        fig, ax = plt.subplots()

        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
        
        self.save_spectrogram(fig, "chroma", True, path)
    
    def get_path(self, kind): 
        path_components = ["./data/processed/images", kind, 
                            self.gesture_label, 
                            self.datestamp + "/"]
        base_path = "/".join(path_components)
        # path = base_path + self.id + ".png"
        path = base_path + self.id
        return path, base_path 
        
        
    def save_spectrogram(self, fig, kind, librosa, path=None): 
        plt.axis('off')
    
        if path is None: 
            path, base_path = self.get_path(kind)
            if not os.path.exists(base_path):
                print("creating directory: ", base_path)
                os.makedirs(base_path)
        
        # resize correctly
        if librosa:
            fig.set_size_inches(1.29, 1.3)
            
        # if os.path.exists(path): 
        #     print("not saving to path: ", path)
        #     return

        plt.savefig(path, dpi=100)
        
        if librosa: 
            trim_image(path)
        
        im = resize_image(path)
        im.save(path)

    def get_all_spectrograms(self): 
        # self.get_mel_spectrogram()
        # self.get_sfft_spectrogram()
        # self.get_chromagram()
        # self.get_sfft_chromagram()
        # self.get_mfcc()
        self.get_spectrogram()


