#!/usr/bin/env python
# Create spectogram from audio file

# Libraries
import os
import sys
import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import subprocess
from shutil import copy2 as cp

# Define output file name

# Get file info   
def get_wav_info(wavname):
    wav = wave.open(wavname, 'r')
    frames = wav.readframes(-1)
    sound_info = np.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

# Define function for plotting
def graph_spectrogram(wavname):
    print(wavname)
    sound_info, frame_rate = get_wav_info(wavname)
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['savefig.facecolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['lines.color'] = 'white'
    plt.rcParams['text.color'] = 'white'    
    plt.rcParams['xtick.color'] = 'white'    
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    fig = plt.figure(num=None, figsize=(1, 1), dpi=100, frameon=False)
    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # ax = fig.add_subplot(111)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(500))
    # ax.tick_params(axis='both', direction='inout')
    # plt.title('Spectrogram of:\n %r' % os.path.basename(music_file))
    # plt.xlabel('time in seconds')
    # plt.ylabel('Frequency (Khz)')
    plt.specgram(sound_info, Fs=frame_rate, cmap='gnuplot')
    # cbar = plt.colorbar()
    # cbar.ax.set_ylabel('dB')
    name = os.path.basename(wavname).strip(".wav")
    plt.savefig('./data/images/'+name+".png")

# Save spectrogram
def main(): 
    directory = "./data/wav"
    for entry in os.scandir(directory):
        if (entry.path.endswith(".wav") and entry.is_file()):
            print(entry.path)
            graph_spectrogram(entry.path)
            # return

main()
