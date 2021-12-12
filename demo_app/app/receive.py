import os
from shutil import copyfile
import matlab.engine
import matplotlib

import matplotlib.pyplot as plt
import numpy as np 
import soundfile as sf

data_root = "./app/static/data"

live_wav_directory = data_root + "/raw"
live_wav_path = os.path.join(live_wav_directory, "source.wav")
historical_wav_directory = data_root + "/raw_historical"
live_spectrogram_path = os.path.join(data_root, "s.png")
live_numpy_path = os.path.join(data_root, "live")

matplotlib.use('Agg')

def cleanup(): 
    if i >= n_keep: 
        last = i - n_keep
        name = str(last) 
        full_path_wav = os.path.join(historical_wav_directory, name + ".wav")
        full_path_mat = os.path.join(historical_wav_directory, name + ".mat")
        
        print("removing: ", full_path_wav, full_path_mat)
        os.remove(full_path_wav)
        os.remove(full_path_mat)


# Used for spectrogram generation 
fs = 48000
nfft = int(0.1 * fs)
noverlap = int(0.095 * fs)

def get_spectrogram_live(wavpath, save_path):
    plt.figure(num=None, figsize=(3, 3), dpi=100, frameon=False)
    si, fr = sf.read(wavpath)
    Pxx, _, _, _ = plt.specgram(si, Fs=fr, NFFT=nfft, window=None, noverlap=noverlap)
    spec = 20 * np.log10(Pxx[0:800,:])

    print("saving spectrogram to: ", save_path)
    plt.savefig(save_path, dpi=100)
    
    np.save(live_numpy_path, spec)
    return spec

def receive(full_path): 
    print("receiving")
    
    eng.Receiver_Signal_FMCW(full_path, nargout=0)
    
    get_spectrogram_live(live_wav_path, full_path + ".png")
    
    print("copying to: ", live_wav_path)
    copyfile(full_path + ".png", live_spectrogram_path)
    copyfile(full_path + ".wav", live_wav_path)
    

if __name__ == '__main__': 
    eng = matlab.engine.start_matlab()

    if not os.path.exists(live_wav_directory): 
        os.makedirs(live_wav_directory)

    if not os.path.exists(historical_wav_directory): 
        os.makedirs(historical_wav_directory)

    n_keep = 20
    i = 0 

    while True: 
        full_path = os.path.join(historical_wav_directory, str(i))
        receive(full_path)
        cleanup()
        i += 1
