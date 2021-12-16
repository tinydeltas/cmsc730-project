import sys
import os 
import time
from datetime import datetime
import matlab.engine
import subprocess
import matplotlib
import soundfile as sf 
import numpy as np

import matplotlib.pyplot as plt

eng = matlab.engine.start_matlab()
run = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")
fs = 48000
nfft = int(0.1 * fs)
noverlap = int(0.095 * fs)

directory_default = "audio_new_microphone_sync_4s"
gesture_label_default = "1s"

def get_spectrogram_live(wavpath, save_path):
    plt.figure(num=None, figsize=(3, 3), dpi=100, frameon=False)
    si, fr = sf.read(wavpath)
    
    # plt.axis(ymin=0, ymax=8000)
    Pxx, _, _, _ = plt.specgram(si, Fs=fr, NFFT=nfft, window=None, noverlap=noverlap)
    spec = 20 * np.log10(Pxx[0:800,:])

    print("saving spectrogram to: ", save_path)
    plt.savefig(save_path, dpi=100)
    
    # np.save(live_numpy_path, spec)
    return spec

def record(directory, gesture_label, i): 
    # print("Starting transmission")
    # child = subprocess.Popen(["python3", "transmit.py"])
    print("Saving to: ", directory)
    if not os.path.exists(directory): 
        print("Making folder")
        os.makedirs(directory)
        
    print("Gesture label", gesture_label)
    fullpath = os.path.join(directory, gesture_label + "/" + run)
    if not os.path.exists(fullpath): 
        print("Making folder ", fullpath)
        os.makedirs(fullpath)

    while True: 
        try: 
            print("Recording now!")
            print("#: t", i)
            
            path = os.path.join(fullpath, str(i))
            print("Saving to: ", path)
            
        
            eng.Receiver_Signal_FMCW(path, nargout=0)
            get_spectrogram_live(path + ".wav", path + ".png")
            i += 1
            
            # input("Press enter to continue, or Ctrl-C to quit")
            print("Finished recording, sleeping for 3")
            time.sleep(3)
        
        except KeyboardInterrupt:
            # child.kill()
            eng.quit()
            sys.exit(0)
    
def main():
    directory = directory_default
    gesture_label = gesture_label_default
    
    if len(sys.argv) > 1: 
        directory = sys.argv[1]

    if len(sys.argv) > 2: 
        gesture_label = sys.argv[2]
    
    record(directory, gesture_label, 0) 

main()