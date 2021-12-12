from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tensorflow import keras
import subprocess
import time
import soundfile as sf 
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

receiver_pid = -1 

gestures = ["close", "cut", "no", "open", "switch", "paste", "maximize", "minimize"]

data_root = "./app/static/data"

model_path = data_root + "/model"

live_wav_directory = data_root + "/raw"
live_wav_path = os.path.join(data_root, "source.wav")
historical_wav_directory = data_root + "/raw_historical"

live_spectrogram_path = os.path.join(data_root, "s.png")
live_gesture_prediction = os.path.join(data_root, "gesture.txt")
live_predictions_path = os.path.join(data_root, "predictions.txt")

dry_run = False

# Used for spectrogram generation 
fs = 48000
nfft = int(0.1 * fs)
noverlap = int(0.095 * fs)


def get_spectrogram_live(wavpath):
    plt.figure(num=None, figsize=(3, 3), dpi=100, frameon=False)
    si, fr = sf.read(wavpath)
    Pxx, _, _, _ = plt.specgram(si, Fs=fr, NFFT=nfft, window=None, noverlap=noverlap)
    spec = 20 * np.log10(Pxx[0:800,:])

    print("saving spectrogram to: ", live_spectrogram_path)
    plt.savefig(live_spectrogram_path, dpi=100)
    return spec

def shape_data(arr):
    data, label = [], []

    data.append(arr)
    label = [ i for i in range(len(gestures))]

    data = np.reshape(np.array(data), (800, 581, 1))
    label = np.array(label)
    
    print(data.shape)
    print(label.shape)

    return data, label

def start_recording(): 
    global receiver_pid
    if not dry_run: 
        print("Starting recording")
        receiver_pid = subprocess.Popen(["python3", "app/receive.py"])

def stop_recording(): 
    receiver_pid.kill() 
        

class FileModifiedHandler(FileSystemEventHandler):
    def __init__(self, model): 
        self.model = model 
        
    def on_modified(self, event): 
        print(".wav file modified!")
        # Also saves the file
        specdata = get_spectrogram_live(live_wav_path)
        data, label = shape_data(specdata)
        
        input = np.concatenate(data, label)
        print(input.shape)
        predictions = self.model.predict(input, verbose=1)
        print(predictions)
        # zipped = zip(all_probs, keys)
        # zipped_sorted = sorted(zipped)
        # print(zipped_sorted)
        
        # s = sorted(all_probs, reverse=True)
        # if (s[0] > self.recognizer.detection_threshold()):
        #     idx = np.argmax(all_probs)
        #     label = keys[idx]
        #     print("predicted gesture: ", label, " with probability: ", all_probs[idx])
        # else: 
        #     label = "None"
        
        # with open(params.live_gesture_prediction, "w") as f: 
        #     f.write(label)
        
        # with open(params.live_predictions_path, "w") as f: 
        #     f.write(probabilities)

def main(): 
    model = keras.models.load_model(model_path)
    print(model.summary())

    start_recording()

    handler = FileModifiedHandler(model)
    
    observer = Observer()
    observer.schedule(handler, live_wav_directory, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        stop_recording()
    
    observer.join()
    
if __name__ == '__main__': 
    main()
    