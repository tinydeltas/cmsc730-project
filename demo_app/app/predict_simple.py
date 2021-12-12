import subprocess
import time
from pprint import pprint

import numpy as np
from tensorflow import keras
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import os

receiver_pid = -1 

gestures = ["close", "cut", "no", "open", "switch", "paste", "maximize", "minimize"]

data_root = "./app/static/data"

model_path = data_root + "/model"

live_wav_directory = data_root + "/raw"
live_wav_path = os.path.join(data_root, "source.wav")
historical_wav_directory = data_root + "/raw_historical"

live_gesture_prediction = os.path.join(data_root, "gesture.txt")
live_predictions_path = os.path.join(data_root, "predictions.txt")
live_numpy_path = os.path.join(data_root, "live.npy")

detection_threshold = 0.35

dry_run = False

def get_spectrogram_array(np_save_path): 
    arr = np.load(np_save_path)
    return arr

def shape_data(arr):
    data = []
    data.append(arr)
    data = np.reshape(np.array(data), (1, 800, 581, 1))
    
    print(data.shape)

    return data

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
        specdata = get_spectrogram_array(live_numpy_path)
        data= shape_data(specdata)
        predictions = self.model.predict(data, verbose=1)
        print(predictions)
        all_probs = predictions[0]
        
        zipped = zip(all_probs, gestures)
        zipped_sorted = sorted(zipped, reverse=True)
        print(zipped_sorted)
        
        if zipped_sorted[0][0] > detection_threshold:
            label = zipped_sorted[0][1]
            print("predicted gesture: ", label, " with probability: ", zipped_sorted[0][0])
        else: 
            label = "None"
        
        with open(live_gesture_prediction, "w") as f: 
            f.write(label)
        
        d = {}
        for t in zipped_sorted: 
            d[t[1]] = t[0]
    
        with open(live_predictions_path, "w") as f: 
            print(d, file=f)

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
    