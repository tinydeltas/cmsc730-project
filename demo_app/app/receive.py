import os
from shutil import copyfile
import matlab.engine

data_root = "./app/static/data"

live_wav_directory = data_root + "/raw"
live_wav_path = os.path.join(live_wav_directory, "source.wav")
historical_wav_directory = data_root + "/raw_historical"

def cleanup(): 
    if i >= n_keep: 
        last = i - n_keep
        name = str(last) 
        full_path_wav = os.path.join(historical_wav_directory, name + ".wav")
        full_path_mat = os.path.join(historical_wav_directory, name + ".mat")
        
        print("removing: ", full_path_wav, full_path_mat)
        os.remove(full_path_wav)
        os.remove(full_path_mat)

def receive(full_path): 
    print("receiving")
    
    eng.Receiver_Signal_FMCW(full_path, nargout=0)
    
    print("copying to: ", live_wav_path)
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
