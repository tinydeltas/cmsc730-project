import os
import sys
import time

import matlab.engine
import params

n_keep = 100
i = 0 

directory = params.live_save_directory
eng = matlab.engine.start_matlab()

if len(sys.argv) > 1: 
    directory = sys.argv[1]

if not os.path.exists(directory): 
    os.makedirs(directory)
    
def cleanup(): 
    if i >= n_keep: 
        last = i - n_keep
        full_name = params.live_wav_filename + "_" + str(last) 
        full_path_wav = os.path.join(directory, full_name + ".wav")
        full_path_mat = os.path.join(directory, full_name + ".mat")
        os.remove(full_path_wav)
        os.remove(full_path_mat)
            
def receive(full_path): 
    print("receiving")
    eng.Receiver_Signal_FMCW(full_path, nargout=0)

while True: 
    full_name = params.live_wav_filename + "_" + str(i)
    full_path = os.path.join(directory, full_name)
    receive(full_path)
    cleanup()
    i += 1
    # time.sleep(1)
    
    
