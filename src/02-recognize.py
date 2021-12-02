import sys
import os 
import time
import matlab.engine
import subprocess
from tensorflow import keras

import constants

from ml import siamese

def main(): 
    loader = siamese.OneshotLoader(data_path, spectrogram_type, run_path)
    
    eng = matlab.engine.start_matlab()

    print("Starting transmission")
    child = subprocess.Popen(["python3", "matlab/transmit.py"])

    print("Starting recording")
    invoke = print("matlab/receive.py %s", run_path + "/current")
    child2 = subprocess.Popen("python3", invoke)
    
    model = keras.models.load_model(constants.param_model_path)
    time.sleep(1)
    
    while True: 
        try: 
            # Get inputs from matlab
            probs = model.predict(inputs)
            print("probs", probs)
            if test_prob > constants.param_prob_cutoff: 
                print("Predicted: ", test_prob, c)
        
        except KeyboardInterrupt:
            child.terminate()
            eng.quit()
            sys.exit(0)
    