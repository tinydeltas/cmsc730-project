import os 
import pickle 

from constants import *

# Generic loader class that sets up some basic filepath stuff
class Loader: 
    def __init__(self, path, spectrogram_type, run_path):
        self.data = {}
        self.categories = {}
        self.info = {}
        
        self.spectrogram_type = spectrogram_type 
        
        weights_folder = os.path.join(run_path, "models")
        if not os.path.exists(weights_folder): 
            os.makedirs(weights_folder)
        self.weights_path = os.path.join(weights_folder, spectrogram_type + "_model_weights.h5")
    
        self.results_folder = os.path.join(run_path, "results/")
        if not os.path.exists(self.results_folder): 
            os.makedirs(self.results_folder)    
        
        for name in data_subsets:
            file_path = os.path.join(path + "processed", name + ".pickle")
            print("loading data from {}".format(file_path))
            
            with open(file_path, "rb") as f:
                (X, c) = pickle.load(f)
                self.data[name] = X
                self.categories[name] = c