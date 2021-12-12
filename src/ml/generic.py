import os 
import pickle 

data_subsets = ["train", "val"]

# Generic loader class that sets up some basic filepath stuff
class Loader: 
    def __init__(self, path, spectrogram_type, run_path):
        self.data = {}
        self.keys = {}
        self.categories = {}
        self.info = {}
        
        self.spectrogram_type = spectrogram_type 
        
        model_folder_base = os.path.join(run_path, "models")
        if not os.path.exists(model_folder_base): 
            os.makedirs(model_folder_base)
        
        model_folder = os.path.join(model_folder_base, spectrogram_type) 
        if not os.path.exists(model_folder): 
            os.makedirs(model_folder)
        
        self.model_path = os.path.join(model_folder, "model")
        self.weights_path = os.path.join(model_folder, "weights")
    
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
            
            key_path = os.path.join(path + "processed", name + ".key")
            with open(key_path, "rb") as f: 
                key = pickle.load(f)
                self.keys[name] = key 
            