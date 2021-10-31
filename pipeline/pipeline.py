import os
from datetime import datetime

from ml_siamese import OneshotLoader
from gen_datasets import DatasetLoader
from constants import gestures, input_image_types

run = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")
dataset_path = "./tmp"
run_path = os.path.join(dataset_path, run)

def main(): 
    for type in input_image_types: 
        dl = DatasetLoader(type, run)
        dl.generate()
    
    
    for spectrogram_type in input_image_types: 
        data_path = os.path.join(run_path, spectrogram_type) + "/"

        # Needed to fix some tensorflow compilation errors
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        loader = OneshotLoader(data_path, spectrogram_type)
        
        loader.train()
        loader.compare()
        return

main()
