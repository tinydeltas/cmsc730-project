import os 

import params 
from spectrograms import SpectrogramLoader

def main(): 
    # Takes about 5 minutes 
    for gesture_type in params.default_gestures:
        print("Generating spectrograms for gesture: ", gesture_type)
        gesture_path = os.path.join(params.source_wav_directory, gesture_type)
        for date_directory in os.scandir(gesture_path): 
            print("Date directory: ", os.path.basename(date_directory))
            if os.path.basename(date_directory)== ".DS_Store": 
                continue
            for file in os.scandir(date_directory): 
                if (file.path.endswith(".wav") and file.is_file()):
                    
                    loader = SpectrogramLoader(gesture_type, 
                                            os.path.basename(date_directory), 
                                            file.path)
                    print("generating: ", file.path)
                    loader.get_all_spectrograms()

main()