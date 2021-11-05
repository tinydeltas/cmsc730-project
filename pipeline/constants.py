from tensorflow.keras.optimizers import Adam

data_subsets = ["train", "val"]

# The input image (spectrogram) types we are interested in feeding 
# to the ML model for training and validation. 
param_input_image_types = [ 
    "mel",              # Mel spectrogram
    "chroma",           # Chromagram
    "stftchroma",       # Chromagram with short-time fourier transform applied
    "mfcc",             # Mel cepstral coefficient transform 
    "stft",             # Short-time fourier transform 
    "spectrogram"       # Generic spectrogram
]

# Labels for the pre-defined gestures we are interested in testing 
# performance for. Corresponds to the names of their respective folders 
# in `data/images`
param_default_gestures = [
    "close", 
    "cut", 
    "no", 
    "minimize", # aka voldown / minimize
    "maximize", # aka volume up
    "open", 
    "switch", 
    "tap"
]

param_loss_function = "binary_crossentropy"
param_optimizer = Adam(lr = 0.00006)

param_N_way = 8 
param_n_val = 7
param_batch_size_per_trial = 7
param_n_trials = 100
param_n_iterations = 1000

param_evaluate_every = 10 
param_print_loss_every = 10 

param_data_path = "./data/images/"
param_dataset_path = "./tmp"
param_training_percentage = 0.55 

param_wav_directory = "./data/wav"