from tensorflow.keras.optimizers import Adam

data_subsets = ["train", "val"]

param_input_image_types = [ 
    "mel", 
    "chroma", 
    "mfcc", 
    "sfft", 
    "sfftchroma",
    "spectrogram" 
]

param_default_gestures = [
    "close", 
    "cut", 
    "no", 
    "oldown", # aka voldown 
    "olup", # aka volup
    "open", 
    "switch", 
    "tap"
]

param_loss_function = "binary_crossentropy"
param_optimizer = Adam(lr = 0.00006)

param_N_way = 8 # how many classes for testing one-shot tasks>
param_n_val = 7 # how many one-shot tasks to validate on?
param_batch_size_per_trial = 7
param_n_trials = 100
param_n_iterations = 1000

param_evaluate_every = 10 # interval for evaluating on one-shot tasks
param_print_loss_every = 10 # interval for printing loss (iterations)

param_data_path = "./data/images/"
param_dataset_path = "./tmp"
param_training_percentage = 0.55 

param_wav_directory = "./data/wav"