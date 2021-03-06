# The input image (spectrogram) types we are interested in feeding 
# to the ML model for training and validation. 
input_image_types = [ 
    "mel",              # Mel spectrogram
    "stft",             # Short-time fourier transform 
    "spectrogram"       # Generic spectrogram
    
    # The following have been retired due to poor performance
    # "chroma",           # Chromagram
    # "stftchroma",       # Chromagram with short-time fourier transform applied
    # "mfcc",             # Mel cepstral coefficient transform 

]

# Labels for the pre-defined gestures we are interested in testing 
# performance for. Corresponds to the names of their respective folders 
# in `data/images`
default_gestures = [
    "cut", 
    "paste", 
    
    "minimize", 
    "maximize", 
    
    # "open", 
    "close", 
    
    "no",
    "switch", 

    # These are new in the second round
    # "zoom", 
    # "quit", 
    # "erase",
    
    # "pause", 
    # "stop", 
    # "rewind",
    
    # "volume_up",
    # "volume_down",

    # "like", 
    # "dislike" 
]

source_wav_directory = "./data/raw/audio"

# Used for spectrogram generation 
fs = 48000
nfft = int(0.1 * fs)
noverlap = int(0.095 * fs)

# Used for prediction 

data_root = "./static/data/"

model_path = data_root + "model"

live_wav_path = data_root + "source.wav"
live_wav_directory = data_root + "/raw"
historical_wav_directory = data_root + "/raw_historical"

live_spectrogram_path = data_root + "s.png"
live_gesture_prediction = data_root + "gesture.txt"
live_predictions_path = data_root = "predictions.txt"

average_window_ms = 500
clip_duration_ms = 1000
clip_stride_ms = 1000
suppression_ms = 500
detection_threshold = 0.5

# Deprecated 
model = "siamese"

x = 800
y = 581
z = 1


predict_spectrogram_type = "mel"
predict_run_path = "../../tmp/best_1000"

n_prediction_per = 10


minimum_count = 4
