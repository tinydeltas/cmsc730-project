# The input image (spectrogram) types we are interested in feeding 
# to the ML model for training and validation. 
input_image_types = [ 
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
gestures = [
    "close", 
    "cut", 
    "no", 
    "oldown", # aka voldown 
    "olup", # aka volup
    "open", 
    "switch", 
    "tap"
]
