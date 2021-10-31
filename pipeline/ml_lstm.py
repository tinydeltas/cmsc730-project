
import os 
import time
import pickle
import numpy as np
import matplotlib as plt
import numpy.random as rng

from sklearn.utils import shuffle

from tensorflow.keras.optimizers import Adam

from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.optimizers import *
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.regularizers import l2

from ml_generic import Loader

param_loss_function = "binary_crossentropy"
param_optimizer = Adam(lr = 0.00006)

evaluate_every = 10 # interval for evaluating on one-shot tasks
loss_every = 10 # interval for printing loss (iterations)
batch_size = 2
n_iterations = 500
N_way = 8 # how many classes for testing one-shot tasks>
n_val = 7 # how many one-shot tasks to validate on?

class LSTMLoader(Loader): 
    def __init__(self, path, spectrogram_type, run_path): 
        super().__init__(self, path, spectrogram_type, run_path)
        
 