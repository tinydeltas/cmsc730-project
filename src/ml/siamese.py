import os
import time

import numpy as np
import cv2 as cv

import numpy.random as rng
from keras import backend as K
from keras.layers import Conv2D, Conv1D, Input, Dropout
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import *
from keras.regularizers import l2
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam

import data.params as dpm 
K.set_image_data_format('channels_last')

import params

from ml.generic import Loader

loss_function = "binary_crossentropy"
optimizer = Adam(lr = 0.00006)
n_iterations = 1000

param_N_way = len(params.default_gestures) 
param_n_val = len(params.default_gestures) - 1
param_batch_size_per_trial = len(params.default_gestures) - 1
param_n_trials = 100

evaluate_every = 10 
print_loss_every = 10 

# Class for Siamese network implementation
class OneshotLoader(Loader): 
    def __init__(self, path, spectrogram_type, run_path):
        super().__init__(path, spectrogram_type, run_path)

    def initialize_bias(self, shape, dtype=None):
        """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
        """
        return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)
    
    def initialize_weights(self, shape, dtype=None):
        """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
        """
        return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)
    
    def get_model(self, in_shape): 
        CNNmodel = Sequential()
        CNNmodel.add(Conv2D(32, (3, 3), activation='relu', input_shape=in_shape))
        CNNmodel.add(MaxPooling2D((2, 2)))
        CNNmodel.add(Dropout(0.2))
        CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
        CNNmodel.add(MaxPooling2D((2, 2)))
        CNNmodel.add(Dropout(0.2))
        CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
        CNNmodel.add(Flatten())
        CNNmodel.add(Dense(64, activation='relu'))
        CNNmodel.add(Dropout(0.2))
        CNNmodel.add(Dense(32, activation='relu'))
        CNNmodel.add(Dense(24, activation='softmax'))
        return CNNmodel

    def get_siamese_model(self, in_shape): 
        print("shape: ", in_shape)
        # Needed to fix some tensorflow compilation errors
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        left = Input(in_shape)
        right = Input(in_shape)
        
        model = Sequential()
        
        print("adding layers")
        # Ported
        model.add(Conv2D(64, 
                        (10,10), 
                        activation='relu', 
                        input_shape=in_shape,
                        kernel_initializer=self.initialize_weights, 
                        kernel_regularizer=l2(2e-4)))
        
        # Ported 
        model.add(MaxPooling2D())
        
        # Ported
        model.add(Conv2D(128, 
                        (7,7),
                        activation='relu',
                        kernel_initializer=self.initialize_weights,
                        bias_initializer=self.initialize_bias,
                        kernel_regularizer=l2(2e-4)))
        
        # Ported
        model.add(MaxPooling2D())
        
        # Ported 
        model.add(Conv2D(128,
                        (4,4),
                        activation='relu',
                        kernel_initializer=self.initialize_weights,
                        bias_initializer=self.initialize_bias,
                        kernel_regularizer=l2(2e-4)))
        
        # Ported 
        model.add(MaxPooling2D())
        
        # Ported
        model.add(Conv2D(1, 
                        (4,4),
                        activation='relu',
                        kernel_initializer=self.initialize_weights,
                        bias_initializer=self.initialize_bias,
                        kernel_regularizer=l2(2e-4)))
        
        # Ported
        model.add(Flatten())
        
        # Ported
        model.add(Dense(4096,
                        activation='sigmoid', 
                        kernel_regularizer=l2(1e-3), 
                        kernel_initializer=self.initialize_weights,
                        bias_initializer=self.initialize_bias))
        
        print("Finished adding layers")
        # Ported
        encoded_l = model(left)
        encoded_r = model(right)

        # Ported
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        
        # Ported
        prediction = Dense(1,
                        activation='sigmoid',
                        bias_initializer=self.initialize_bias)(L1_distance)
        
        net = Model(inputs=[left, right], outputs=prediction)
        return net 

    def get_batch(self, batch_size, s="train"):
        """
        Create batch of n pairs, half same class, half different class
        """
        X = self.data[s]
        # print("Data: ", X)
        
        n_classes, n_examples, w, h, z = X.shape
        print("get_batch: ", n_classes, n_examples, w, h, z)
        # Randomly sample several classes to use in the batch
        categories = rng.choice(n_classes, 
                                size=(batch_size,),
                                replace=False)
        
        # Initialize 2 empty arrays for the input image batch
        pairs =[ np.zeros((batch_size, w, h, z)) for i in range(2) ]
        
        # Initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets = np.zeros((batch_size,))
        targets[ batch_size//2: ] = 1
        
        for i in range(batch_size):
            category = categories[i]
            
            idx_1 = rng.randint(0, n_examples)
            
            pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, z)
            idx_2 = rng.randint(0, n_examples)
            
            # Pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                category_2 = category  
            else: 
                # Add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                category_2 = (category + rng.randint(1,n_classes)) % n_classes
            
            pairs[1][i,:,:,:] = X[category_2, idx_2].reshape(w, h, z)
        return pairs, targets
    
    def generate(self, batch_size, s="train"):
        """
        A generator for batches, so model.fit_generator can be used.
        """
        
        while True:
            pairs, targets = self.get_batch(batch_size, s)
            yield (pairs, targets)    

    def make_task(self, N, s="val", test_image_path=None):
        """
        Create pairs of test image, support set for testing N way one-shot learning. 
        """
        X = self.data[s]
        n_classes, n_examples, w, h, _ = X.shape
        print(n_classes, n_examples, w, h)
        indices = rng.randint(0, n_examples, size=(N,))
        
        categories = rng.choice(range(n_classes), size=(N,), replace=False)            
        true_category = categories[0]
        ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))

        test_image = np.asarray([ X[true_category,ex1,:,:] ] * N).reshape(N, w, h, params.z)
        
        if test_image_path is not None: 
            # Load in the image here 

            if dpm.using_image:
                img_raw = cv.imread(test_image_path)
                img_arr = np.asarray(img_raw).reshape(w, h, params.z) 
            else: 
                img_arr = np.load(test_image_path).reshape(w, h, params.z)
                test_image = np.asarray([img_arr] * N).reshape(N, w, h, params.z)
        
        support_set = X[categories,indices,:,:]
        support_set[0,:,:] = X[true_category,ex2]
        support_set = support_set.reshape(N, w, h, params.z)
        
        targets = np.zeros((N,))
        targets[0] = 1

        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image, support_set]
        
        return pairs, targets
    
    
    def test_live(self, model, N, k, test_image_path, s="val"): 
        all_probs = []
        for i in range(k): 
            inputs, _ = self.make_task(N, s, test_image_path)
            probs = model.predict(inputs)
            all_probs.append(probs)
            
        final_probs = np.mean(np.stack(all_probs), axis=0)
        return final_probs 
        
        
    def test(self, model, N, k, s="val", verbose=False):
        """
        Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks
        """
        n_correct = 0
        
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
        
        for i in range(k):
            inputs, targets = self.make_task(N, s, None)
    
            probs = model.predict(inputs)
            
            # print("N", N)
            # print("Inputs: ", len(inputs))
            # print("Inputs[0]", inputs[0].shape)
            # print("Inputs[1]", inputs[1].shape)
            # print("Probs: ", probs)
            # print("Targets: " , targets)

            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
        
        percent_correct = (100.0 * n_correct / k)
        
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct, N))
        
        return percent_correct
    
    def train(self): 
        best = -1 
        
        print("getting model")
        model = self.get_model((params.x, params.y, 1))
        print("compiling model")
        model.compile(loss=loss_function,
                    optimizer=optimizer,
                    metrics = ['accuracy'])
        model.summary()
    
        
        print("Starting training process!")
        print("-------------------------------------")
        t_start = time.time()

        for i in range(1, n_iterations):
            (inputs, targets) = self.get_batch(param_batch_size_per_trial)
            loss = model.train_on_batch(inputs, targets)
            # print("\n ------------- \n")
            # print("Loss: {0}".format(loss)) 
            
            if i % evaluate_every == 0:
                print("Time for {0} iterations: {1}".format(i, time.time()-t_start))
                val_acc = self.test(model, param_N_way, param_n_val, verbose=True)
                
                if val_acc >= best:
                    print("Current best: {0}, previous best: {1}".format(val_acc, best))
                    print("Saving weights to: {0} \n".format(self.weights_path))
                    model.save_weights(self.weights_path)
                    best = val_acc
            
            if i % print_loss_every == 0:
                print("iteration", i)
                print("training loss: ", loss)

        return model
    