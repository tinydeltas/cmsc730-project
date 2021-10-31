import pickle
import os
import shutil
from datetime import datetime

import numpy as np
import cv2 as cv

from constants import input_image_types, gestures

param_data_path = "./data/images/"
param_dataset_path = "./tmp"
param_training_percentage = 0.55 

class DatasetLoader: 
    def __init__(self, spectrogram_type, run): 
        data_path = os.path.join(param_data_path, spectrogram_type)
        
        self.all_folder = os.path.join(data_path, 'all')
        
        save_path = os.path.join(param_dataset_path, run, spectrogram_type)
        
        self.train_folder = os.path.join(save_path, "train")
        os.makedirs(self.train_folder)
            
        self.validation_folder = os.path.join(save_path, "validate")
        os.makedirs(self.validation_folder)
        
        self.save_folder = os.path.join(save_path, "processed")
        os.makedirs(self.save_folder)
        
        self.gest_dict = {}

        self.X = [] 
        self.Y = []
    
    def copy(self, f, g, path, folder): 
        basename = os.path.basename(f)
        from_filename = os.path.join(path, f) 
        to_filename = os.path.join(folder, g + "/" + basename)
        shutil.copy(from_filename, to_filename)
        
    def divide(self): 
        test_gesture_path = os.path.join(self.all_folder, gestures[0])
                
        n_examples = len([name for name in os.listdir(test_gesture_path)])
        n_training = int(n_examples * param_training_percentage)
        n_validate = n_examples - n_training 
        
        # print("Training examples: ", n_training)
        # print("Validation examples: ", n_validate)
        
        for g in gestures: 
            gesture_path = os.path.join(self.all_folder, g)
            all_samples = os.listdir(gesture_path)
            training_samples = np.random.choice(all_samples, n_training, replace=False) 
            validation_samples = np.setdiff1d(all_samples, training_samples)
            
            os.makedirs(os.path.join(self.train_folder, g))
            for f in training_samples: 
                self.copy(f, g, gesture_path, self.train_folder)
                

            os.makedirs(os.path.join(self.validation_folder, g))
            for f in validation_samples:
                self.copy(f, g, gesture_path, self.validation_folder)
    
    def load(self, path):
        curr_y = 0
        cat_dict = {}
        X, Y = [], [] 

        for gesture in os.listdir(path):
            self.gest_dict[gesture] = [curr_y, None]

            cat_dict[curr_y] = (gesture, gesture)
            category_images=[]
            
            letter_path = os.path.join(path, gesture)
            
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = cv.imread(image_path)
                category_images.append(image)
                Y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            
            self.gest_dict[gesture][1] = curr_y - 1
        
        self.Y = np.vstack(Y)
        self.X = np.stack(X)

    def generate(self): 
        self.divide()
        
        self.load(self.train_folder)
        with open(os.path.join(self.save_folder, "train.pickle"), "wb") as f:
            pickle.dump((self.X, self.gest_dict), f)
        
        self.load(self.validation_folder)
        with open(os.path.join(self.save_folder, "val.pickle"), "wb") as f:
            pickle.dump((self.X, self.gest_dict), f)
        
    

