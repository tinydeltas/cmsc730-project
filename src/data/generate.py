import pickle
import os
import shutil
from datetime import datetime
from librosa.core.notation import key_to_degrees

import numpy as np
import cv2 as cv

import data.params as p

class DatasetLoader: 
    def __init__(self, spectrogram_type, run): 
        data_path = os.path.join(p.processed_image_directory, spectrogram_type)
        
        self.all_folder = data_path
        
        save_path = os.path.join(p.run_directory, run, "images", spectrogram_type)
        
        self.train_folder = os.path.join(save_path, "train")
        os.makedirs(self.train_folder)
            
        self.validation_folder = os.path.join(save_path, "validate")
        os.makedirs(self.validation_folder)
        
        self.save_folder = os.path.join(save_path, "processed")
        os.makedirs(self.save_folder)
        
        self.gest_dict = {}

        self.X = [] 
        self.Y = []
    
    def copy(self, idx, from_filename, gesture, save_folder):         
        basename = idx + p.data_type
        to_filename = os.path.join(save_folder, gesture + "/" + basename)
    
        shutil.copy(from_filename, to_filename)
        return 
        
    def divide(self, gestures): 
        gesture_data_paths = {}
        min_examples = np.inf 
        
        # get n_samples as minimum # of samples n the gesture folders
        for g in gestures: 
            gesture_path = os.path.join(self.all_folder, g)
            gesture_data_paths[g] = []
            for date in os.listdir(gesture_path): 
                path = os.path.join(gesture_path, date)
                files = [ os.path.join(path, name) for name in os.listdir(path) if name.endswith(p.data_type)]
                if len(files) < min_examples: 
                    min_examples = len(files) 
                if len(files) > 0: 
                    gesture_data_paths[g] += files
        
        n_examples = min_examples 
        n_training = int(n_examples * p.training_percentage)
        n_validation = n_examples - n_training 
        
        for gesture_label in gesture_data_paths:
            all_samples = gesture_data_paths[gesture_label]
                
            training_samples = np.random.choice(all_samples, n_training, replace=False) 
            leftover_samples = np.setdiff1d(all_samples, training_samples)
            validation_samples = np.random.choice(leftover_samples, n_validation, replace=False)
            
            os.makedirs(os.path.join(self.train_folder, gesture_label))
            for idx in range(len(training_samples)): 
                from_path = training_samples[idx]
                self.copy(str(idx), from_path, gesture_label, self.train_folder)
            
            os.makedirs(os.path.join(self.validation_folder, gesture_label))
            for idx in range(len(validation_samples)):
                from_path = validation_samples[idx]
                self.copy(str(idx), from_path, gesture_label, self.validation_folder)

    
    def load(self, path):
        curr_y = 0
        cat_dict = {}
        X, Y = [], [] 
        key = [] 

        for gesture in os.listdir(path):
            key.append(gesture)
            self.gest_dict[gesture] = [curr_y, None]

            cat_dict[curr_y] = (gesture, gesture)
            category_images=[]
            
            letter_path = os.path.join(path, gesture)
            
            for filename in os.listdir(letter_path):
                data_path = os.path.join(letter_path, filename)
                if p.using_image: 
                    data = cv.imread(data_path)
                else: 
                    data = np.load(data_path)
                category_images.append(data)
                Y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            except ValueError as e:
                print(e)
                # print("error - category_images:", category_images)
            curr_y += 1
            
            self.gest_dict[gesture][1] = curr_y - 1
        
        self.Y = np.vstack(Y)
        self.X = np.stack(X)
        return key


    def generate(self, gestures): 
        self.divide(gestures)
        
        train_key = self.load(self.train_folder)
        with open(os.path.join(self.save_folder, "train.pickle"), "wb") as f:
            pickle.dump((self.X, self.gest_dict), f)
        
        val_key = self.load(self.validation_folder)
        with open(os.path.join(self.save_folder, "val.pickle"), "wb") as f:
            pickle.dump((self.X, self.gest_dict), f)
        
        with open(os.path.join(self.save_folder, "train.key"), "wb") as f: 
            pickle.dump(train_key, f)
        
        with open(os.path.join(self.save_folder, "val.key"), "wb") as f: 
            pickle.dump(val_key, f)
        

        
    

