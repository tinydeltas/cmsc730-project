import sys
import numpy as np
import cv2 as cv
import pickle
import os
import matplotlib.pyplot as plt

"""Script to preprocess the omniglot dataset and pickle it into an array that's easy
    to index my character type"""

data_path = os.path.join('data/')
train_folder = os.path.join(data_path,'images/train')
valpath = os.path.join(data_path,'images/test')

save_path = 'data/images/processed'

gest_dict = {}

def loadimgs(path,n=0):
    #if data not already unzipped, unzip it.
    if not os.path.exists(path):
        print("unzipping")
        os.chdir(data_path)
        os.system("unzip {}".format(path+".zip" ))
    X = []
    y = []
    cat_dict = {}
    gest_dict = {}
    curr_y = n

    for gesture in os.listdir(path):
        print("loading gesture: " + gesture)
        gest_dict[gesture] = [curr_y, None]

        cat_dict[curr_y] = (gesture, gesture)
        category_images=[]
        
        letter_path = os.path.join(path, gesture)
        
        for filename in os.listdir(letter_path):
            image_path = os.path.join(letter_path, filename)
            image = cv.imread(image_path)
            category_images.append(image)
            y.append(curr_y)
        try:
            X.append(np.stack(category_images))
        except ValueError as e:
            print(e)
            print("error - category_images:", category_images)
        curr_y += 1
        gest_dict[gesture][1] = curr_y - 1
    
    y = np.vstack(y)
 
    X = np.stack(X)
    return X,y,gest_dict

X,y,c=loadimgs(train_folder)


with open(os.path.join(save_path,"train.pickle"), "wb") as f:
	pickle.dump((X,c),f)


X,y,c=loadimgs(valpath)
with open(os.path.join(save_path,"val.pickle"), "wb") as f:
	pickle.dump((X,c),f)
