#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:35:13 2020

@author: ordovas
"""

import numpy as np
# Image related libraris
import cv2
import matplotlib.pyplot as plt

# Neural net related libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import model_from_json






#Select different dices

json_file = open('../models/model_difdices_final.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("../models/model_difdices_final.h5")

# Predict dices score

json_file = open('../models/model_1d6_zoom.json', 'r')
loaded_model_json = json_file.read()
loaded_model_blue = model_from_json(loaded_model_json)

loaded_model_blue.load_weights("../models/model_1d6_zoom.h5")

json_file = open('../models/model_2d6_zoom.json', 'r')
loaded_model_json = json_file.read()
loaded_model_red = model_from_json(loaded_model_json)

loaded_model_red.load_weights("../models/model_2d6_zoom.h5")

d6_labels=[5, 1, 4, 3, 6, 2]
d10_labels=[8, 5, 1, 7, 4, 0, 3, 6, 2, 9]
d20_labels=[3, 12, 14, 7, 13, 20, 17, 4, 8, 5, 15, 1, 18, 11, 16, 10, 9, 6, 19, 2]


#This function is to detect the dice type
def pred_dice_type(die,model=loaded_model):
    resized= cv2.resize(die, (50,50), interpolation = cv2.INTER_AREA) 
    dice_type=model.predict_classes( np.array( [resized,] ))  
    return dice_type

#This function is to obtain the score of a D6 dice
def get_score(die,model):
    # 1d6 dice labels, in the order that TF ordered it
    labels=[5, 1, 4, 3, 6, 2]
    resized= cv2.resize(die, (100,100), interpolation = cv2.INTER_AREA)
    print(model.predict( np.array( [resized,] )))
    score = labels[model.predict_classes( np.array( [resized,] )  )[0]]
    return score

#This function is to obtain the score of a D6 dice
def predict_dice_score(die,model=loaded_model
                       ,model_red=loaded_model_red
                       ,model_blue=loaded_model_blue):
    dt=pred_dice_type(die,model)
    if dt==0:
        color="red"
        score=get_score(die,model_red)
    else:
        color="blue"
        score=get_score(die,model_blue)
    return score,color