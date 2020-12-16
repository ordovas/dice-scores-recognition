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
from PIL import Image





# Loads an image
def load_dice_img(lnk):
    im = Image.open(lnk)
    im = np.asarray(im)
    return im


# Returns two arguments:
# 1) list of all regions
# 2) images with
def detect_dices(img):
    detect_dices = cv2.CascadeClassifier('cascade.xml')
    dices = detect_dices.detectMultiScale(img,scaleFactor=1.1,
                                              minNeighbors=5,
                                              flags=cv2.CASCADE_SCALE_IMAGE)
    
    #print(f"HaarCascadeClassifier:--- Detected {len(dices)} dices")
    img_detected=img.copy()
    for (x, y, w, h) in dices:
        # Draw rectangle around dices in green
        cv2.rectangle(img_detected, (x, y), (x+w, y+h), (0, 255, 0), 5)
    return dices,img_detected


# Returns the image of a certain dice
def obtain_dice(image,region):
    x, y, w, h = region
    die = image[y:y+h,x:x+w,:]
    # Zoom in
    die2=die[int(0.2*die.shape[0]):int(0.8*die.shape[0]),
                 int(0.2*die.shape[1]):int(0.8*die.shape[1])]
     
    return die2

# Returns a list whose elements are the croped images with each dice
def segment_dices(image,regions):
    slices=[]
    
    for (x, y, w, h) in regions:
        die = image[y:y+h,x:x+w,:]
        # Zoom in
        die2=die[int(0.2*die.shape[0]):int(0.8*die.shape[0]),
                 int(0.2*die.shape[1]):int(0.8*die.shape[1])]
        slices.append(die2)
    return slices


