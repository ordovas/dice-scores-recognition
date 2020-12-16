#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:59:34 2020

@author: ordovas
"""


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

from detect_functions.score_pred_func import *
from detect_functions.dices_detect_func import *


# Plots the original image and a the one with the dice detections in green
def plot_detection(img0,img0_detected):
    img=img0.copy()
    img_detected=img0_detected.copy()
    
    fig = plt.figure(figsize=(6,12))
    if img0.shape[0] < img0.shape[1]:
        img=  cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_detected=  cv2.rotate(img_detected, cv2.ROTATE_90_CLOCKWISE)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(img_detected)
    plt.title("Dice detection")
    plt.show()
    


# Plots all D6 dices with their score predicted by the CNN 
def plot_dices_d6(dices):
    i=0
    a=len(dices)//2 if len(dices)%2==0 else len(dices)//2+1
    b=len(dices)//2
    fig = plt.figure(figsize=(6,6*len(dices)//2))
    for dice in dices:
        i+=1
        score, dice_color, dice_type = predict_dice_score(dice)
        plt.subplot(a, max(b,2), i)
        plt.imshow(dice)
        plt.title(f"{dice_color.capitalize()} {dice_type} score = {score}")

    plt.show()

 # Plots all D10 dices with their score predicted by the CNN    
def plot_dices_d10(dices):
    i=0
    fig = plt.figure(figsize=(6*len(dices)//2,6*len(dices)//2))
    a=len(dices)//2 if len(dices)%2==0 else len(dices)//2+1
    b=len(dices)//2
    for dice in dices:
        i+=1
        score, dice_color, dice_type = predict_dice10_score(dice)
        plt.subplot(b, a, i)
        plt.imshow(dice)
        plt.title(f"{dice_color.capitalize()} {dice_type} score = {score}")

    plt.show()
