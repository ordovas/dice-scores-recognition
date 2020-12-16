#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:23:56 2020

@author: ordovas
"""

import numpy as np
import os

import cv2

from detect_functions.score_pred_func import *
from detect_functions.dices_detect_func import *

def webcam_dice_detector():
    # Loading the webcam
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    # Loading the Cascade Classifier
    classif = cv2.CascadeClassifier('cascade.xml')
    
    
    
    while True:
        # Obtain frame from the webcam
        ret,frame = vc.read()
        # Detect dices in the screen
        dices = classif.detectMultiScale(frame,scaleFactor=1.1,
                                                  minNeighbors=5,
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
        total_score = 0
        # For each detected dic
        for (x,y,w,h) in dices:
            # Draw the rectangle of the dice region
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            # Get the predicted score
            score_dice=predict_dice_score(obtain_dice(frame,[x,y,w,h]))
            # Put the score as text near the box
            cv2.putText(frame,f'{score_dice[0]}',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
            # Add the score to the total
            total_score+=score_dice[0]
            
        # Print the sum of all the dices 
        cv2.putText(frame,f'Total score = {total_score} points',(10,30),2,0.7,(0,255,0),2,cv2.LINE_AA)
        # Shows the video frame
        cv2.imshow('frame',frame)
        
        # Finish when the "Q" key is pressed
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    vc.release()
    cv2.destroyAllWindows()



