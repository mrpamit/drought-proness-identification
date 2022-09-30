import numpy as np
import cv2
import os
import csv

def Mask(path):    
    frame = cv2.imread(path)
    frame = cv2.resize(frame,(96,96))
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    eyeMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    eyeMask = cv2.erode(eyeMask, kernel, iterations = 2)
    eyeMask = cv2.dilate(eyeMask, kernel, iterations = 2)
    lowerBoundary = np.array([170,80,30],dtype="uint8")
    upperBoundary = np.array([180,255,250],dtype="uint8")
    eyeMask2 = cv2.inRange(converted, lowerBoundary, upperBoundary)
    eyeMask = cv2.addWeighted(eyeMask,0.5,eyeMask2,0.5,0.0)
    eyeMask = cv2.medianBlur(eyeMask, 5)
    skin = cv2.bitwise_and(frame, frame, mask = eyeMask)
    frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
    skin = cv2.bitwise_and(frame, frame, mask = eyeMask)
    h,w = skin.shape[:2]
    bw_image = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)
    bw_image = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    bw_image = cv2.GaussianBlur(bw_image,(5,5),0)
    threshold = 1
    for i in range(h):
        for j in range(w):
            if bw_image[i][j] > threshold:
               bw_image[i][j] = 0
            else:
               bw_image[i][j] = 255
    return bw_image


TrainData="train_path"
for (dirpath,dirnames,filenames) in os.walk(TrainData):
    for dirname in dirnames:
        for(direcpath,direcnames,files) in os.walk(TrainData+"\\"+dirname):
            for file in files:
                path=TrainData+"\\\\"+dirname+"\\\\"+file
                bw_image=Mask(path)
                cv2.imshow('Binary Image', ~bw_image)
                cv2.waitKey(200)






