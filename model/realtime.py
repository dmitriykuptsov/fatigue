import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.metrics import Precision, Recall

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np

import threading 
from threading import Lock

from PIL import Image
import time
import cv2 as cv

import cv2
vidcap = cv2.VideoCapture("rtsp://admin:admin@192.168.1.21:554/media/video1")


new_model = tensorflow.keras.models.load_model("model.h5")


FPS = 25
CALIBRATION = 1.5

def skipFrames(timegap):
   global FPS,cap
   latest = None
   while True :  
      for i in range(int(timegap*FPS/CALIBRATION)) :
        _,latest = vidcap.read()
        if(not _):
           time.sleep(0.5)#refreshing time
           break
      else:
        break
   return latest

gap = 0.1

while vidcap.isOpened(): 
    image = skipFrames(gap)
    s = time.time()

    
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    predictions = new_model.predict(np.array([gray_img]))
    print(predictions)    
    #print(predictions[0][i][0], predictions[0][i][1], predictions[0][i][2])
    if predictions[0][0][0] > predictions[0][0][1] and predictions[0][0][0] > predictions[0][0][2]:
        c = "Regular"
    if predictions[0][0][1] > predictions[0][0][0] and predictions[0][0][1] > predictions[0][0][2]:
        c = "Distraction"
    if predictions[0][0][2] > predictions[0][0][0] and predictions[0][0][2] > predictions[0][0][1]:
        c = "Fatigue"
    box = predictions[1][0]
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(image)
    rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.text(20, 100, c, color='white', fontweight='bold', bbox=dict(fill=False, edgecolor='red', linewidth=2))
    plt.show()
    gap = time.time() - s
    
#print(predictions)
