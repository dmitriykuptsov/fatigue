import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.metrics import Precision, Recall

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import cv2 as cv

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import cv2 as cv

new_model = tensorflow.keras.models.load_model("model.h5")
images = []
for i in range(0, 900):
    img=mpimg.imread("../training/frame" + str(i) + ".jpg")
    #gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    images.append(img)

#print(len(
predictions = new_model.predict(np.array(images))

for i in range(0, 900):
       print(predictions[0][i][0], predictions[0][i][1], predictions[0][i][2])
#print(predictions)
