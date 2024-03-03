import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.metrics import Precision, Recall

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np

from PIL import Image

import cv2 as cv

new_model = tensorflow.keras.models.load_model("model.h5")
images = []

for i in range(0, 20):
    for j in range(0, 5):
        img=mpimg.imread(f"../aug_data/images/frame{i}_{j}.jpg")
        #img=mpimg.imread(f"../training/frame{i}.jpg")
        #img = np.array(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        images = np.array([img])
        predictions = new_model.predict(np.array(images))
        print("Predictions::::::")
        print(predictions)
        if not predictions:
            continue
        print(predictions[0][0])
        c = "No face"
        if predictions[0][0] > 0.5:
            c = "Face"
        box = predictions[1][0]
        # Create figure and axes
        fig, ax = plt.subplots()
        # Display the image
        ax.imshow(images[0])
        w = 1280
        h = 720
        print(box[0])
        print(box[1])
        print(box[2])
        print(box[3])
        startX = int(box[0] * w)
        startY = int(box[1] * h)
        endX = int(box[2] * w) - startX
        endY = int(box[3] * h) - startY
        print(startX)
        print(startY)
        rect = patches.Rectangle((startX, startY), endX, endY, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(20, 100, c, color='white', fontweight='bold', bbox=dict(fill=False, edgecolor='red', linewidth=2))
        plt.show()
