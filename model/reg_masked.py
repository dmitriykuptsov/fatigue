import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Reshape
from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K
from keras.models import load_model
import numpy as np
import sys
import os
import re
import cv2 as cv
import math
import argparse

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import matplotlib.pyplot as plt
import matplotlib.patches as patches


x_train = []
y_train = []

nb_boxes=1
grid_w=4
grid_h=4
cell_w=61
cell_h=61
grid_size = 61
img_w=grid_w*cell_w
img_h=grid_h*cell_h
input_size = 244
num_classes = 2

#
# Load all images and append to vector

def prepare_input_output(img, box, c):
    x, y, w, h = box[0], box[1], box[2], box[3]
    #print(x, y, w, h)
    ih, iw, chan = img.shape
    x = x * iw
    y = y * ih
    w = w * iw
    h = h * ih
    max_size = max(iw, ih)
    r = max_size / input_size
    new_height = int(ih / r)
    new_width = int(iw / r)
    new_size = (new_width, new_height)
    resized = cv.resize(img.astype(np.uint8), new_size, interpolation= cv.INTER_LINEAR)
    new_image = np.zeros((input_size, input_size, chan), dtype=np.uint8)
    new_image[0:new_height, 0:new_width, 0:chan] = resized
    new_box = [(x + 0.5*w) / r, (y + 0.5*h) / r, (w / r), (h / r)]
    #new_box = [(x) / r, (y) / r, (w / r), (h / r)]
    ca = np.array([0] * num_classes)
    ca[c] = 1
    x = (x + 0.5*w) / r
    y = (y + 0.5*h) / r

    i = math.floor(x / grid_size)
    j = math.floor(y / grid_size)
    
    new_box[0] = new_box[0] / input_size
    new_box[1] = new_box[1] / input_size
    new_box[2] = new_box[2] / input_size
    new_box[3] = new_box[3] / input_size
    
    output = np.float32(np.array([0.0] * (grid_w*grid_h*(nb_boxes*5 + num_classes))))
    output = np.reshape(output, (grid_w, grid_h, (nb_boxes*5 + num_classes)))
    new_box = np.array(new_box)
    #print(new_box)
    
    output[i][j] = np.float32(np.concatenate([ca, new_box, np.array([1])]))
    return new_image, output, new_box

images = []
outputs1 = []
outputs2 = []
# Read labels

classes = ["Masked", "Non masked"]

all_entries = os.listdir("../masked/")

files = [entry for entry in all_entries if not os.path.isdir(entry)]

labels = []
for file in files:
    if re.match("^.*\.txt$", file):
        labels.append(file)

counter = 1
for file in labels:
    with open("../masked/" + file) as f:
        label = f.readlines()[0]
    parts = label.split(" ")
    image_file = file.split(".")[0]
    img=cv.imread("../masked/" + image_file + ".jpg")
    img_orig = cv.imread("../masked/" + image_file + ".jpg")
    ih, iw, chan = img.shape
    box = [float(parts[1]) - float(parts[3]) / 2, float(parts[2]) - float(parts[4]) / 2, float(parts[3]), float(parts[4])]
    box_orig = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
    #x, y, w, h = box[0], box[1], box[2], box[3]
    #box = [x-w/2, y-h/2, w, h]
    clss = int(parts[0])
    img, output, box = prepare_input_output(img, box, clss)

    output = box
    
    images.append(np.array(img))
    outputs2.append(box)
    outputs1.append(clss)
    
    if counter < 0:
        ax = plt.subplot()
        box_orig = box
        img_orig = img
        print(box_orig)
        ih, iw, chan = img_orig.shape
        print(ih, iw)
        imgplot = plt.imshow(img_orig)
        color = ['g','r','b','0'][clss]
        x, y, w, h = box_orig[0], box_orig[1], box_orig[2], box_orig[3]
        x = x * iw - w * iw / 2
        y = y * ih - h * ih / 2
        w = w * iw
        h = h * ih
        rect = patches.Rectangle((x, y), w, h, color=color, fill=False)
        plt.text(20, 20, classes[clss], color='white', fontweight='bold', bbox=dict(fill=False, edgecolor='red', linewidth=2))
        ax.add_patch(rect)
        plt.show()
    #else:
    #    break
    counter += 1

w = 244
h = 244
# MODEL PARAMETERS 
INPUT_SHAPE = (h, w, 3, )
FILTER1_SIZE = 16
FILTER2_SIZE = 32
FILTER3_SIZE = 64
FILTER_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
FULLY_CONNECT_NUM = 64
NUM_CLASSES = 1
BBOX_SIZE = 4
OUTPUT_LAYER = BBOX_SIZE
OUTPUT_LAYER2 = NUM_CLASSES

def build_feature_extractor(inputs):
    # Feature extraction layer
    x = tf.keras.layers.Conv2D(FILTER1_SIZE, FILTER_SHAPE, activation='relu', input_shape=INPUT_SHAPE)(inputs)
    x = tf.keras.layers.AveragePooling2D(POOL_SHAPE)(x)

    x = tf.keras.layers.Conv2D(FILTER2_SIZE, FILTER_SHAPE, activation = 'relu')(x)
    x = tf.keras.layers.AveragePooling2D(POOL_SHAPE)(x)

    x = tf.keras.layers.Conv2D(FILTER3_SIZE, FILTER_SHAPE, activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling2D(POOL_SHAPE)(x)

    x = tf.keras.layers.Conv2D(FILTER3_SIZE, FILTER_SHAPE, activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling2D(POOL_SHAPE)(x)

    return x

def build_model_adaptor(inputs):
    # Fully connected layer
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(FULLY_CONNECT_NUM, activation='relu')(x)
    return x

def build_classifier_head(inputs):
    # Classifier output
    return tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid', name = 'classifier_head')(inputs)

def build_regressor_head(inputs):
    # Regressor output
    return tf.keras.layers.Dense(BBOX_SIZE, activation='sigmoid', name = 'regressor_head')(inputs)

def build_model(inputs):
    # Compose the model
    feature_extractor = build_feature_extractor(inputs)
    model_adaptor = build_model_adaptor(feature_extractor)
    classification_head = build_classifier_head(model_adaptor)
    regressor_head = build_regressor_head(model_adaptor)
    model = tf.keras.Model(inputs = inputs, outputs = [classification_head, regressor_head])
    return model

# Build the model
model = build_model(tf.keras.layers.Input(shape=INPUT_SHAPE))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), 
    loss = {'classifier_head' : 'binary_crossentropy', 'regressor_head' : 'mse' }, 
    metrics = {'classifier_head' : 'accuracy', 'regressor_head' : 'mse' })

# Print the summary
model.summary()

#
# Training the network
#
parser = argparse.ArgumentParser(description='YOLO')
parser.add_argument('--train', help='train', action='store_true')
parser.add_argument('--epoch', help='epoch', const='int', nargs='?', default=1)
args = parser.parse_args()

images = np.array(images)
outputs1 = np.array(outputs1)
outputs2 = np.array(outputs2)

if args.train:
    model.fit(images, (outputs1, outputs2), batch_size=32, epochs=int(args.epoch))
    model.save_weights('masked_reg.h5')
    exit()
else:
    model.load_weights('masked_reg.h5')

#
# Predict bounding box and classes for the first 25 images
#
def prepare_image(img):
    ih, iw, chan = img.shape
    max_size = max(iw, ih)
    r = max_size / input_size
    new_height = int(ih / r)
    new_width = int(iw / r)
    new_size = (new_width, new_height)
    resized = cv.resize(img.astype(np.uint8), new_size, interpolation= cv.INTER_LINEAR)
    new_image = np.zeros((input_size, input_size, chan), dtype=np.uint8)
    new_image[0:new_height, 0:new_width, 0:chan] = resized
    return new_image

all_entries = os.listdir("../validate/")

files = [entry for entry in all_entries if not os.path.isdir(entry)]

images = []
for file in files:
    if re.match("^.*\.jpg$", file):
        images.append(file)

for file in images:
    print(f"Doing image # {file}")
    img=prepare_image(cv.imread("../validate/" + file))
    predictions = model.predict(np.array([img]))
    c = "MASK"
    if predictions[0][0] > 0.5:
        c = "NO MASK"
    box = predictions[1][0]
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img)
    w = 244
    h = 244
    print(box[0])
    print(box[1])
    print(box[2])
    print(box[3])
    startX = int(box[0] * w)
    startY = int(box[1] * h)
    endX = int(box[2] * w)
    endY = int(box[3] * h)
    print(startX)
    print(startY)
    rect = patches.Rectangle((startX, startY), endX, endY, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.text(20, 100, c, color='red', fontweight='bold', bbox=dict(fill=False, edgecolor='red', linewidth=2))
    plt.show()
