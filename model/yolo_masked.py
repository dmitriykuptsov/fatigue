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
    print(x, y, w, h)
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
outputs = []
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
    
    images.append(np.array(img))
    outputs.append(output)
    
    if counter < 20:
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


#
# Define the deep learning network
#
# model 2
i = Input(shape=(img_h,img_w,3))
x = Conv2D(16, (3, 3))(i)
x = Conv2D(32, (3, 3))(x)
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(16, (3, 3))(x)
x = Conv2D(32, (3, 3))(x)
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(16, (3, 3))(x)
x = Conv2D(32, (3, 3))(x)
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='sigmoid')(x)
x = Dense(grid_w*grid_h*(num_classes + nb_boxes*5), activation='sigmoid')(x)
x = Reshape((grid_w, grid_h,(num_classes+nb_boxes*5)))(x)

model = Model(i, x)

def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max

def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores

classes = ["Mask", "No mask"]
#
# The loss function orient the backpropagation algorithm toward the best direction.
#It does so by outputting a number. The larger the number, the further we are from a correct solution.
#Keras also accept that we output a tensor. In that case it will just sum all the numbers to get a single number.
# 
# y_true is training data
# y_pred is value predicted by the network
def custom_loss(y_true, y_pred):
    # first three values are classes : cat, rat, and none.
    # However yolo doesn't predict none as a class, none is everything else and is just not predicted
    # so I don't use it in the loss
    y_true_class = y_true[...,:2]
    y_pred_class = y_pred[...,:2]

    # reshape array as a list of grid / grid cells / boxes / of 5 elements
    pred_boxes = K.reshape(y_pred[...,2:], (-1,grid_w, grid_h,nb_boxes,5))
    true_boxes = K.reshape(y_true[...,2:], (-1,grid_w, grid_h,nb_boxes,5))
    
    # sum coordinates of center of boxes with cell offsets.
    # as pred boxes are limited to 0 to 1 range, pred x,y + offset is limited to predicting elements inside a cell
    y_pred_xy   = pred_boxes[...,0:2]
    # w and h predicted are 0 to 1 with 1 being image size
    y_pred_wh   = pred_boxes[...,2:4]
    # probability that there is something to predict here
    y_pred_conf = pred_boxes[...,4]

    # same as predicate except that we don't need to add an offset, coordinate are already between 0 and cell count
    y_true_xy   = true_boxes[...,0:2]
    # with and height
    y_true_wh   = true_boxes[...,2:4]
    # probability that there is something in that cell. 0 or 1 here as it's a certitude.
    y_true_conf = true_boxes[...,4]

    clss_loss  = K.sum(K.square(y_true_class - y_pred_class)*y_true_conf,  axis=-1)
    xy_loss    = 5 * K.sum(K.sum(K.square(y_true_xy - y_pred_xy),axis=-1)*y_true_conf, axis=-1)
    wh_loss    = 5 * K.sum(K.sum(K.square(K.sqrt(y_true_wh) - K.sqrt(y_pred_wh)), axis=-1)*y_true_conf, axis=-1)

    # when we add the confidence the box prediction lower in quality but we gain the estimation of the quality of the box
    # however the training is a bit unstable

    box_xy = pred_boxes[..., :2] * input_size
    box_wh = pred_boxes[..., 2:4] * input_size
    predict_xy_min, predict_xy_max = xywh2minmax(box_xy, box_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2    

    box_xy = true_boxes[..., :2] * input_size
    box_wh = true_boxes[..., 2:4] * input_size
    label_xy_min, label_xy_max = xywh2minmax(box_xy, box_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    #conf_loss = K.sum(K.square(0.5*y_true_conf*iou_scores - y_pred_conf), axis=-1)
    conf_loss = K.sum(y_true_conf*K.square(1 - y_pred_conf), axis=-1)
    noobj_conf_loss = K.sum(0.5*(1 - y_true_conf)*K.square(y_pred_conf), axis=-1)

    # final loss function
    d = xy_loss + wh_loss + conf_loss + clss_loss + noobj_conf_loss
    return d

model = Model(i, x)

adam = keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss=custom_loss, optimizer=adam) # better

print(model.summary())

#
# Training the network
#
parser = argparse.ArgumentParser(description='YOLO')
parser.add_argument('--train', help='train', action='store_true')
parser.add_argument('--epoch', help='epoch', const='int', nargs='?', default=1)
args = parser.parse_args()

images = np.array(images)
outputs = np.array(outputs)

print(outputs.shape)
print(images.shape)
if args.train:
    model.fit(images, outputs, batch_size=64, epochs=int(args.epoch))
    model.save_weights('masked.h5')
    exit()
else:
    model.load_weights('masked.h5')


axes=[0 for _ in range(100)]
fig, axes = plt.subplots()

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


all_entries = os.listdir("../masked/")

files = [entry for entry in all_entries if not os.path.isdir(entry)]

images = []
for file in files:
    if re.match("^.*\.jpg$", file):
        images.append(file)

for file in images:
    print(f"Doing image # {i}")
    img=prepare_image(cv.imread("../masked/" + file))
    #
    # Predict bounding box and classes
    P = model.predict(np.array([ img_to_array(img) ]))
    print("PREDICTING .....")
    #print(P.shape)
    #
    # Draw each boxes and class score over each images using pyplot
    #
    col = 0
    for row in range(grid_w):
        for col in range(grid_h):
            #p = P[0][col*grid_h+row]
            p = P[0][row][col]
            #print(p.shape)
            boxes = p[2:].reshape(nb_boxes,5)
            clss = np.argmax(p[0:2])
            ax = plt.subplot()
            for b in boxes:
                x, y, w, h = box_orig[0], box_orig[1], box_orig[2], box_orig[3]
                x = x * input_size - w * input_size / 2
                y = y * input_size - h * input_size / 2
                w = w * input_size
                h = h * input_size
                #x = b[0] * input_size
                #y = b[1] * input_size
                #w = b[2] * input_size
                #h = b[3] * input_size
                conf = b[4]
                
                if conf < 0.5:
                    continue
                print(x, y, w, h, conf)
                #print(b)
                print("-------------------++++++++++++++++----------------------")
                imgplot = plt.imshow(img)
                color = ['g','r','b','0'][clss]
                rect = patches.Rectangle((x-w/2, y-h/2), w, h, color=color, fill=False)
                plt.text(20, 100, classes[clss], color='red', fontweight='bold', bbox=dict(fill=False, edgecolor='red', linewidth=2))
                ax.add_patch(rect)
                plt.show()
    
plt.show()
