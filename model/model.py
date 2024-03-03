# NN modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# Plotting stuff
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# Image manipulation
import cv2 as cv
# Image augmentations
import albumentations as alb
# OS stuff
import os
# JSON stuff
import json

w = 1280
h = 720
# MODEL PARAMETERS 
INPUT_SHAPE = (h, w, 1, )
FILTER1_SIZE = 8
FILTER2_SIZE = 16
FILTER3_SIZE = 32
FILTER_SHAPE = (3, 3)
POOL_SHAPE = (3, 3)
FULLY_CONNECT_NUM = 64
NUM_CLASSES = 1
BBOX_SIZE = 4
OUTPUT_LAYER = BBOX_SIZE
OUTPUT_LAYER2 = NUM_CLASSES

def build_feature_extractor(inputs):
    # Feature extraction layer
    x = tf.keras.layers.Conv2D(FILTER1_SIZE, FILTER_SHAPE, activation='relu', input_shape=INPUT_SHAPE)(inputs)
    x = tf.keras.layers.MaxPooling2D(POOL_SHAPE)(x)

    x = tf.keras.layers.Conv2D(FILTER2_SIZE, FILTER_SHAPE, activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling2D(POOL_SHAPE)(x)

    x = tf.keras.layers.Conv2D(FILTER3_SIZE, FILTER_SHAPE, activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling2D(POOL_SHAPE)(x)

    return x

def build_model_adaptor(inputs):
    # Fully connected layer
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(FULLY_CONNECT_NUM, activation='relu')(x)
    x = tf.keras.layers.Dense(FULLY_CONNECT_NUM, activation='relu')(x)
    return x

def build_classifier_head(inputs):
    # Classifier output
    return tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid', name = 'classifier_head')(inputs)

def build_regressor_head(inputs):
    # Regressor output
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    return tf.keras.layers.Dense(BBOX_SIZE, name = 'regressor_head')(x)

def build_model(inputs):
    # Compose the model
    feature_extractor = build_feature_extractor(inputs)

    model_adaptor = build_model_adaptor(feature_extractor)

    classification_head = build_classifier_head(model_adaptor)

    regressor_head = build_regressor_head(model_adaptor)

    model = tf.keras.Model(inputs = inputs, outputs = [classification_head, regressor_head])

    return model

# BATCH SIZE AND EPOCHS SIZE
BATCH_SIZE = 32
EPOCHS = 20

# Build the model
model = build_model(tf.keras.layers.Input(shape=INPUT_SHAPE))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), 
    loss = {'classifier_head' : 'binary_crossentropy', 'regressor_head' : 'mse' }, 
    metrics = {'classifier_head' : 'accuracy', 'regressor_head' : 'mse' })

# Print the summary
model.summary()

# Read labels
lables = open("../validate/labels.txt")

# Input and output values
boxes = []
classes = []
input_size = 1280
images = []

for label in lables:
    if label == "":
        continue
    # Read the labels
    parts = label.split(" ")
    img=mpimg.imread("../validate/frame" + parts[0] + ".jpg")
    gray_img = np.array(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    box = [int(parts[1]) / w,  int(parts[2]) / h, (int(parts[1]) + int(parts[3])) / w, (int(parts[2]) + int(parts[4])) / h]
    if int(parts[5]) == 0:
        c = 0
    if int(parts[5]) > 0:
        c = 1
    images.append(np.array(gray_img))
    boxes.append(box)
    classes.append(c)

# Create manipulated images
augmentor = alb.Compose([alb.RandomCrop(width=w, height=h, always_apply=True), 
                         alb.HorizontalFlip(p=0.4),
                         alb.VerticalFlip(p=0.4),
                         alb.RandomBrightnessContrast(p=0.2)], 
                         bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))

num_images = len(images)
num_aug = 30

for i in range(0, num_images):
    for j in range(0, num_aug):
        augmented = augmentor(image = images[i], bboxes = [boxes[i]], class_labels = [classes[i]])
        cv.imwrite(f"../aug_data/images/frame{i}_{j}.jpg", augmented["image"])
        annotation = {}
        annotation["image"] = f"frame{i}.jpg"
        annotation["class"] = classes[i]
        if len(augmented["bboxes"]) == 0:
            continue
        annotation["bbox"] = augmented["bboxes"][0]
        with open(f"../aug_data/labels/frame{i}_{j}.json", "w") as f:
            json.dump(annotation, f)

classes = []
boxes = []
images = []

total = 0

# Load boxes and images
for i in range(0, num_images):
    for j in range(0, num_aug):
        try:
            with open(f"../aug_data/labels/frame{i}_{j}.json", "r") as f:
                ann = json.load(f)
            img=mpimg.imread(f"../aug_data/images/frame{i}_{j}.jpg")
            if ann["class"] == 0:
                classes.append(0)
            if ann["class"] > 0:
                classes.append(1)
            boxes.append(ann["bbox"])
            images.append(gray_img)
            total += 1
        except Exception as e:
            print(e)
            pass

# Read labels
lables = open("../training/labels.txt")

# Input and output values


classes = np.reshape(classes, (len(classes), 1))
boxes = np.reshape(boxes, (len(boxes), 4))
images = np.array(images)
images = np.reshape(images, (total, h, w, 1))

# Finally train the model and save the weights to the file
training_history = model.fit(images, (classes, boxes), epochs=EPOCHS, batch_size=BATCH_SIZE)

model.save('model.h5')
