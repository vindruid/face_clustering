import cv2
from keras.models import load_model
import os
import numpy as np
from skimage.transform import resize
import pickle

## DEFINE PATH & PARAMETER
path_prototxt = 'models/face_detection/deploy.prototxt'
path_model = 'models/face_detection/res10_300x300_ssd_iter_140000.caffemodel'
confidence = 0.60

## LOAD MODEL
print("[INFO] loading Face Detector model...\n")
net = cv2.dnn.readNetFromCaffe(path_prototxt, path_model)

img_path = 'images/1.debat-capres-2019-jokowi-vs-prabowo.jpg'

print(img_path)
img = cv2.imread(img_path)
(H, W) = img.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(
    img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    if detections[0,0,i,2] > confidence: # filter detected box with low confidence
        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H]) #convert back to original size
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(img, (startX, startY),
                            (endX, endY), (0, 255, 0), 2)
