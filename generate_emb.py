import cv2
from keras.models import load_model
import os
import numpy as np
from skimage.transform import resize
import pickle

## DEFINE PARAMETER
path_prototxt = 'models/face_detection/deploy.prototxt'
path_model = 'models/face_detection/res10_300x300_ssd_iter_140000.caffemodel'
path_embedding_model = 'models/embeddings/facenet_keras.h5'
confidence = 0.60

## LOAD MODEL
print("[INFO] loading Face Detector model...\n")
net = cv2.dnn.readNetFromCaffe(path_prototxt, path_model)
print("[INFO] Loaded !")
print("==" * 20)
print("[INFO] loading Face Embeding model...\n")
facenet = load_model(path_embedding_model)
facenet._make_predict_function()
print("[INFO] Loaded !")

## DEFINE FUNCTION

def calc_embeds(imgs, facenet):
    """ for calculate embeddings from facenet keras """
    image = resize(imgs, (160, 160), mode='reflect')
    aligned_images = prewhiten(image)
    embs = l2_normalize(facenet.predict(aligned_images))
    embs = embs.astype(float)
    embs = list(embs.flatten())
    return embs

def prewhiten(x):
    """" prewithen image before as facenet input """
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    axis = (1, 2, 3)
    size = x.size

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    """ normalize face vector """
    output = x / np.sqrt(np.maximum(
        np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

## iterate image
path_dir = 'images/'
list_images = sorted([os.path.join(path_dir,img_name) for img_name in os.listdir(path_dir) 
                    if os.path.join(path_dir,img_name).split('.')[-1] in ['png', 'webp', 'jpeg', 'JPG', 'jpg'] ])


dict_embs = {}
for img_path in list_images:
    print(img_path)
    img = cv2.imread(img_path)
    (H, W) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(
        img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        if detections[0,0,i,2] > confidence: # filter detected box with low confidence
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(img, (startX, startY),
                            (endX, endY), (0, 255, 0), 2)
            face_crop = img[startY:endY, startX:endX]
            emb = calc_embeds(face_crop, facenet)
            dict_embs[img_path + '_' + str(i)] = emb

with open("embeddings.pkl", 'wb') as filename: 
    pickle.dump(dict_embs, filename)
print("SAVED")
