from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import pickle
import cv2
import os
from shutil import copyfile

with open("embeddings.pkl", "rb") as filename:
    dict_emb = pickle.load(filename) 

encodings = list(dict_emb.values())
filenames = list(dict_emb.keys())

clt = DBSCAN(eps = 0.680,metric="euclidean", n_jobs=1 , min_samples = 5)
preds = clt.fit_predict(encodings)

labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

# list(preds).count(0),list(preds).count(1),list(preds).count(2),list(preds).count(3), list(preds).count(4), list(preds).count(-1)


initial_path = 'images'
target_path = 'clustered_images'

for name_, id_ in zip(filenames, preds):
    if id_ != -1: #remove image with unknown faces
        target_path_id = os.path.join(target_path, str(id_))
        if not os.path.exists(target_path_id):
            os.makedirs(target_path_id)

        filename = '_'.join(name_.split('/')[1].split('_')[:-1])
        from_path = os.path.join(initial_path,filename)
        to_path = os.path.join(target_path_id, filename)

        copyfile(from_path, to_path)
    

# SHOW IMAGE MONTAGES
target_path = 'clustered_images'
total_show = 9
montages_show_shape = (3,3)
montages_img_shape = (200,200)

for id_ in sorted(labelIDs):
    list_image = []
    
    if id_ != -1: #remove image with unknown faces
        target_path_id = os.path.join(target_path, str(id_))
        list_img_filename = os.listdir(target_path_id)[:total_show]
        for img_filename in list_img_filename:
            path_image = os.path.join(target_path_id, img_filename)
            img = cv2.imread(path_image)
            
            img = cv2.resize(img, montages_img_shape)
            list_image.append(img)

        # create a montage using 96x96 "tiles" with 5 rows and 5 columns
        montage = build_montages(list_image, montages_img_shape, montages_show_shape)[0]

        # show the output montage
        title = "Face ID #{}".format(id_)
        cv2.imshow(title, montage)
        cv2.waitKey(0)
        
cv2.destroyAllWindows()