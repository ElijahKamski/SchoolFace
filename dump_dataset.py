import pickle
import face_recognition
import os
import glob
from PIL import Image, ImageDraw
import numpy as np
import pickle

def prepare_database():
    database = {}
    for file in glob.glob("ds/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = face_recognition.load_image_file(file)
    return database


Polina = prepare_database()
for kk in Polina.keys():
    try:
        Polina[kk] = face_recognition.face_encodings(Polina[kk])[0]
    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
        continue
with open('dataset_faces.dat', 'ab') as f:
    pickle.dump(Polina, f)
