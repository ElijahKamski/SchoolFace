import face_recognition
import os
import glob
import pickle
def prepare_database():
    database = {}
    for file in glob.glob("dd/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = face_recognition.load_image_file(file)
    return database

with open('dataset_faces.dat', 'rb') as f:
    kara = pickle.load(f)
Polina = prepare_database()
for kk in Polina.keys():
    try:
        kara[kk] = face_recognition.face_encodings(Polina[kk])[0]
    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
        continue
print(kara)
with open('dataset_faces.dat', 'ab') as f:
    pickle.dump(kara, f)
