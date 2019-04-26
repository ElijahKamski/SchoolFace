#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Khripkov Yaroslav'
#Импортируем необходимые библиотеки
import face_recognition
import os
import glob
from PIL import Image, ImageDraw
import numpy as np
import pickle
#Загружаем подготовленную базу лиц учеников (Настраивается с помощью файлаdump_dataset.py
with open('dataset_faces.dat', 'rb') as f:
    Polina = pickle.load(f)
#Загружаем фотографию класса для распознавания
file_name = input('Введите имя файла>>')+'.jpg'
pic = face_recognition.load_image_file(file_name)
#-----------------------------------------------
#pic_enc = face_recognition.face_encodings(pic)[0]
#Создаём списки учащихся и преобразований
users = list(Polina.keys())
encodings = [Polina[kk] for kk in Polina.keys()]
face_encodings = face_recognition.face_encodings(pic)
pupils = set()
#Распознавание каждого лица на фото
for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(encodings, face_encoding)

    name = "Unknown"

    # If a match was found in encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        pupils.add(users[best_match_index])
print(pupils)
#-----------------------------------------------
#Вывод и сохранение размеченых изображений для анализа
if input('Хотите ли вы вывести изображение?>>')=='y':
    face_locations = face_recognition.face_locations(pic)
    face_encodings = face_recognition.face_encodings(pic, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(pic)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(encodings, face_encoding)

        name = "Unknown"

        # If a match was found in encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        #print(face_distances)
        if matches[best_match_index]:
            name = users[best_match_index]

        # Отрисовка границ лица
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Отрисовка подписей под лицами
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


    del draw
    #Вывыод изображения
    pil_image.show()
if input('Хотите ли вы сохранить размеченое изображение?>>')=='y':
    #Сохранение изображения
    pil_image.save(file_name+'with_boxes'+'.jpg')