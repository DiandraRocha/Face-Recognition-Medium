# -*- coding: utf-8 -*-

import face_recognition

image_obama = face_recognition.load_image_file("BarackObama.jpg")
second_image_obama = face_recognition.load_image_file("BarackObama2.jpg")
image_trump = face_recognition.load_image_file("DonaldTrump.jpg")

face_locations_obama = face_recognition.face_locations(image_obama)
second_face_locations_obama = face_recognition.face_locations(second_image_obama)
face_locations_trump = face_recognition.face_locations(image_trump)

encodings_obama = face_recognition.face_encodings(image_obama)[0]
second_encodings_obama = face_recognition.face_encodings(second_image_obama)[0]
encodings_trump = face_recognition.face_encodings(image_trump)[0]

# Realiza a comparação euclidiana entre os dois arrays
compare_first_round = face_recognition.compare_faces([encodings_obama], encodings_trump)
compare_second_round = face_recognition.compare_faces([encodings_obama], second_encodings_obama)

# Resultado: false
print(compare_first_round)

# Resultado: true
print(compare_second_round)