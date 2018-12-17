# -*- coding: utf-8 -*-

import face_recognition

image = face_recognition.load_image_file("BarackObama.jpg")

# Array de 4 valores referentes a localização do rosto na imagem
face_locations = face_recognition.face_locations(image)

print(face_locations)

# Reconhece os pontos faciais, como não foi passado o terceiro argumento que define por padrão 
# o tamanho da imagem para large, ele utiliza o padrão de predição de 68 pontos
# o resultado são arrays com 12 tuplas cada que indicam a localização de lábios, olhos, sobrancelhas, nariz e queixo
face_landmarks_list = face_recognition.face_landmarks(image)

print(face_landmarks_list)

# Gera os 128 encodings particulares do rosto
encoding = face_recognition.face_encodings(image)[0]

print(encoding)
