# -*- coding: utf-8 -*-

import PIL.Image
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

# Instancia o objeto responsável pela codificação do rosto baseado no modelo de aprendizado
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

im = PIL.Image.open("BarackObama.jpg")

np_array = np.array(im)

rgb = np_array[:, :, ::-1]

rects = detector(rgb, 0)

print (rects)

if not rects:
    print('Nenhuma face encontrada na imagem!')

else:
    shape = predictor(rgb, rects[0])
    
    # Utiliza o face encoder, treinado pelo arquivo dat fornecido, 
    # usando o numpy array RGB e os pontos indicados pelo predictor 
    # para gerar os encodings baseado em 5 pontos faciais: cantos dos olhos e base do nariz
    encodings = np.array(face_encoder.compute_face_descriptor(rgb, shape, 1))
    print (encodings) 
