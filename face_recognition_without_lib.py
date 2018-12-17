# -*- coding: utf-8 -*-

import PIL.Image
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()

#Instancia o objeto responsável pela identificação dos pontos de referência (landmarks)
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

im = PIL.Image.open("BarackObama.jpg")

np_array = np.array(im)

rgb = np_array[:, :, ::-1]
rects = detector(rgb, 0)

print (rects)

# Caso rects seja um retorno vazio, quer dizer que não há nenhuma face na imagem
if not rects:
    print('Nenhuma face encontrada na imagem!')

# Caso haja...
else:
    
    # Define e retorna no objeto shape os 5 pontos de referência dentro do rosto encontrado
    # Pode encontrar uma ou mais, porém utiliza-se somente a primeira
    shape = predictor(rgb, rects[0])
