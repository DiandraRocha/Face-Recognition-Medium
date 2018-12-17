# -*- coding: utf-8 -*-

import dlib
import PIL.Image
import numpy as np

# Instancia o objeto de detecção 
detector = dlib.get_frontal_face_detector()

# Utiliza o Pillow para abrir a imagem
im = PIL.Image.open("BarackObama.jpg")

# Utiliza a imagem já aberta pelo PIL para criar um numpy array
np_array = np.array(im)
  
# Transforma o numpay array em array RGB
rgb = np_array[:, :, ::-1]

# Usa a instância configurada do detector de faces para encontrar no RGB as faces contidas na imagem original
# O número 1 como argumento indica que a imagem vai ser processada uma vez
rects = detector(rgb, 1)

# Resultado: objeto rectangle de 4 pontos indicando a localização do rosto
print(rects)