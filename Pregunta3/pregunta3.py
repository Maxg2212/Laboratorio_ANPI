import numpy as np
import cv2
import os
from Pregunta1 import SVDCompact
import matplotlib.pyplot as mplt


#Datos generales
sizeLength = 40
numImages = 9
totalImg = 360
totalPixels = 10304

#Paso 1

S = np.zeros((totalPixels, totalImg))
columna_s = 0
for k in range(1, sizeLength + 1):
    for m in range(1, numImages + 1):
        directory = f'training/s{k}/{m}.jpg'
        A = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
        B = cv2.normalize(A.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        x = B.flatten()
        S[:, columna_s] = x
        columna_s += 1

# Paso 2
fMean = np.zeros((totalPixels, 1))

for i in range(totalPixels):
    fMean[i] = np.sum(S[i,:])

fMean = 1/totalImg * fMean

# Paso 3 y 4
A = np.zeros((totalPixels, totalImg))
for i in range(totalImg):
    A[:,i] = S[:,i] - fMean.flatten()

# Paso 5
U, S, V = SVDCompact.svdCompact(A)

# Paso 13
for n in range(1, sizeLength + 1):
    selectedImg = ''
    minValue = 0
    direccion = f'compare/p{n}.jpg'
    A1 = cv2.imread(direccion)
    B = cv2.normalize(A1.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    f = B.flatten()
    x = np.dot(np.transpose(U), (f - fMean))

    for k in range(1, sizeLength + 1):
        for m in range(1, numImages + 1):
            direccion = f'training/s{k}/{m}.jpg'
            A = cv2.imread(direccion)
            B = cv2.normalize(A.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            f = B.flatten()
            xi = np.dot(np.transpose(U), (f - fMean))
            dif = np.linalg.norm(x - xi)
            #dif = (np.dot(np.transpose((x-xi)), (x-xi)))**(1/2)

            if minValue == 0:
                minValue = dif
            elif dif < minValue:
                minValue = dif
                selectedImg = direccion

    mplt.subplot(1,2,1)
    mplt.imshow(A1)
    mplt.title('Imagen Buscada')

    A2 = mplt.imread(selectedImg)
    mplt.subplot(1,2,2)
    mplt.imshow(A2)
    mplt.title('Imagen encontrada')
    mplt.pause(0.2)


