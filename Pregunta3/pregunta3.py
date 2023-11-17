# Imports
import numpy as np
import cv2
import os
from Pregunta2 import SVDCompact
import matplotlib.pyplot as mplt

# ----------------------------------------------------------------------------------------------------------------------
# Funcion pregunta3
# Parametros: no tiene.
# Resultado: imagenes del folder 'training' que coinciden con las del folder 'compare'.
#
def pregunta3():
    """
                Funcion de reconocimiento facial 'pregunta3' donde se utiliza un directorio de entrenamiento donde contiene los directorios de cada persona
                y otro directorio donde se encuentran las imagenes que se desean comparar.
                """
    # Datos generales
    sizeLength = 40
    numImages = 9
    totalImg = 360
    totalPixels = 10304

    # Calculo de la matriz S con cada una de las imagenes cara

    S = np.zeros((totalPixels, totalImg))
    columna_s = 0
    for k in range(1, sizeLength + 1):
        for m in range(1, numImages + 1):
            directory = os.path.join('training', f's{k}', f'{m}.jpg')
            A = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
            B = np.divide(A.astype('float'), 255.0)
            x = B.flatten()
            S[:, columna_s] = x
            columna_s += 1



    # Calculo de f promedio sumando las columnas
    fProm = np.zeros((totalPixels, 1))

    for i in range(totalPixels):
        fProm[i] = np.sum(S[i, :])


    fProm = 1 / totalImg * fProm

    # Calculo de la matriz A restando la columna con fProm
    A = np.zeros((totalPixels, totalImg))
    for i in range(totalImg):
        A[:,i] = S[:,i] - fProm.flatten()



    # Descomposicion de la matriz A por medio de svdCompact
    U, S, V = SVDCompact.svdCompact(A)


    # Comparacion del X de la imagen nueva con el X_i de las imagenes buscada
    for n in range(1, sizeLength + 1):
        correctImage = ''
        minValue = 0
        direccion = os.path.join('compare', f'p{n}.jpg')
        A_compare = cv2.imread(direccion, cv2.IMREAD_GRAYSCALE)
        B_compare = np.divide(A_compare.astype('float'), 255.0)
        f = B_compare.flatten()
        x = np.dot(U.conj().T,  (f - fProm.flatten()))

        for k in range(1, sizeLength + 1):
            for m in range(1, numImages + 1):
                trainingDirectory = os.path.join('training', f's{k}', f'{m}.jpg')
                A_training = cv2.imread(trainingDirectory, cv2.IMREAD_GRAYSCALE)
                B_training = np.divide(A_training.astype('float'), 255.0)
                f_i = B_training.flatten()
                x_i = np.dot(U.conj().T,  (f_i - fProm.flatten()))
                epsilon = np.linalg.norm(x - x_i)

                if minValue == 0:
                    minValue = epsilon
                elif epsilon < minValue:
                    minValue = epsilon
                    correctImage = direccion

        mplt.subplot(1,2,1)
        mplt.imshow(A_compare, cmap='gray')
        mplt.title('Rostro nuevo')

        readImage = cv2.imread(correctImage, cv2.IMREAD_GRAYSCALE)
        mplt.subplot(1,2,2)
        mplt.imshow(readImage, cmap='gray')
        mplt.title('Rostro identificado')

        mplt.show()
        mplt.pause(1)

# Llamada a la funcion pregunta3
pregunta3()





