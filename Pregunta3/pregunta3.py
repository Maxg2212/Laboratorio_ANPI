import numpy as np
import cv2
import os
from Pregunta2 import SVDCompact
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
        directory = os.path.join('training', f's{k}', f'{m}.jpg')
        A = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
        B = np.divide(A.astype('float'), 255.0)
        #print(directory)
        x = B.flatten()
        S[:, columna_s] = x
        columna_s += 1
    #print(S)


# Paso 2
fProm = np.zeros((totalPixels, 1))

for i in range(totalPixels):
    fProm[i] = np.sum(S[i, :])


fProm = 1 / totalImg * fProm
#print(fProm)

# Paso 3 y 4
A = np.zeros((totalPixels, totalImg))
for i in range(totalImg):
    A[:,i] = S[:,i] - fProm.flatten()
#print(A)



# Paso 5
U, S, V = SVDCompact.svdCompact(A)
#print(U)


# Paso 13
for n in range(1, sizeLength + 1):
    correctImage = ''
    minValue = 0
    direccion = os.path.join('compare', f'p{n}.jpg')
    A_compare = cv2.imread(direccion, cv2.IMREAD_GRAYSCALE)
    B_compare = np.divide(A_compare.astype('float'), 255.0)
    print(direccion)
    f = B_compare.flatten()
    #print(f)
    x = np.dot(U.conj().T,  (f - fProm.flatten()))
    #print(x)
    
    for k in range(1, sizeLength + 1):
        for m in range(1, numImages + 1):
            trainingDirectory = os.path.join('training', f's{k}', f'{m}.jpg')
            A_training = cv2.imread(trainingDirectory, cv2.IMREAD_GRAYSCALE)
            print(trainingDirectory)
            #direccion = f'training/s{k}/{m}.jpg'
            #A_training = cv2.imread(direccion)
            B_training = np.divide(A_training.astype('float'), 255.0)
            print("Entro al for 2 del paso 13")
            f_i = B_training.flatten()
            x_i = np.dot(U.conj().T,  (f_i - fProm.flatten()))
            dif = np.linalg.norm(x - x_i)
            #dif = (np.dot(np.transpose((x-xi)), (x-xi)))**(1/2)

            if minValue == 0:
                minValue = dif
            elif dif < minValue:
                minValue = dif
                correctImage = direccion

    mplt.subplot(1,2,1)
    mplt.imshow(A_compare, cmap='gray')
    mplt.title('Imagen Buscada')

    A2 = cv2.imread(correctImage, cv2.IMREAD_GRAYSCALE)
    mplt.subplot(1,2,2)
    mplt.imshow(A2, cmap='gray')
    mplt.title('Imagen encontrada')

    mplt.show()
    mplt.pause(0.2)





