import numpy as np
import cv2

ANCHO = 92
ALTURA = 112
CONTADOR_PIXELES = ANCHO * ALTURA

vectoresTraining = []

for folderIndex in range(1,41):
    S = np.zeros((CONTADOR_PIXELES, 9))
    for imageIndex in range(1,10):
        imagePath = "training/s" + str(folderIndex) + "/" + str(imageIndex) + ".jpg"
        print(imagePath)
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        imgList = np.transpose(np.array([[pixel for row in img.tolist() for pixel in row]]))
        S[:,imageIndex-1] = imgList[:,0]
    vectoresTraining.append(S)
