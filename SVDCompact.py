import numpy as np

def svdCompact(A):
    m, n = A.shape
    if m > n:
        M1 = np.dot(A, np.transpose(A)) 
        V1, D = np.linalg.eig(M1)
        y1 = D
        const = m * np.max(y1) * np.finfo(np.float64).eps
        y2 = (y1 > const).astype(int)
        rA = np.sum(y2)
        y3 = y1 * y2  # rango de la matriz

        s1 = np.sort((np.sqrt(y3)))[::-1]

        order = np.array(list(range(len(s1))))[::-1]

        V2 = V1[:,order]

        Vr = V2[:,0:rA]

        Sr = s1[0:rA]
        
        Ur = (1/np.transpose(Sr))*(np.dot(A, Vr))
        return Ur, Sr, Vr
    else:
        M1 = np.dot(A, np.transpose(A)) 
        U1, D = np.linalg.eig(M1)
        y1 = D
        const = m * np.max(y1) * np.finfo(np.float64).eps
        y2 = (y1 > const).astype(int)
        rA = np.sum(y2)  # rango de la matriz
        y3 = y1 * y2

        s1 = np.sort((np.sqrt(y3)))[::-1]

        order = np.array(list(range(len(s1))))[::-1]

        U2 = U1[:,order]

        Ur = U2[:,0:rA]
        
        Sr = s1[0:rA]

        Vr = (1 / np.transpose(Sr)) * np.dot(np.transpose(A), Ur)
        return Ur, Sr, Vr



