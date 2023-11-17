import numpy as np
import time
import matplotlib.pyplot as mplt

import SVDCompact

def pregunta2():
    V = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13])

    n = len(V)

    tiemposSVD = np.zeros(n)
    tiemposSVDCompact = np.zeros(n)

    tiempo1 = []
    tiempo2 = []

    for i in range(0, n):
        K = V[i]
        print(K)
        A = np.random.rand(2**(K), 2**(K-1))
        t1 = time.time()
        U1, S1, V1 = np.linalg.svd(A)  # Se usa operacion prefabricada
        et1 = time.time()-t1
        tiemposSVD[i] = et1
        # tiempo1.append(et1)
        print(et1)

        t2 = time.time()
        U2, S2, V2 = SVDCompact.svdCompact(A)
        et2 = time.time()-t2
        tiemposSVDCompact[i] = et2
        # tiempo2.append(et2)
        print(et2)
    # V = np.array(V)
    # mplt.figure()

    mplt.plot(V, tiemposSVD)
    mplt.plot(V, tiemposSVDCompact)
    # mplt.plot(V, tiempo1, 'r')
    # mplt.plot(V, tiempo2, 'b')
    mplt.legend()
    mplt.show()

pregunta2()
