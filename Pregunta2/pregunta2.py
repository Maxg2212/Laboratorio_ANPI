# Imports
import numpy as np
import time
import matplotlib.pyplot as mplt

import SVDCompact

# ----------------------------------------------------------------------------------------------------------------------
# Funcion pregunta2
# Parametros: no tiene.
# Resultado: grafica comparativa entre los metodos SVD y SVDCompact, en la cual el eje x es Parametro K y en el eje y es el tiempo en segundos.
#
def pregunta2():
    """
            La funcion `pregunta2` utiliza valores de un array k_values, con el cual se va probando los metodos 'SVD' y 'SVDCompact' y va midiendo el
            tiempo que cada metodo tarda en cada valor de dicho array.
            """
    k_values = np.array([5, 6, 7, 8, 9, 10, 11, 12])

    num_elements = len(k_values)

    tiemposSVD = np.zeros(num_elements)
    tiemposSVDCompact = np.zeros(num_elements)


    for i in range(0, num_elements):
        K = k_values[i]
        print(K)
        # Generar matriz aleatoria A de tamaño 2^k x 2^(k-1)
        A = np.random.rand(2**(K), 2**(K-1))
        t1 = time.time()
        U1, S1, V1 = np.linalg.svd(A)  # Se usa operacion prefabricada
        et1 = time.time()-t1
        tiemposSVD[i] = et1

        print('Tiempo en SVD: ' + str(et1))

        t2 = time.time()
        U2, S2, V2 = SVDCompact.svdCompact(A)
        et2 = time.time()-t2
        tiemposSVDCompact[i] = et2

        print('Tiempo en SVDCompact: ' + str(et2))

    mplt.plot(k_values, tiemposSVD, 'r')
    mplt.plot(k_values, tiemposSVDCompact, 'g')
    mplt.xlabel('Parametro K')
    mplt.ylabel('Tiempo en segundos')
    mplt.legend(['GNU Octave','Nuevo Metodo'])
    mplt.show()

# Llamada de funcion pregunta2
pregunta2()
