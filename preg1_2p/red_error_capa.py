# -*- coding: utf-8 -*-
"""

@author: Alvaro
"""
import numpy as np
####
# Vamos a crear una pequeña red neuronal que compute
# los valores de la tabla XOR
##Primera parte, generemos un input que sea una matriz
## Matriz de 2x4
## 0 0
## 0 1
## 1 0
## 1 1

INPUT_X = np.array([[0,0],[0,1],[1,0],[1,1]])
print ("INPUT_X:\n", INPUT_X)
 
##Luego la matriz de los resultados esperados
## en nuestro caso
## 0
## 1
## 1
## 0
EXPECTED_RESULT = np.array([[0,1,1,0]]).T # queremos en formato columna
print ("EXPECTED_RESULT:\n", EXPECTED_RESULT)
## Resultado: la tabla de verdad de la operación XOR
print ("RESULT:\n", np.append(INPUT_X,EXPECTED_RESULT, axis=1))

def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))
 
#Seteamos una SEED que nos ayudarà a que los numeros esten
# distribuidos de forma random, pero siempre igual para entender
# en qué afectan los cambios que realizamos
np.random.seed(0)
 

SYN0 = 2*np.random.random((2,3)) - 1
## **Y las conexiones de la CAPA1 a la CAPA2 que es el output
SYN1 = 2*np.random.random((3,1)) - 1
 
# Vamos a preparar iteraciones para aprender
# Probad a cambiar este numero para ver en qué afecta
for i in range(20000):

    l0 = INPUT_X

    l1 = sigmoid(np.dot(l0, SYN0))

    l2 = sigmoid(np.dot(l1, SYN1))
 
    #Computamos el error de la capa final
    l2_error = EXPECTED_RESULT - l2

    ## ** a los pesos para "aprender"
    l2_delta = l2_error * sigmoid(l2, True)

    l1_error = np.dot(l2_delta, SYN1.T)

    l1_delta = l1_error * sigmoid(l1, True)

    SYN0 += np.dot(l0.T, l1_delta)

    SYN1 += np.dot(l1.T, l2_delta)
    if (i % 1000) == 0 :
        print ("Error:" + str(np.mean(np.abs(l2_error))))
 
print (l2)



