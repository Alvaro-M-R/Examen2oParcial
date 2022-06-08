# -*- coding: utf-8 -*-
"""
@author: Alvaro
"""

import numpy as np
import pandas as pd


df = pd.read_csv('iris.csv')
print(df)

##Tomamos las primeras 100 filas y dividimos el conjunto de datos en x (característica) e y (objetivo)luego, 
##np.where transforma los datos nominales y de texto a numérico
x = df.iloc[0:100,[0, 1, 2, 3]].values
y = df.iloc[0:100,4].values
y = np.where(y=='Setosa', 0, 1)
##División de datos
#Las filas 1~50 son setosa 
##asi que recopilamos la fila 1~40 y la fila 51~90 para que sean el conjunto de entrenamiento, y las 
#filas restantes son el conjunto de prueba.
x_train = np.empty((80, 4))
x_test = np.empty((20, 4))
y_train = np.empty(80)
y_test = np.empty(20)
x_train[:40],x_train[40:] = x[:40],x[50:90]
x_test[:10],x_test[10:] = x[40:50],x[90:100]
y_train[:40],y_train[40:] = y[:40],y[50:90]
y_test[:10],y_test[10:] = y[40:50],y[90:100]
##Definir funciones
def sigmoid(x):
    return 1/(1+np.exp(-x))
def activation(x, w, b):
    return sigmoid(np.dot(x, w)+b)
def update(x, y_train, w, b, eta): 
    y_pred = activation(x, w, b) 
    # activator
    a = (y_pred - y_train) * y_pred * (1- y_pred) 
    # función de pérdida derivada parcial
    for i in range(4):
        w[i] -= eta * 1/float(len(y)) * np.sum(a*x[:,i])
    b -= eta * 1/float(len(y))*np.sum(a)
    return w, b

##Entrenamiento
#Tasa de aprendizaje : establecer eta = 1
#generacion : Ejecute ambas generaciones = 15 y generacion = 100
weights = np.ones(4)/10 
bias = np.ones(1)/10 
eta = 0.1
for _ in range(1000):# Ejecutar tanto =15 como =100
 weights, bias = update(x_train, y_train, weights, bias, eta=0.1)

##Pruebas
 
print("Epocas = 15") 
print('weights = ', weights, 'bias = ', bias)
print("y_test = {}".format(y_test))
print(activation(x_test, weights, bias) )
#print()
'''Las primeras 10 predicciones de epoccas = 15 están entre 0,46 y 0,49, 
mientras que las primeras 10 predicciones de epocas = 100 están entre 0,23 y 0,30. 
Las últimas 10 predicciones de epochs = 15 están entre 0,57 y 0,63, mientras que las últimas 10 predicciones 
de epochs = 100 están entre 0,64 y 0,81. A medida que aumentan las epocas, 
los valores de los dos grupos de flores se vuelven más distantes.'''


