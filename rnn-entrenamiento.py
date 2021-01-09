# Redes Neuronales Recurrentes (RNR)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 19:32:12 2020

@author: juangabriel
"""

# Parte 1 - Preprocesado de los datos

# Importación de las librerías
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

periodos = 60 # numero de periodos para testar

dir_path = '/home/ubuntu/Scripts/notebooks/DeepLearningAaZ/RNN/divisas'
valor = 'EUR-JPY'

# Importar el dataset de entrenamiento
dataset_train = pd.read_csv(os.path.join(dir_path,'datasets',"{}.csv".format(valor)))

# datafame reverse xq los datos vienen de mas a menos
dataset_train = dataset_train.iloc[::-1]

data_train = dataset_train.iloc[:-periodos, 1:2]
data_test = dataset_train.iloc[-periodos:, 1:2]

data_train['Close'] = data_train.iloc[:,:].apply(lambda x: x.replace(',','.'))
data_test['Close'] = data_test.iloc[:,:].apply(lambda x: x.replace(',','.'))

data_train['Close'] = data_train['Close'].astype(float)
data_test['Close'] = data_test['Close'].astype(float)

training_set = data_train.iloc[:, 1:2].values
testing_set  = data_test.iloc[:, 1:2].values

# Escalado de características
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Crear una estructura de datos con 60 timesteps y 1 salida
X_train = []
y_train = []
for i in range(60, training_set.shape[0]):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
    
X_train, y_train = np.array(X_train), np.array(y_train)

# Redimensión de los datos
# Una red neuronal recurrente necesita tres dimensiones

# En caso de que solo tengamos un valor a tratar

# (X_train.shape[0], X_train.shape[1], 1) indica la nueva dimension,
# (fila,columna,profundidad)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Como añadir otro array numpy de las mismas dimensiones a una matriz numpy
# X_train_p = X_train 
# X_train_p.shape
# X_train_2 = []
# for i in range(0,X_train_p.shape[0]):
#     X_train_2.append(X_train_p[i][:]+1)
# X_train_f = np.stack((X_train, X_train_2), axis=2)  # along dimension 2


# Parte 2 - Construcción de la RNR
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Inicialización del modelo
regressor = Sequential()

# Añadir la primera capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1) ))
regressor.add(Dropout(0.2))

# Añadir la segunda capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

# Añadir la tercera capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

# Añadir la cuarta capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Añadir la capa de salida
regressor.add(Dense(units = 1))

# Compilar la RNR
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Ajustar la RNR al conjunto de entrenamiento
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
regressor.save(os.path.join(dir_path,'models',"{}.h5".format(valor)))

# Parte 3 - Ajustar las predicciones y visualizar los resultados

# Obtener el valor de las acciones reales  de Enero de 2017
# dataset_test = pd.read_csv(os.path.join(dir_path,'Google_Stock_Price_Test.csv'))
real_stock_price = testing_set

# Obtener la predicción de la acción con la RNR para Enero de 2017
dataset_total = pd.concat((data_train['Close'], data_test['Close']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(data_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizar los Resultados
plt.plot(real_stock_price, color = 'red', label = 'Precio Real del {}'.format(valor))
plt.plot(predicted_stock_price, color = 'blue', label = 'Precio Predicho del {}'.format(valor))
plt.title("Prediccion con una RNR del valor {}".format(valor))
plt.xlabel("Fecha")
plt.ylabel("Precio del {}".format(valor))
plt.legend()
plt.savefig(os.path.join(dir_path,'models','{}-test.jpg'.format(valor)))
plt.show()

