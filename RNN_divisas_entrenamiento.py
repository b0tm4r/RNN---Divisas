## -*- coding: utf-8 -*-

'''
__author__ = "b0tm4r"
__credits__ = ["https://github.com/joanby - Juan Gabriel Gomila"]
__version__ = "1.0.1"
__maintainer__ = "b0tm4r"
__email__ = "b0tm4r@gmail.com"
__status__ = "Testing"

Basado en https://github.com/joanby/deeplearning-az/blob/master/datasets/Part%203%20-%20Recurrent%20Neural%20Networks%20(RNN)/rnn.py

'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import talib

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model

## Importar el dataset de entrenamiento
def get_dataframe(dir_path, valor):
    df1 = pd.read_csv(os.path.join(dir_path,'datasets',"{}.csv".format(valor)))
    df2 = pd.read_csv(os.path.join(dir_path,'datasets',"{}-total.csv".format(valor)))
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop_duplicates(['Fecha'])
    
    ## Datafame  df.iloc[::-1] para reverse xq los datos 
    ## vienen de mas a menos y reindexamos    
    df = df.iloc[::-1]
    df = df.reset_index(drop=True)
    ## Close
    df['Close'] = df.iloc[:,1:2]
    df['Close'] = df['Close'].apply(lambda x: x.replace(',','.')).astype(float)
    ## Open
    df['Open'] = df.iloc[:,2:3]
    df['Open'] = df['Open'].apply(lambda x: x.replace(',','.')).astype(float)
    ## High
    df['High'] = df.iloc[:,3:4]
    df['High'] = df['High'].apply(lambda x: x.replace(',','.')).astype(float)
    ## Low
    df['Low'] = df.iloc[:,4:5]
    df['Low'] = df['Low'].apply(lambda x: x.replace(',','.')).astype(float)
    ## RSI 14
    rsi = talib.RSI(df['Close'])
    df['Rsi'] = rsi
    ## Selección de columnas
    df = df.iloc[:,[0,6,7,8,9,10]]
    return df

# Parte 1 - Carga y procesado de datos
    
## Numero de periodos para entrenamiento    
dir_path = 'C:\\Users\\marcos\\AnacondaProjects\\jup\\tensorflow\\RNN-divisas'
periodos = 60 
valor = 'EUR-USD'

dataset_train = get_dataframe(dir_path,valor)
dataset_train = dataset_train.dropna()
dataset_train.head(3)

## Columnas con valores a tratar, precio y rsi
training_set = []
training_set.append( dataset_train.iloc[:-periodos, 1:2].values )
training_set.append( dataset_train.iloc[:-periodos, 5:].values )

## Escalado de características
sc = []
for i in training_set:
    sc.append(MinMaxScaler(feature_range = (0, 1)))

training_set_scaled = []
for i in range(0,len(training_set)):
    training_set_scaled.append(sc[i].fit_transform(training_set[i]))
    
## Crear una estructura de datos con 60 timesteps por columna y 1 salida 
X_train = []
y_train = []
for i in range(0,len(training_set)):
    X_train_set = []
    y_train_set = []
    for x in range(periodos, training_set[i].shape[0]):
       X_train_set.append(training_set_scaled[i][x-periodos:x, 0])
       y_train_set.append(training_set_scaled[i][x, 0])

    X_train.append(X_train_set)
    y_train.append(y_train_set)    

X_train_list = []
y_train_list = []
for i in range(0,len(training_set)):
    X_train_data, y_train_data = np.array(X_train[i]), np.array(y_train[i])
    X_train_list.append(X_train_data)
    y_train_list.append(y_train_data)

## axis = dimensiones
X_train = np.stack(X_train_list, axis=len(training_set))
## y_train se corresponde a la columna base, en este caso el precio
## xq es en aquello que estamos interesados en predecir
y_train = y_train_list[0] 

# Parte 2 - Construcción de la RNR

## Inicialización del modelo
regressor = Sequential()

## Añadir la primera capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], len(training_set)) ))
regressor.add(Dropout(0.2))

## Añadir la segunda capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

## Añadir la tercera capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

## Añadir la cuarta capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

## Añadir la capa de salida
regressor.add(Dense(units = 1))

## Compilar la RNR
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

## Ajustar la RNR al conjunto de entrenamiento
regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)

# Parte 3 -  Guardamos el modelo
regressor.save(os.path.join(dir_path,'models',"{}-precio-rsi.h5".format(valor)))
