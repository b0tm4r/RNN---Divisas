# -*- coding: utf-8 -*-

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
    # Datafame reverse xq los datos vienen de mas a menos y reindexamos
    
    df = df.iloc[::-1]
    df = df.reset_index(drop=True)
    # Close
    df['Close'] = df.iloc[:,1:2]
    df['Close'] = df['Close'].apply(lambda x: x.replace(',','.')).astype(float)
    # Open
    df['Open'] = df.iloc[:,2:3]
    df['Open'] = df['Open'].apply(lambda x: x.replace(',','.')).astype(float)
    # High
    df['High'] = df.iloc[:,3:4]
    df['High'] = df['High'].apply(lambda x: x.replace(',','.')).astype(float)
    # Low
    df['Low'] = df.iloc[:,4:5]
    df['Low'] = df['Low'].apply(lambda x: x.replace(',','.')).astype(float)
    # RSI 14
    rsi = talib.RSI(df['Close'])
    df['Rsi'] = rsi
    # Selección de columnas
    df = df.iloc[:,[0,6,7,8,9,10]]
    return df
    
## Variables de entrada
## periodos = numero de periodos de entrenamiento
dir_path = 'C:\\Users\\marcos\\AnacondaProjects\\jup\\tensorflow\\RNN-divisas'
periodos = 60 
valor = 'EUR-USD'

# Parte 3 - Test y visualizar los resultados

## La red se ha creado con dos capas de profundidad,
## La primera corresponde al precio, en posicion 1:2 del dataset
# La segunda corresponde al rsi, en posicion 5:6 del dataset

data_position = [[1,2],[5,6]]

modelo = load_model(os.path.join(dir_path,'models',"{}-precio-rsi.h5".format(valor)))
dataset_train = get_dataframe(dir_path,valor)
dataset_train = dataset_train.dropna()

dataset_train.head(3)

testing_set = dataset_train.iloc[-periodos:, data_position[0][0]:data_position[0][1] ].values
real_stock_price = testing_set

sc = [] 
X_values = []
for i in range(0,len(data_position)):
    sc.append(MinMaxScaler(feature_range = (0, 1)))
    values = dataset_train.iloc[ len(dataset_train) - len(testing_set) - periodos:, data_position[i][0]:data_position[i][1] ].values
    values = values.reshape(-1,1)
    values = sc[i].fit_transform(values)
    X_values.append(values)

X_test_list = []
for i in range(0,len(data_position)):
    X_test_data = []
    for x in range(periodos, X_values[i].shape[0]):
        X_test_data.append(X_values[i][x-periodos:x, 0])
    
    X_test_list.append(X_test_data)
        
X_test_np = []
for i in range(0,len(data_position)):
    X_test_np.append( np.array(X_test_list[i]) )

## axis = dimensiones
X_test = np.stack(X_test_np, axis=len(data_position))  

predicted_stock_price = modelo.predict(X_test)
predicted_stock_price = sc[0].inverse_transform(predicted_stock_price)

## Visualizar los Resultados

plt.plot(real_stock_price, color = 'red', label = 'Precio Real del {}'.format(valor))
plt.plot(predicted_stock_price, color = 'blue', label = 'Precio Predicho del {}'.format(valor))
plt.title("Testing del valor {}".format(valor))
plt.xlabel("Periodos")
plt.ylabel("Precio")
plt.legend()
plt.savefig(os.path.join(dir_path,'models','{}-test.jpg'.format(valor)))
plt.show()
