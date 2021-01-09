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
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Prediccion a 14 periodos
periodos_a_predecir = 14
valor = 'EUR-JPY'

# file path
dir_path = '/home/ubuntu/Scripts/notebooks/DeepLearningAaZ/RNN/divisas'
# Importar el dataset de entrenamiento
dataset_train = pd.read_csv(os.path.join(dir_path,'datasets',"{}.csv".format(valor)))
# datafame reverse xq los datos vienen de mas a menos
dataset_train = dataset_train.iloc[::-1]

data_train = dataset_train.iloc[:, 1:2]
data_train['Close'] = data_train.iloc[:,:].apply(lambda x: x.replace(',','.'))
data_train['Close'] = data_train['Close'].astype(float)
data_train = data_train.iloc[:, 1:2]

# Parte 3 - Ajustar las predicciones y visualizar los resultados
mi_modelo = load_model(os.path.join(dir_path,'models',"{}.h5".format(valor)))
sc = MinMaxScaler(feature_range = (0, 1))

dataset_total = data_train['Close']
datataset_total = dataset_total.reset_index(drop=True)
for x in range(0,periodos_a_predecir):
    inputs = dataset_total[-60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.fit_transform(inputs)

    X_test = []

    X_test.append(inputs[:, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = mi_modelo.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    predicted_stock_price_value =  predicted_stock_price[0][0]
    dataset_total = dataset_total.append( pd.Series([predicted_stock_price_value]), ignore_index=True)
 
# Visualizar los Resultados
plt.plot(dataset_total[-periodos_a_predecir-1:].values, color = 'blue', label = 'Precio Predicho del {}'.format(valor))
plt.title("Prediccion con una RNR del valor {}".format(valor))
plt.xlabel("Fecha")
plt.ylabel("Precio del {}".format(valor))
plt.legend()
plt.savefig(os.path.join(dir_path,'result','{}-prediccion-{}-periodos.jpg'.format(valor,periodos_a_predecir)))
plt.show()
