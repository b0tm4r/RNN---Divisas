# RNN---Divisas


## Error de carga de modelo

Si al realizar el load_model nos tira el siguiente error:

KeyError: 'sample_weight_mode' #14040

Sustituir `from keras.models import load_model` por `from tensorflow.keras.models import load_model` 

Cargar con `model = load_model('model.h5', compile = False)`

[Github issue](https://github.com/keras-team/keras/issues/14040)
