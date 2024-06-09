# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:47:56 2024

@author: nacho
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.optimizers import SGD 

from tensorflow import keras
from keras import regularizers
from keras.models import load_model

from scipy.interpolate import interp1d

def increase_sample_size(vector, new_size):
    x_old = np.linspace(0, 1, num=len(vector))
    x_new = np.linspace(0, 1, num=new_size)
    f = interp1d(x_old, vector, kind='linear')  # Puedes cambiar 'linear' a 'quadratic' o 'cubic' 
    return f(x_new)
def read_datos(file_tbl, file_csv):
    chunk_size = 10000  # Tamaño del chunk para leer datos
    all_indices_vmp = []
    all_data_I_mod = []
    all_valores_vmp = []
    curva_v_final = []

    with open(file_tbl, 'r') as f:
        lines = f.readlines()
        curva_v = pd.Series([float(x.replace(',', '.')) for x in lines[1].split()])

    # Leer tabla_curvas_i 
    tabla_curvas_i_iter = pd.read_csv(file_tbl, skiprows=2, sep="\s+", chunksize=chunk_size)

    for data_chunk in tabla_curvas_i_iter:
        tabla_curvas_i = data_chunk.T.values  
        
        indices_vmp = []
        data_I_mod_chunk = []
        valores_vmp_chunk = []

        for i, data in enumerate(tabla_curvas_i):
            pot = curva_v * data
            indice_vmp = np.argmax(pot)
            indices_vmp.append(indice_vmp)
            
            curva_v_exp, curva_I_exp = increase_sample_vmp(indice_vmp, curva_v, data)
            data_I_mod_chunk.append(curva_I_exp)
            for _ in range(len(curva_v_exp)):
    # Agrega el valor deseado al final de valores_vmp_chunk
                valores_vmp_chunk.append(curva_v_exp[np.argmax(curva_v_exp * curva_I_exp)])
          #  valores_vmp_chunk.append(curva_v_exp[np.argmax(curva_v_exp * curva_I_exp)])
            curva_v_final.append(curva_v_exp)

        all_indices_vmp.extend(indices_vmp)
        all_data_I_mod.extend(data_I_mod_chunk)
        all_valores_vmp.extend(valores_vmp_chunk)
    
    # Aseguramos que todos los arrays sean unidimensionales y estén concatenados correctamente
    all_data_I_mod = np.concatenate(all_data_I_mod)
    curva_v_final = np.concatenate(curva_v_final)
    all_valores_vmp = np.array(all_valores_vmp)

    return all_data_I_mod, curva_v_final, all_valores_vmp

   
def build_model(dataset,num_capas,function):
    
    model=Sequential()
    model.add(Input(shape=(dataset.shape[1],)))
   
    for i in range(num_capas):

       model.add(Dense(64,activation=function))
   
    model.add(Dense(1))                  
    return model                  


def increase_sample_vmp(indice,vector,vector_I):
    num_pasos = 40
    ancho_intervalo = min(6, indice, len(vector) - indice - 1)
    valor_menos = vector[indice - ancho_intervalo]
    valor_mas = vector[indice + ancho_intervalo]
    vector_lineal = np.linspace(valor_menos, valor_mas, num_pasos)
    valores_intervalo_I = vector_I[indice - ancho_intervalo:indice + ancho_intervalo + 1]

    x_original = np.linspace(0, 1, num=len(valores_intervalo_I))
    x_interpolado = np.linspace(0, 1, num=num_pasos)
    f_interpolacion = interp1d(x_original, valores_intervalo_I)
    valores_interpolados = f_interpolacion(x_interpolado)

    segmento_1_v = vector[:indice - ancho_intervalo]
    segmento_2_v = vector_lineal
    segmento_3_v = vector[indice + ancho_intervalo + 1:]
    vector_completo_v = np.concatenate((segmento_1_v, segmento_2_v, segmento_3_v))

    segmento_1_I = vector_I[:indice - ancho_intervalo]
    segmento_2_I = valores_interpolados
    segmento_3_I = vector_I[indice + ancho_intervalo + 1:]
    vector_I_ampliado = np.concatenate((segmento_1_I, segmento_2_I, segmento_3_I))
    return vector_completo_v,vector_I_ampliado


data_I, data_tension, results_Vmp = read_datos(
    ".\shadowandframeDef.tbl", "shadowandframeDef_pmp.csv")
#Creacion del de dataset
data_total = pd.DataFrame({'data_I': data_I, 'data_tension': data_tension, 'Vmp': results_Vmp})

#%% Primer Entrenamiento
ini=0
final=5000
dataset_1 = pd.DataFrame( {'data_I': data_total.data_I.iloc[ini:final], 'data_tension': data_total.data_tension.iloc[ini:final]})
dataset_1 = dataset_1.drop_duplicates()
train_data = dataset_1.sample(frac=0.8, random_state=0)
test_data = dataset_1.drop(train_data.index)
train_labels = data_total.Vmp[train_data.index]
test_labels = data_total.Vmp[test_data.index]

num_capas=5
function_act='sigmoid'
#function_act='elu'
#function_act='tanh'
#function_act='softmax'
model_1 = build_model(train_data,num_capas,function_act)
OPTIM = keras.optimizers.Nadam()


# Compila modelo con el optimizador personalizado
model_1.compile(optimizer=OPTIM, loss='mean_squared_error', metrics=[
    'mean_squared_error', 'mean_absolute_error'])

EPOCHS = 250
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='mean_squared_error', patience=10, min_delta=0.08, mode='min') 
history = model_1.fit(train_data, train_labels, epochs=EPOCHS, callbacks=[
    early_stop], validation_split=0.3, verbose=1, batch_size=120) 

results = model_1.evaluate(test_data, test_labels)
print(results)

model_1.summary()
titulo = f'{num_capas}_'+function_act+f'_data_{ini}_{final}'

   
model_1.save('Modelos/'+titulo + '.keras')
model_1.save('Modelos/'+titulo + '.h5')

#%% Segundo Entrenamiento

ini=210500
final=214500
dataset_2 = pd.DataFrame( {'data_I': data_total.data_I.iloc[ini:final], 'data_tension': data_total.data_tension.iloc[ini:final]})
dataset_2 = dataset_2.drop_duplicates()
train_data_2 = dataset_2.sample(frac=0.8, random_state=0)
test_data_2= dataset_2.drop(train_data.index)
train_labels_2 = data_total.Vmp[train_data.index]
test_labels_2 = data_total.Vmp[test_data.index]

function_act='sigmoid'
#function_act='elu'
#function_act='tanh'
#function_act='softmax'
num_capas=5
model_2 = build_model(train_data,num_capas)
OPTIM = keras.optimizers.Nadam()


# Compila modelo con el optimizador personalizado

model_1.compile(optimizer=OPTIM, loss='mean_squared_error', metrics=[
    'mean_squared_error', 'mean_absolute_error'])

EPOCHS = 250
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='mean_squared_error', patience=10, min_delta=0.08, mode='min') 
history = model_2.fit(test_data_2, test_labels_2, epochs=EPOCHS, callbacks=[
    early_stop], validation_split=0.3, verbose=1, batch_size=120)  

results = model_2.evaluate(test_data, test_labels)
print(results)

model_2.summary()

titulo = f'{num_capas}_'+function_act+f'_data_{ini}_{final}'

   
model_2.save('Modelos/'+titulo + '.keras')
model_2.save('Modelos/'+titulo + '.h5')


#%% Tercer Entrenamiento
dataset = pd.DataFrame( {'data_I': data_total.data_I, 'data_tension': data_total.data_tension})
train_data = dataset.sample(frac=0.8, random_state=0)
test_data = dataset.drop(train_data.index)
train_labels = data_total.Vmp[train_data.index]
test_labels = data_total.Vmp[test_data.index]

function_act='sigmoid'
#function_act='elu'
#function_act='tanh'
#function_act='softmax'
num_capas=5
model = build_model(train_data,num_capas,function_act)
OPTIM = keras.optimizers.Nadam()


# Compila tu modelo con el optimizador personalizado

model.compile(optimizer=OPTIM, loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])

EPOCHS = 250
validation=0.3
early_stop = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=10, min_delta=0.08, mode='min')  # callbacks=[early_stop],
history = model.fit(train_data, train_labels, epochs=EPOCHS, callbacks=[early_stop], validation_split=validation, verbose=1, batch_size=1024)  # ,batch_size=120

results = model.evaluate(test_data, test_labels)
print(results)

titulo = f'{num_capas}_'+function_act+f'_alldata'

   
model.save('Modelos/'+titulo + '.keras')
model.save('Modelos/'+titulo + '.h5')
