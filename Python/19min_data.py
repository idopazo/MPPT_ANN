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
    f = interp1d(x_old, vector, kind='linear')  # Puedes cambiar 'linear' a 'quadratic' o 'cubic' si deseas
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

    # Leer tabla_curvas_i en chunks
    tabla_curvas_i_iter = pd.read_csv(file_tbl, skiprows=2, sep="\s+", chunksize=chunk_size)

    for data_chunk in tabla_curvas_i_iter:
        tabla_curvas_i = data_chunk.T.values  # Transponer y convertir a numpy array
        
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
# def read_datos(file_tbl, file_csv):
#     datos = []
   
   
#     data = pd.read_table(file_tbl)
#     NOMBRE_FICHERO = file_tbl
#     with open(NOMBRE_FICHERO, "r") as f:
#         curva_v = pd.Series([float(x) for x in f.readlines()[1].split()])

    
#     tabla_curvas_i = pd.read_csv(NOMBRE_FICHERO, skiprows=2, sep="\s+").T
#     tabla_curvas_i=tabla_curvas_i.values
#     #curva_v_expanded = increase_sample_size(curva_v, 200)
#     tabla_samplesize = []
#     # for vector in tabla_curvas_i:
#     #     interp = interp1d(np.arange(50), vector)
#     #     vector_ampliado = interp(np.linspace(0, 49, 200))
#     #     tabla_samplesize.append(vector_ampliado)
    
#    # tabla_curvas_i = np.transpose(tabla_curvas_i)
#    # matrix_data = np.transpose(np.array(datos))
#     matrix_pot =[]
#     curva_v.values
#     tabla_samplesize=np.array(tabla_samplesize)
#     for data in tabla_curvas_i:
#          matrix_pot.append( curva_v*data)
   
#     matrix_pot=np.array(matrix_pot)
#     #valores_vmp=curva_v[np.argmax(matrix_pot,axis=1)]
#     indices_vmp=np.argmax(matrix_pot,axis=1)
#     #localizacion del punto maximo

#     data_I_mod=[]
   
#    # valores_vmp=np.repeat(valores_vmp,50)
#    # valores_vmp=valores_vmp.values
#     curva_v_final=[]
#     j=0
#     matrix_pot =[]
#     for curva_I,indices in zip(tabla_curvas_i,indices_vmp):
#         curva_v_exp,curva_I_exp=increase_sample_vmp(indices, curva_v, curva_I)
#         data_I_mod.append(curva_I_exp)
       
#         curva_v_f=np.tile(curva_v_exp,(curva_I_exp.shape[0],1))
#         curva_v_final.append(curva_v_f)
        
#         matrix_pot.append(curva_v_f*curva_I_exp)
      
#         # matrix_pot[:, i] = curva_v[i]*tabla_curvas_i[:, i]
#     #puntos_pmp=np.repeat(puntos_pmp,50)
#     valores_vmp=[]
#     for curva in curva_v_final:
#         valores_vmp.append(curva[np.argmax(matrix_pot,axis=1)])
#     valores_vmp=np.array(valores_vmp)
#     valores_vmp_expandidos = np.concatenate([np.repeat(vmp, len(vec)) for vmp, vec in zip(valores_vmp, curva_v_final)])
    
#     tabla_curvas_i=data_I_mod.flatten()
    
#     curva_v_final=curva_v_final.flatten()
#    #puntos_pmp=np.repeat(puntos_pmp,50)
#     #curva_v=curva_v.values
#    # curva_v=np.tile(curva_v,(tabla_curvas_i.shape[0],1))
#     #tabla_curvas_i=tabla_curvas_i.flatten()

#   #  curva_v=curva_v.flatten()
    
#     return tabla_curvas_i, curva_v,valores_vmp_expandidos
   
def build_model(dataset):
    
    model=Sequential()
    model.add(Input(shape=(dataset.shape[1],)))
   # model.add(Dense(128,activation='tanh'))
    for i in range(5):
    #    model.add(Dense(64,activation='tanh'))
        model.add(Dense(64,activation='sigmoid'))
        #model.add(Dense(64,activation='elu',kernel_regularizer=regularizers.l2(0.008)))
    # for i in range(2):
    #     model.add(Dense(64,activation='sigmoid'))
    # for i in range(3):
    #         model.add(Dense(64,activation='tanh'))
            
    #○model.add(Dense(64,activation='elu'))
    model.add(Dense(1))                  
    return model                  


def increase_sample_vmp(indice,vector,vector_I):
    num_pasos = 20
    ancho_intervalo = min(5, indice, len(vector) - indice - 1)
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


if __name__ == "__main__":
    data_I, data_tension, results_Vmp= read_datos(".\shadowandframeDef.tbl", "shadowandframeDef_pmp.csv")
    
    #Creacion del diccionario de datos 
    data_total=pd.DataFrame({'data_I':data_I,'data_tension':data_tension,'Vmp':results_Vmp})
    data_total=data_total.drop_duplicates()
   # data_total=data_total.groupby('Vmp').head(200)

#vmaxpot=data_total.vmp.drop_duplicates()
    keys=['V','I','Shadow','Vmp']
    # dataset=[None]*2
    # dataset[0]=[data_tension,data_I,shadow_cell]
    # dataset[1]=[data_tension,data_I]
    dataset=pd.DataFrame({'data_I':data_total.data_I,'data_tension':data_total.data_tension})
    # dataset=dataset.loc[vmaxpot.index]
    train_data=dataset.sample(frac=0.8,random_state=0)
    test_data=dataset.drop(train_data.index)
    train_labels=data_total.Vmp[train_data.index]
    test_labels=data_total.Vmp[test_data.index]
    
    model=build_model(train_data)
    OPTIM=keras.optimizers.Nadam()
   # sgd_with_momentum = SGD(learning_rate=0.001, momentum=0.9)

    # Compila tu modelo con el optimizador personalizado
  #  model.compile(optimizer=sgd_with_momentum, loss='mse', metrics=['mse','mae'])
    model.compile(optimizer=OPTIM, loss='mean_squared_error', metrics=['mean_squared_error','mean_absolute_error'])
    
    EPOCHS=250
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=10, min_delta=0.05, mode='min') #callbacks=[early_stop],
    history= model.fit(train_data,train_labels,epochs=EPOCHS,callbacks=[early_stop],validation_split=0.3,verbose=1,batch_size=1024) #,batch_size=120
    
    results = model.evaluate(test_data, test_labels)
    print(results)
    
    model.summary()
