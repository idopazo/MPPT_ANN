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

def read_datos(file_tbl, file_csv):
    datos = []

    data = pd.read_table(file_tbl)
    NOMBRE_FICHERO = file_tbl
    with open(NOMBRE_FICHERO, "r") as f:
        curva_v = pd.Series([float(x) for x in f.readlines()[1].split()])

    interp = interp1d(np.arange(50), curva_v)
    curva_v= interp(np.linspace(0, 49, 200))
    tabla_curvas_i = pd.read_csv(NOMBRE_FICHERO, skiprows=2, sep="\s+").T
    tabla_curvas_i=tabla_curvas_i.values
    data_I_ampliada = []
    
    for vector in tabla_curvas_i:
        interp = interp1d(np.arange(50), vector)
        vector_ampliado = interp(np.linspace(0, 49, 200))
        data_I_ampliada.append(vector_ampliado)
   # tabla_curvas_i = np.transpose(tabla_curvas_i)
   # matrix_data = np.transpose(np.array(datos))
    matrix_pot =[]

    for vector in data_I_ampliada:
         matrix_pot.append(curva_v*vector)
         
    
        
    #localizacion del punto maximo
    puntos_pmp=np.max(matrix_pot,axis=1)
    puntos_pmp=np.repeat(puntos_pmp,200)
    valores_vmp=curva_v[np.argmax(matrix_pot,axis=1)]
    valores_vmp=np.repeat(valores_vmp,200)
    curva_v=np.tile(curva_v,np.array(tabla_curvas_i).shape[0])

    tabla_curvas_i=[elemento for sublista in data_I_ampliada for elemento in sublista]
    
    
    tabla_curvas_i=np.array(tabla_curvas_i)
    # curva_v= np.transpose(curva_v)
    # curva_v=curva_v.flatten()
    #curva_v=curva_v.flatten()
    
    return puntos_pmp, tabla_curvas_i,curva_v,valores_vmp
   
def build_model(dataset):
    
    model=Sequential()
    model.add(Input(shape=(dataset.shape[1],)))

    for i in range(5):
        model.add(Dense(64,activation='relu'))
       # model.add(Dense(64,activation='elu',kernel_regularizer=regularizers.l1(0.005)))
    # for i in range(2):
    #     model.add(Dense(64,activation='sigmoid'))
    # for i in range(3):
    #         model.add(Dense(64,activation='tanh'))
            
    #○model.add(Dense(64,activation='elu'))
    model.add(Dense(1))                  
    return model                  

if __name__ == "__main__":
    puntos_pmp, data_I, data_tension, results_Vmp= read_datos(".\shadowandframeDef.tbl", "shadowandframeDef_pmp.csv")
    
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
    train_data=dataset.sample(frac=0.6,random_state=0)
    test_data=dataset.drop(train_data.index)
    train_labels=data_total.Vmp[train_data.index]
    test_labels=data_total.Vmp[test_data.index]
    
    model=build_model(train_data)
    OPTIM=keras.optimizers.Adam()
   # sgd_with_momentum = SGD(learning_rate=0.001, momentum=0.9)

    # Compila tu modelo con el optimizador personalizado
  #  model.compile(optimizer=sgd_with_momentum, loss='mse', metrics=['mse','mae'])
    model.compile(optimizer=OPTIM, loss='mean_squared_error', metrics=['mean_squared_error','mean_absolute_error'])
    
    EPOCHS=250
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=10, min_delta=0.05, mode='min') #callbacks=[early_stop],
    history= model.fit(train_data,train_labels,epochs=EPOCHS,callbacks=[early_stop],validation_split=0.3,verbose=1,batch_size=500) #,batch_size=120
    
    results = model.evaluate(test_data, test_labels)
    print(results)
    
    model.summary()
