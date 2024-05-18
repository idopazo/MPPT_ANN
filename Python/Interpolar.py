# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:13:25 2024

@author: nacho
"""
from scipy.interpolate import interp1d


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.optimizers import SGD 

from tensorflow import keras
from keras import regularizers
from keras.models import load_model

def interpolar(array_entrada,array_salida, valor):
    f_inter=interp1d(array_entrada,array_salida)
    
    interpolado=f_inter(valor)
    
    return interpolado



def read_datos(file_tbl, file_csv):
    datos = []
    puntos_pmp = pd.read_csv(file_csv,usecols=[1])
    puntos_pmp = np.array(puntos_pmp)
   
    data = pd.read_table(file_tbl)
    NOMBRE_FICHERO = file_tbl
    with open(NOMBRE_FICHERO, "r") as f:
        curva_v = pd.Series([float(x) for x in f.readlines()[1].split()])

    
    tabla_curvas_i = pd.read_csv(NOMBRE_FICHERO, skiprows=2, sep="\s+").T
    tabla_curvas_i=tabla_curvas_i.values
   # tabla_curvas_i = np.transpose(tabla_curvas_i)
   # matrix_data = np.transpose(np.array(datos))
    matrix_pot = np.zeros_like(tabla_curvas_i)
  
    for i in range(curva_v.size):
         matrix_pot[:, i] = curva_v[i]*tabla_curvas_i[:, i]
    
        
    #localizacion del punto maximo
    valores_vmp=curva_v[np.argmax(matrix_pot,axis=1)]
   
    valores_vmp=valores_vmp.values
    
    curva_v=curva_v.values
   
   
   
  
    #curva_v=curva_v.flatten()
    
    return puntos_pmp, tabla_curvas_i,valores_vmp,curva_v
   
prueba_pmp, vectores_data_I, pruebaresults_Vmp,curvavsample= read_datos(".\shadowandframeDef.tbl", "shadowandframeDef_pmp.csv")

indicespredicciones=range(210500,214500)

model=keras.models.load_model('1tanh1elutodoslosdatosNadam.keras')
data_V=curvavsample
data_I=vectores_data_I[indicespredicciones]

predicciones = []
array_verdadero = []
vpredicha=[]

for muestra_i in data_I:
    # Seleccionar 5 índices aleatorios de data_I
    indices = np.arange(0, 50, 10)  # Ejemplo: seleccionar 5 índices aleatorios
    indices = sorted(indices)  # Ordenar los índices para asegurar la correspondencia

    # Seleccionar los valores correspondientes de data_I y data_V
    subconjunto_i = muestra_i[indices]
    subconjunto_v = data_V[indices]

    # Formatear los datos para el modelo de Keras
    
      # Formato requerido por el modelo
    for i,v in zip (subconjunto_i,subconjunto_v):
        # Predicción con el modelo de Keras
        entrada_modelo = np.array([[i, v]])
        prediccion = model.predict(entrada_modelo)
        # Agregar las predicciones a la lista de predicciones
        vpredicha.append(prediccion[0])
        resultado_interpolacion = interpolar(data_V, muestra_i, prediccion)
    # Almacenar la predicción y el valor verdadero de VMP
        predicciones.append(resultado_interpolacion*prediccion)
   

  

    # Agregar el valor verdadero de VMP a la lista de valores verdaderos
    array_verdadero=prueba_pmp[indicespredicciones]  # Supongamos que el valor verdadero de VMP es el máximo de los valores de muestra_v

array_verdadero= np.repeat(array_verdadero, 5)
predicciones=np.concatenate(predicciones)
plt.plot(range(len(predicciones)), predicciones,label="predicciones")
plt.xlabel('Instante')

plt.ylabel('Potencia[W] maximas')
plt.legend()
plt.title('Potencia')
plt.show()

plt.plot(range(len(array_verdadero)), array_verdadero,label="Real")
plt.xlabel('Instante')
plt.ylabel('Potencia[W] maximas')
plt.legend()
plt.title('Potencia')
plt.show()