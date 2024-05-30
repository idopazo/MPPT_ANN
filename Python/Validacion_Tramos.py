# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:18:27 2024

@author: nacho
"""


from scipy.interpolate import interp1d


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.optimizers import SGD 

from tensorflow import keras
from keras import regularizers
from keras.models import load_model
from sklearn.metrics import mean_absolute_error

import time
def mppt_predict(model,datov,dato_I):
    entrada_modelo = np.array([[dato_I,datov]])
    vmp = model.predict(entrada_modelo)
    return vmp

def interpolar(array_entrada,array_salida, valor):
    f_inter=interp1d(array_entrada,array_salida)
    
    interpolado=f_inter(valor)
    
    return interpolado

def increase_sample_vmp(indice,vector,vector_I):
    num_pasos=50
    if indice- 5 >= 0 and indice + 5 < len(vector):
    # Obtener los valores +- 5 posiciones
        ancho_intervalo=5
        valor_menos_5 = vector[indice - ancho_intervalo]
        valor_mas_5 = vector[indice + ancho_intervalo]
        valores_intervalo_I = vector_I[indice - ancho_intervalo:indice + ancho_intervalo+1]

    # Crear una función de interpolación
        x_original = np.linspace(0, 1, num=len(valores_intervalo_I))
        x_interpolado = np.linspace(0, 1, num=num_pasos)
        f_interpolacion = interp1d(x_original, valores_intervalo_I)  # Puedes cambiar 'linear' por 'quadratic' o 'cubic'
    # Obtener los valores interpolados
        valores_interpolados = f_interpolacion(x_interpolado)
  
    else :
        ancho_intervalo=2
        valor_menos_5 = vector[indice - ancho_intervalo]
        valor_mas_5 = vector[indice + ancho_intervalo]      
        vector_lineal = np.linspace(valor_menos_5, valor_mas_5, num_pasos)
        x_original = np.linspace(0, 1, num=len(valores_intervalo_I))
        x_interpolado = np.linspace(0, 1, num=num_pasos)
        f_interpolacion = interp1d(x_original, valores_intervalo_I)  # Puedes cambiar 'linear' por 'quadratic' o 'cubic'
# Obtener los valores interpolados
        valores_interpolados = f_interpolacion(x_interpolado)
# Concatenar los tres segmentos
    segmento_1 = vector[:indice - ancho_intervalo-1]  # Hasta valor menos 5
    segmento_2 = vector_lineal  # De valor menos 5 a valor mas 5
    segmento_3 = vector[indice + ancho_intervalo+1:]  # Desde valor mas 5 hasta el final
  
 
    segmento_1_I = vector_I[:indice - ancho_intervalo-1]  # Hasta valor menos 5
    segmento_2_I = valores_interpolados  # De valor menos 5 a valor mas 5
    segmento_3_I = vector_I[indice + ancho_intervalo+1:]  # Desde valor mas 5 hasta el final

# Unir los segmentos en un solo vector
    vector_completo_v = np.concatenate((segmento_1, segmento_2, segmento_3))
    vector_I_ampliado=np.concatenate((segmento_1_I, segmento_2_I, segmento_3_I))
    return vector_completo_v, vector_I_ampliado
    

def increase_sample_size(vector, new_size):
    x_old = np.linspace(0, 1, num=len(vector))
    x_new = np.linspace(0, 1, num=new_size)
    f = interp1d(x_old, vector, kind='linear')  # Puedes cambiar 'linear' a 'quadratic' o 'cubic' si deseas
    return f(x_new)

def read_datos(file_tbl, file_csv):
    datos = []
    puntos_pmp = pd.read_csv(file_csv,usecols=[1])
    puntos_pmp = np.array(puntos_pmp)
   
    data = pd.read_table(file_tbl)
    NOMBRE_FICHERO = file_tbl
    with open(NOMBRE_FICHERO, "r") as f:
        curva_v = pd.Series([float(x) for x in f.readlines()[1].split()])

    interp = interp1d(np.arange(50), curva_v)
   # curva_v= interp(np.linspace(0, 49, 200))
    tabla_curvas_i = pd.read_csv(NOMBRE_FICHERO, skiprows=2, sep="\s+").T
    tabla_curvas_i=tabla_curvas_i.values
    curva_v_expanded = increase_sample_size(curva_v, 200)
    data_I_ampliada = []
    for vector in tabla_curvas_i:
        interp = interp1d(np.arange(50), vector)
        vector_ampliado = interp(np.linspace(0, 49, 200))
        data_I_ampliada.append(vector_ampliado)
   # tabla_curvas_i = np.transpose(tabla_curvas_i)
   # matrix_data = np.transpose(np.array(datos))
    matrix_pot =[]

   # for vector in data_I_ampliada:
    for vector in data_I_ampliada:
        matrix_pot.append(curva_v_expanded*vector)
        
   
       
   #localizacion del punto maximo
    puntos_pmp=np.max(matrix_pot,axis=1)
    #localizacion del punto maximo
    valores_vmp=curva_v_expanded[np.argmax(matrix_pot,axis=1)]
   
  
   
  
    #curva_v=curva_v.flatten()
    
    return puntos_pmp, data_I_ampliada,valores_vmp,curva_v_expanded
#%%   

prueba_pmp, vectores_data_I, pruebaresults_Vmp,curvavsample= read_datos(".\shadowandframeDef.tbl", "shadowandframeDef_pmp.csv")
#%%
start_time=time.time()
#dinamico
#indicespredicciones=range(43000,43300)
#bultitos
indicespredicciones=range(1800,2400)
# bultitos
# indicespredicciones=range(71700,72800)
#estatico
#indicespredicciones=range(0,1800)
 #2 mins
#indicespredicciones=range(0,20000) #2 mins
indicespredicciones=range(28310,33310)
#indicespredicciones=range(210500,214500)
model=keras.models.load_model('Modelos/5sigmoidnadaminterpolado_final.keras')
data_V=curvavsample
vectores_data_I=np.array(vectores_data_I)

data_I=vectores_data_I[indicespredicciones]
#data_I=vectores_data_I
predicciones1 = []
predicciones2=[]
array_verdadero = []
vpredicha1=[]
vpredicha2=[]
indice_aleatorio =120
primerav=data_V[indice_aleatorio]
primerai=data_I[0][indice_aleatorio]

entrada_modelo = np.array([[primerai, primerav]])

prediccion_anteriorV = model.predict(entrada_modelo)
resultado_interpolacion = interpolar(data_V, data_I[0], prediccion_anteriorV)
# Almacenar la predicción y el valor verdadero de VMP
predicciones1.append(primerav*primerai)
vpredicha1.insert(0,primerav)
for muestra_i in data_I[1:]:
        # Predicción con el modelo de Keras    
        resultado_interpolacion = interpolar(data_V, muestra_i, prediccion_anteriorV)
        
       # entrada_modelo = np.array([[resultado_interpolacion[0][0],prediccion_anteriorV[0][0]]])
        prediccion_anteriorV = mppt_predict(model, prediccion_anteriorV[0][0], resultado_interpolacion[0][0])
        resultado_interpolacion = interpolar(data_V, muestra_i, prediccion_anteriorV)
        
        predicciones1.append(resultado_interpolacion[0][0]*prediccion_anteriorV[0][0])
        # Agregar las predicciones a la lista de predicciones
        vpredicha1.append(prediccion_anteriorV[0])
        
endtime=time.time()
executiontime=endtime-start_time
array_verdadero=prueba_pmp[indicespredicciones]  
predarray=np.array(predicciones1)

x_values = [indicespredicciones[0] + i for i in range(len(predicciones1))]
plt.figure()
plt.plot(x_values, predicciones1,label="Predicciones",linewidth="0.9")
plt.xlabel('Instante')

plt.ylabel('Potencia[W] ')
plt.legend()
plt.title('Potencia')
plt.show()

plt.plot(x_values, array_verdadero,label="Real",linewidth="0.9")
plt.xlabel('Instante')
plt.ylabel('Potencia[W] ')
plt.legend()
plt.title('Potencia')
plt.show()
energiareal1=sum(array_verdadero*0.004)
eff=sum(predicciones1)/sum(array_verdadero)
print(f'eff={eff:.1%}')
#%%

plt.figure()
plt.plot(range(len(predicciones1)), vpredicha1,label="Predicciones",linewidth="0.9")
plt.xlabel('Instante')

plt.ylabel('Tensión[V] ')
plt.legend()
plt.title('Tensión')
plt.show()

plt.plot(range(len(pruebaresults_Vmp[indicespredicciones])), pruebaresults_Vmp[indicespredicciones],label="Real",linewidth="0.9")
plt.xlabel('Instante')
plt.ylabel('Tensión[V] ')
plt.legend()
plt.title('Tensión')
plt.show()
#%%
maev1 = mean_absolute_error(pruebaresults_Vmp[indicespredicciones], vpredicha1)
meapv1=np.mean( np.abs((pruebaresults_Vmp[indicespredicciones]-vpredicha1)/pruebaresults_Vmp[indicespredicciones]))

mae1 = mean_absolute_error(array_verdadero, predicciones1)
meap1=np.mean( np.abs((array_verdadero-predarray)/array_verdadero))

#%%

energiareal1=sum(array_verdadero*0.004)
eff=sum(predicciones1)/sum(array_verdadero)
print(f'eff={eff:.1%}')
energiareal1=energiareal1/3600
energiaproducido1=sum(predicciones1)*0.004
energiaproducido1=energiaproducido1/3600
#%%
data_I=vectores_data_I[indicespredicciones2]
indice_aleatorio = 35
primerav=data_V[indice_aleatorio]
primerai=data_I[0][indice_aleatorio]
entrada_modelo = np.array([[primerai, primerav]])

prediccion_anteriorV = model.predict(entrada_modelo)
resultado_interpolacion = interpolar(data_V, data_I[0], prediccion_anteriorV)
# Almacenar la predicción y el valor verdadero de VMP

predicciones2.append(primerav*primerai)

for muestra_i in data_I[1:]:
        # Predicción con el modelo de Keras    
        resultado_interpolacion = interpolar(data_V, muestra_i, prediccion_anteriorV)
        prediccion_anteriorV = mppt_predict(model, prediccion_anteriorV[0][0], resultado_interpolacion[0][0])
        resultado_interpolacion = interpolar(data_V, muestra_i, prediccion_anteriorV)
        
        predicciones2.append(resultado_interpolacion[0][0]*prediccion_anteriorV[0][0])
        # Agregar las predicciones a la lista de predicciones
        vpredicha2.append(prediccion_anteriorV[0])
    # Agregar el valor verdadero de VMP a la lista de valores verdaderos

array_verdadero=prueba_pmp[indicespredicciones2]  
predarray=np.array(predicciones2)
x_values = [indicespredicciones2[0] + i for i in range(len(predicciones2))]
#%%
plt.figure()
plt.plot(x_values, predicciones2,label="Predicciones",linewidth="0.9")
plt.xlabel('Instante')

plt.ylabel('Potencia[W] ')
plt.legend()
plt.title('Potencia')
plt.show()

plt.plot(x_values, array_verdadero,label="Real",linewidth="0.9")
plt.xlabel('Instante')
plt.ylabel('Potencia[W] ')
plt.legend()
plt.title('Potencia')
plt.show()



vpredicha2.insert(0,vpredicha2[0])
plt.figure()
plt.plot(x_values, vpredicha2,label="Predicciones",linewidth="1")
plt.xlabel('Instante')

plt.ylabel('Tensión[V] ')
plt.legend()
plt.title('Tensión')
plt.show()

plt.plot(x_values, pruebaresults_Vmp[indicespredicciones2],label="Real",linewidth="1")
plt.xlabel('Instante')
plt.ylabel('Tensión[V] ')
plt.legend()
plt.title('Tensión')
plt.show()
#%%
maev2 = mean_absolute_error(pruebaresults_Vmp[indicespredicciones2], vpredicha2)
meapv2=np.mean( np.abs((pruebaresults_Vmp[indicespredicciones2]-vpredicha2)/pruebaresults_Vmp[indicespredicciones2]))
energiareal2=sum(array_verdadero*0.004)
energiareal2=energiareal2/3600

energiaproducido2=sum(predarray*0.004)
energiaproducido2=energiaproducido2/3600
eff2=energiaproducido2/energiareal2
mae2 = mean_absolute_error(array_verdadero, predicciones2)
meap2=np.mean( np.abs((array_verdadero-predarray)/array_verdadero))