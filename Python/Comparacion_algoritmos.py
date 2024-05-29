# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:51:30 2024

@author: nacho
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import scipy.io

from scipy.interpolate import interp1d

V_INICIAL = 17.7
def po(p1, p0, v0, v1, delta_v):
    if p1 == p0:
        v2 = v1
        if p0 == 0:
            v2 = V_INICIAL # sale de enganches a pot=0
    elif p1 > p0:
        if v1 > v0:
            v2 = v1 + delta_v
        else:
            v2 = v1 - delta_v
    else:
        if v1 < v0:
            v2 = v1 + delta_v
        else:
            v2 = v1 - delta_v
            
    return v2


MUESTRA_PUNTOS_IV = False
GENERA_GRAFICA_POTENCIA = False

# NOMBRE_FICHERO = 'n1_n000_shadowandframeDef_paso1_chunk_0-30000_paso1_chunk_1500-3000.tbl'
# NOMBRE_FICHERO = 'n000_shadowandframeDef_paso1_chunk_0-30000.tbl'
# NOMBRE_FICHERO = 'n0_shadowandframeDef_paso1_chunk_0-60000.tbl'
NOMBRE_FICHERO = 'shadowandframeDef.tbl'

with open(NOMBRE_FICHERO, "r") as f:
    curva_v = pd.Series([float(x) for x in f.readlines()[1].split()])

tabla_curvas_i = pd.read_csv(NOMBRE_FICHERO, skiprows=2, sep="\s+").T
    
def test_po(frame_ini, frame_fin, paso_v, factor_submuestreo, v_fijo=False):

    if MUESTRA_PUNTOS_IV:
        fig, ax = plt.subplots()
    
    v0 = 0
    p0 = 0
    v1 = V_INICIAL
    p_tot = []
    v_tot = []
    
    for idx, curva_i in tabla_curvas_i.iloc[frame_ini:frame_fin].iterrows():
    
        i1 = np.interp(x=v1, xp=curva_v, fp=curva_i, right=0)# fuerza a que i1 no sea < 0 -> p no <0
        p1 = i1 * v1 
        
        if int(idx) % factor_submuestreo == 0: # actualiza v
            if v_fijo:
                v2 = V_INICIAL
            else:
                v2 = po(p1=p1, p0=p0, v0=v0, v1=v1, delta_v=paso_v)
                v2 = v2 if v2 > 0 else 0.001 # fuerza a que v2 no sea < 0 -> p no <0
            
            p0 = p1
            v0 = v1
            v1 = v2
            
        else:
            v2 = v1
        
        if MUESTRA_PUNTOS_IV:
            print(idx, v0, v1, v2, p1)
            ax.plot(curva_v, curva_i)
            ax.plot(v1, i1, 'o')
    
    
        p_tot += [p1]
        v_tot += [v1]
    
    pot_po = pd.Series(index=range(frame_ini, frame_fin), data=p_tot)
    v1_po = pd.Series(index=range(frame_ini, frame_fin), data=v_tot)

    pot_mod = pd.read_csv('shadowandframeDef_pmp.csv', index_col=[0])['0'].iloc[frame_ini:frame_fin:1]
    pot_ann = pd.read_csv('predicciones1.csv', header=None).squeeze()
    pot_ann=pot_ann.iloc[frame_ini:frame_fin:1]
    eff = pot_po.sum() / pot_mod.sum()
    print(f'eff={eff:.1%}')

    if GENERA_GRAFICA_POTENCIA:
        fig, ax = plt.subplots()
        pot_po.plot(ax=ax, style='.-')
        pot_po[::factor_submuestreo].plot(ax=ax, style='.')
        pot_mod.plot(ax=ax)
        
        titulo = f'{frame_ini}-{frame_fin}_factor_submuestreo={factor_submuestreo} paso_v={paso_v} eff={eff:.1%}'
    
        
        plt.savefig(titulo + '.png')
    
    return eff, pot_po, v1_po,pot_mod,pot_ann
def increase_sample_size(vector, new_size):
    x_old = np.linspace(0, 1, num=len(vector))
    x_new = np.linspace(0, 1, num=new_size)
    f = interp1d(x_old, vector, kind='linear')  # Puedes cambiar 'linear' a 'quadratic' o 'cubic' si deseas
    return f(x_new)
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
tramos = [
    (0, 20000, 0.1),
    (0, 20000, 1.1),
    (0, 1800, 0.1),
    (28850, 29300, 0.1),
    (12500, 13100, 1.1),
    (1800, 2400, 0.1),
    (28310, 33310, 0.1),
    (28310, 33310, 1.1),
    (43000, 43300, 0.1),
    (43000, 43300, 2.1),
    (71700, 72800, 0.1),
    (71700, 72800, 1.1),
    (210500, 214500, 0.1)
]
# Obtener el tamaño de la pantalla
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# Convertir el tamaño de la pantalla a pulgadas (matplotlib usa pulgadas)
dpi = 100
fig_width = screen_width / dpi
fig_height = screen_height / dpi
# Lista para almacenar los vectores de potencia P&O
pot_po_list = []
pot_mod_list=[]

prueba_pmp, vectores_data_I, pruebaresults_Vmp,curvavsample= read_datos(".\shadowandframeDef.tbl", "shadowandframeDef_pmp.csv")
# Iterar sobre cada tramo y calcular las potencias P&O
for (frame_ini, frame_fin, delta_v) in tramos:
    eff_v, pot_po, v1,pot_mod,pot_ann = test_po(frame_ini, frame_fin, paso_v=delta_v, factor_submuestreo=1)
    eff_v_fijo, pot_po_fijo, v1_po_fijo,pot_mod,pot_an= test_po(frame_ini, frame_fin, paso_v=1, factor_submuestreo=1, v_fijo=True)
    pot_po_list.append(pot_po)
    # Aquí puedes incluir la lógica para crear las gráficas como se mostró anteriormente
    # Por ejemplo, puedes leer los datos reales y predichos como en el código anterior

    # Crear el gráfico (ajustar según tu lógica de datos reales y predichos)
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(range(frame_ini, frame_fin), prueba_pmp[frame_ini: frame_fin],label='Potencia Real',linewidth="0.9")  # Ajustar según tus datos
    plt.plot(range(frame_ini, frame_fin), pot_ann, label='ANN',linewidth="0.9")  # Ajustar según tus datos
    plt.plot(range(frame_ini, frame_fin), pot_po, label=f'P&OΔV={delta_v}V  ',linewidth="0.9")
    plt.plot(range(frame_ini, frame_fin), pot_po_fijo,label='Punto Fijo',linewidth="0.9")  # Ajustar según tus datos
    
    plt.xlabel('Instante')
    plt.ylabel('Potencia [W]')
    plt.title(f'Potencias Tramo {frame_ini}-{frame_fin} ')
    plt.legend()
   
    plt.show()
            
    titulo = f'{frame_ini}-{frame_fin}paso_v={delta_v}'
    
    plt.savefig(titulo + '.png')
    eff_pot_po_fijo = pot_po_fijo[frame_ini:frame_fin:1].sum() / prueba_pmp[frame_ini:frame_fin:1].sum()

    # Calcula eff utilizando pot_ann en lugar de pot_mod
    eff_pot_po = pot_po[frame_ini:frame_fin:1].sum() / prueba_pmp[frame_ini:frame_fin:1].sum()
    eff_pot_ann=pot_ann[frame_ini:frame_fin:1].sum() / prueba_pmp[frame_ini:frame_fin:1].sum()
    # Imprime los valores de eff
    print("tramo:", frame_ini,frame_fin)
    print("eff utilizando pot_po_fijo:", eff_pot_po_fijo)
    print("eff utilizando pot_po:", eff_pot_po)
    print("eff utilizando pot_ann:", eff_pot_ann)
    