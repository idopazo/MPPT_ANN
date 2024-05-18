# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:24:53 2024

@author: Ruben
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    
    if MUESTRA_PUNTOS_IV:
        titulo = f'{frame_ini}-{frame_fin}_factor_submuestreo={factor_submuestreo} paso_v={paso_v}_curvasIV'
        ax.set_title(titulo)
        plt.savefig(titulo + '.png')
    
    #%%
    pot_mod = pd.read_csv('shadowandframeDef_pmp.csv', index_col=[0])['0'].iloc[frame_ini:frame_fin:1]
    
    eff = pot_po.sum() / pot_mod.sum()
    print(f'eff={eff:.1%}')

    if GENERA_GRAFICA_POTENCIA:
        fig, ax = plt.subplots()
        pot_po.plot(ax=ax, style='.-')
        pot_po[::factor_submuestreo].plot(ax=ax, style='.')
        pot_mod.plot(ax=ax)
        
        titulo = f'{frame_ini}-{frame_fin}_factor_submuestreo={factor_submuestreo} paso_v={paso_v} eff={eff:.1%}'
        # ax.set_title(titulo)
        ax.set_ylim([0, 60])
        
        plt.savefig(titulo + '.png')
    
    return eff, pot_po, v1_po

#%%
# FULL
# frame_ini = 0
# frame_fin = len(tabla_curvas_i)

# 2 MINUTOS
frame_ini = 0
frame_fin = 30000

# 4 MINUTOS
# frame_ini = 0
# frame_fin = 60000

# STATIC
# frame_ini = 0
# frame_fin = 1800

# REALISTIC
# frame_ini = 1800
# frame_fin = 2400

# DINAMICO1
# frame_ini = 34100
# frame_fin = 34500

# DINAMICO2
# frame_ini = 43000
# frame_fin = 43300

# frame_ini = 502
# frame_fin = 625

# bultitos
# frame_ini = 71700
# frame_fin = 72800

# bultitos
# frame_ini = 87700
# frame_fin = 88500

# bultitos
# frame_ini = 103000
# frame_fin = 103500

# frame_ini = 83100
# frame_fin = 84800

# frame_ini = 87700
# frame_fin = 89000

# frame_ini = 151000
# frame_fin = 158000

# frame_ini = 138500
# frame_fin = 139700

# bultitos
# frame_ini = 141000
# frame_fin = 144000

# frame_ini = 146500
# frame_fin = 150000

# frame_ini = 151000
# frame_fin = 158000

# bultitos
# frame_ini = 153000
# frame_fin = 155000

# frame_ini = 177150
# frame_fin = 178200

# frame_ini = 178600
# frame_fin = 179250

# bultitos
# frame_ini = 178600
# frame_fin = 178950

# frame_ini = 180550
# frame_fin = 180600

# frame_ini = 213800
# frame_fin = 217500

# frame_ini = 219000
# frame_fin = 220000

# frame_ini = 250000
# frame_fin = 254500

#%%
d_eff = []

for paso in np.arange(start=0.1, stop=4, step=1):
    for factor in np.arange(start=1, stop=100, step=20):
        eff,_,_ = test_po(frame_ini, frame_fin, paso_v=paso, factor_submuestreo=factor)
        print(f'{frame_ini}-{frame_fin}_factor_submuestreo={factor} paso_v={paso} eff={eff*100:.1f}%')
        d_eff.append([paso, factor, eff])

datos_eff = pd.DataFrame(columns=['paso_v', 'factor_submuestreo', 'eff'], data=d_eff)

print('Punto fijo:')
eff_v_fijo, pot_po_fijo, v1_po_fijo = test_po(frame_ini, frame_fin, paso_v=1, factor_submuestreo=1, v_fijo=True)

#%%
fig, ax = plt.subplots(figsize=(10, 6))

cs = ax.tricontourf(datos_eff.paso_v, datos_eff.factor_submuestreo, datos_eff.eff, cmap='Oranges', levels=20)
cb = plt.colorbar(cs, format=lambda x, _: f"{x:.0%}", label='$\eta_{MPPT}$')
cb.ax.set_ylabel('$\eta_{MPPT}$', fontsize=16)
cb.ax.tick_params(labelsize=16)
# cb = ax.scatter(datos_eff.paso_v, datos_eff.factor_submuestreo, c=datos_eff.eff, cmap='viridis')
# plt.colorbar(cb)

ax.set_xlabel('$\Delta V_{op}$ [V]', fontsize=16)
ax.set_ylabel('$T_{per}$ [s]', fontsize=16)

# ax.set_yticks(np.append(1, ax.get_yticks()[:-1]))

ax.set_yticklabels([f'{x:.2}' for x in ax.get_yticks() * 1/240], fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)

titulo = f'Efficiency - Frames {frame_ini}-{frame_fin} - eff_pto_fijo {eff_v_fijo:.1%}'
# ax.set_title(titulo)

plt.savefig(titulo + '.png')

fig2, ax2 = plt.subplots()
pd.read_csv('shadowandframeDef_pmp.csv', index_col=[0])['0'].iloc[frame_ini:frame_fin:1].plot(ax=ax2)
plt.savefig(titulo + '_pot.png')

#%%
eff, pot_po, v1_po = test_po(frame_ini, frame_fin, paso_v=0.1, factor_submuestreo=1)
pot_mod = pd.read_csv('shadowandframeDef_pmp.csv', index_col=[0])['0']

#%%
ax=pot_mod.plot()
pot_po.plot(ax=ax)
(pot_po / pot_mod).plot(ax=ax)

#%%
if False:
    fig, ax = plt.subplots()
    
    pd.read_csv('shadowandframeDef_pmp.csv', index_col=[0])['0'].iloc[frame_ini:frame_fin:1].plot(label='pmp', legend=True)
    eff_v, pot_po, v1 = test_po(frame_ini, frame_fin, paso_v=1, factor_submuestreo=1);pot_po.plot(label='1', legend=True)
    eff_v, pot_po, v1 = test_po(frame_ini, frame_fin, paso_v=1, factor_submuestreo=2);pot_po.plot(label='1', legend=True)
    eff_v, pot_po, v1 = test_po(frame_ini, frame_fin, paso_v=1, factor_submuestreo=10);pot_po.plot(label='10', legend=True)
    eff_v, pot_po, v1 = test_po(frame_ini, frame_fin, paso_v=1, factor_submuestreo=100);pot_po.plot(label='100', legend=True)
