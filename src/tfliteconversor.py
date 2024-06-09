# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:01:12 2024

@author: nacho
"""
import tensorflow as tf

ruta_modelo='Modelos/5sigmoidnadaminterpolado_final.keras'
ruta_export='Modelos/5sigmoidfinal'
model=tf.keras.models.load_model(ruta_modelo)
model.export(ruta_export)

converter=tf.lite.TFLiteConverter.from_saved_model(ruta_export)
tflite_model=converter.convert()

with open('Modelos/convert_5sigmoid.tflite','wb') as f:
        f.write(tflite_model)

# print(f'Model converted and saved to {tflite_model_path}')