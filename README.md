# MPPT ANN

## Descripción del Proyecto

Este proyecto tiene como objetivo la obtención de un algoritmo MPPT basado en una ANN. EN este respositorio se encuentra el código necesario para realizar el entrenamiento y la validación, así como diferentes modelos ya creados. A parte, se incluye el modelo con el que se obtuvieron los mejores resultados de eficiencia convertido al formato TFLite, listo para usarse en un microcontrolador o dispositivo portátil

## Contenido del Repositorio

- En 'main/src/' Se encuentra el código fuente de los diferentes programas para el desarrollo del método
- En 'main/src/Modelos/' Se encuentran una serie de Modelos realizados
- El programa 'Training_19mindata.py' Se encarga de la creación y entrenamiento del modelo
- En 'Validacion_Tramos.py' Se presenta la posibilidad de validar la red desarrollada haciendo un seguimiento del MPP en tramos diferenciados
- En 'Comparacion_algoritmos.py' Se compara el funcionamiento de la ANN con el algoritmo P&O y fijar un valor de tensión fijo de polarización
- '5sigmoidnadaminterpolado_final.keras' Es el archivo de la ANN con el que se han obtenido mejores resultados.
- 'convert_5sigmoid.tflite' Es el archivo de la ANN con el que se han obtenido mejores resultados en un formato válido para ser usado en microcontroladores o dispositivos portátiles
