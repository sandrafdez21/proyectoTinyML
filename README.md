# PROYECTO TINY ML

## IDEA PRINCIPAL
Con este proyecto se prentende investigar acerca de las tecnologías TinyML para la implementación de un IA que pueda funcionar en un sistema embebido el cual cuenta con limitaciones de recursos. 
Para tomar un ejemplo, la idea es diseñar un modelo de IA destinado a un RaspBerry Pi Pico. Consistirá en la detección de caracoles, una posible utilidad sería la de evitar una plaga de caracoles en nuestro huerto que se pueda comer nuestros cultivos. Esto es solo un ejemplo pero el objetivo es estudiar su viabilidad para otras posibles implementaciones.

## INSTALACIÓN
Se recomienda la configuración de un entorno virtual de python.
Para ello, se ejecutaría en la Bash:

`python3 -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

En el fichero `requirements.txt` se encuentran todas las dependencias necesarias que hay que instalar para que el código se ejecute correctamente.

## ESTRUCTURA DEL PROYECTO
La estructura del proyecto será la siguiente:
<pre>
.
├── code                                     # Aquí se incluyen los scripts del diseño de los modelos.
│   ├── mobileNetV2.py
│   └── modelosIniciales.py
├── datasets                                 # Este es el dataset empleado, "miniDataset" es un pequeño ejemplo. Para la ejecución de 
│   ├── datasetCompleto.txt                  # del código, lo mejor sería descargar el dataset completo y añadirle a este directorio
│   ├── miniDataset                          # como "dataset".
│   │   ├── noSnail
│   │   └── snail
├── documents                                # Documentos informativos de ciertos procesos llevados a cabo.
│   ├── comparacion_modelos.pdf
│   └── Modelo-MobileNetV2.pdf
├── models                                   # Modelos exportados.
│   ├── modelo_caracoles_base.tflite
│   └── modelo_caracoles_original.keras
├── README.md
├── requirements.txt                         # Dependecias que requiere el código.
└── src
</pre>

## IMPLEMENTACIÓN DEL MODELO
El modelo implementado es una variación de MobileNetV2, que se trata de un modelo preentrenado con miles de fotos de ImageNet y clasificadas entre 1000 categorias distintas.
Para el entreno específico de nuestro propósito, se ha generado un dataset con alrededor de 500 imágenes que contienen caracoles y otras tantas que no lo contienen.
La implementación de este modelo se ha realizado en el script `mobileNetV2.py`, también se incluye en este script una prueba con imágenes nunca antes vistas por el modelo y su exportación en formato `.keras`y `.tflite`.

Adicionalmente, existe un script llamado `modelosIniciales.py` en el que se incluye la implementación de varios modelos que han sido creados con la finalidad de investigar la forma en la que se diseñan los modelos y probar si se podría crear de cero un modelo que funcionase razonablemente bien.

Exiten unos documentos, `comparacion_modelos.pdf` y `Modelo-MobileNetV2.pdf`en los que se explica cuál ha sido el proceso seguido y algunos detalles de ciertas decisiones tomadas.

### USO DE TENSORBOARD
Se ha empleado una herramienta proporcionada por TensorFlow llamada TensorBoard, que nos permite generar gráficos que muestran la evolución de aprendizaje de cada modelo.

Podemos ver los resultados de los modelos implementados en nuestro script ejecutando lo siguiente:

`tensorboard --logdir logs`

Y abrimos en el buscador:

http://localhost:6006/

