# PROYECTO TINY ML

## IDEA PRINCIPAL
Con este proyecto se prentende investigar acerca de las tecnologías TinyML para la implementación de un IA que pueda funcionar en un sistema embebido el cual cuenta con limitaciones de recursos. 
Para tomar un ejemplo, la idea es diseñar un modelo de IA destinado a un RaspBerry Pi Pico. Consistirá en la detección de caracoles, una posible utilidad sería la de evitar una plaga de caracoles en nuestro huerto que se pueda comer nuestros cultivos. Esto es solo un ejemplo pero el objetivo es estudiar su viabilidad para otras posibles implementaciones.

## INSTALACIÓN Y MATERIALES PREVIOS
Se recomienda la configuración de un entorno virtual de python.
Para ello, se ejecutaría en la Bash:

`python3 -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

En el fichero `requirements.txt` se encuentran todas las dependencias necesarias que hay que instalar para que el código se ejecute correctamente.

Adicionalmente, será necesaria una Raspberry Pi Pico con el RP2040, la cual nos generará las principales limitaciones de este proyecto que son el espacio en memoria Flash (2MB) y en memoria RAM (264 KBytes)

## ESTRUCTURA DEL PROYECTO
La estructura del proyecto será la siguiente:
<pre>
.
├── code                            // Directorio en el que se encuentra codigo
│   │                               // para la creacion y cuantizacion de los modelos
│   │                               // Tambien hay pruebas de la raspberry pico
│   ├── creacionModelos
│   │   ├── mobileNetV2.py
│   │   └── modelosIniciales.py
│   ├── cuantizacion
│   │   ├── cuantizacion.py
│   │   └── pruebaCuantizado.py
│   └── pruebasRaspberry
│       ├── prueba1
│       ├── prueba2
│       └── prueba3
├── datasets                        // Utilizado para el entrenamiento del modelo
│   ├── conversion_imagenes.py
│   ├── conversion_imagenes_byn.py
│   ├── dataset
│   │   ├── noSnail
│   │   └── snail
│   ├── datasetCompleto.txt
│   ├── miniDataset
│   │   ├── noSnail
│   │   └── snail
│   └── tests
│       ├── noSnail
│       └── snail
├── documents                       // Documentos donde se explican los procesos realizados
│   ├── 01_modelosIniciales.pdf
│   ├── 02_modeloMobileNetV2.pdf
│   ├── 03_metricasMobileNetV2.pdf
│   ├── 04_cuantizacion.pdf
│   ├── 05_implementacionEnPico.pdf
│   └── 06_pruebasRaspberry.pdf
├── logs                            // Generado para observar el funcionamiento de los modelos
├── models                          // Se guarda los modelos, también los lite y los cuantizados
│   ├── modelo1
│   ├── modelo2
│   ├── modelo3
│   ├── modeloMobileNetV2_224x224
│   └── modeloMobileNetV2_96x96
├── README.md
├── requirements.txt                // Software requerido para el proyecto
├── src                             // Codigo c++ para ejecutar el modelo en la pico
│   ├── benchmark_results.txt
│   ├── build
│   │   └── examples
│   │       └── deteccionCaracol
│   │           └── deteccionCaracol.uf2    // Fichero que se anhade a la pico
│   ├── examples
│   │   ├── deteccion_caracol_mobilenetv2   // Codigo para MobileNetV2
│   │   │   ├── CMakeLists.txt
│   │   │   ├── imagen_prueba.cpp
│   │   │   ├── imagen_prueba.h
│   │   │   ├── main.cpp
│   │   │   ├── main_functions.cpp
│   │   │   ├── main_functions.h
│   │   │   ├── model_data.cpp
│   │   │   └── model_data.h
│   │   ├── deteccion_caracol_modelo3       // Codigo para modelo propio
│   │   │   ├── CMakeLists.txt
│   │   │   ├── imagen_prueba.cpp
│   │   │   ├── imagen_prueba.h
│   │   │   ├── main.cpp
│   │   │   ├── main_functions.cpp
│   │   │   ├── main_functions.h
│   │   │   ├── model_data.cpp
│   │   │   └── model_data.h
│   │   └── hello_world                     // Proyecto ejemplo tomado como base
│   ├── pico_sdk_import.cmake
│   ├── README.md
│   ├── src
│   ├── sync
│   └── tests
└── venv
</pre>

## FASES DEL DESARROLLO

## 1. IMPLEMENTACIÓN Y ENTRENAMIENTO DEL MODELO

Esta fase se ha llevado a cabo de dos formas distintas:

### 1.1. Creación de varios modelos manualmente:
  
En primer lugar se ha tratado de conocer el funcionamiento de las Redes Neuronales que componen a un modelo y en función de lo requerido se han empleado capas Convolucionales, estas capas son las más indicadas para el aprendizaje de imágenes. Tras el diseño de los modelos, ha sido necesario obtener un dataset con muchas imágenes (Alrededor de 1000) divididas en las que contienen caracol y las que no lo contienen. Finalmente, se ha llevado a cabo el entrenamiento del modelo, mediante el cual la IA adquiere los conocimientos suficientes para ser capaz de reconocer los caracoles. En adición, se han exportado los modelos en dos formatos distintos, `.keras`y `.tflite`.

La implementacion de estos modelos se ha llevado a cabo en el script `modelosIniciales.py`. Además, existe un documento `01_modelosIniciales.pdf` en el que se explica más en detalle cuáles son los pasos realizados así como la justificación de ciertas decisiones tomadas.

### 1.2. Modificación de un modelo base como MobileNetV2

El modelo implementado es una variación de MobileNetV2, que se trata de un modelo preentrenado con miles de fotos de ImageNet y clasificadas entre 1000 categorias distintas.
Para el entreno específico de nuestro propósito, se ha vuelto a emplear el mismo dataset que antes.
La implementación de este modelo se ha realizado en el script `mobileNetV2.py`, también se incluye en este script una prueba con imágenes nunca antes vistas por el modelo y su exportación en formato `.keras`y `.tflite`.

Existe un documento llamado `02_modeloMobileNetV2.pdf` en los que se explica cuál ha sido el proceso seguido y algunos detalles de ciertas decisiones tomadas.

## 2. METRICAS Y EVALUACIÓN DE LOS MODELOS

Los modelos principalmente han sido comparados gracias a las siguientes herramientas:

### Uso de TensorBoard

Se ha empleado una herramienta proporcionada por TensorFlow llamada TensorBoard, que nos permite generar gráficos que muestran la evolución de aprendizaje de cada modelo.

Podemos ver los resultados de los modelos implementados en nuestro script ejecutando lo siguiente:

`tensorboard --logdir logs`

Y abrimos en el buscador:

http://localhost:6006/

Los gráficos resultantes de esta herramienta aparecen también explicados en los documentos `01_modelosIniciales.pdf` y `02_modeloMobileNetV2.pdf`.

### Análisis de Métricas

Otra forma interesante de medir el buen funcionamiento del modelo es el uso de métricas. Estas son útiles puesto que nos ayudan a realizar estadísticas concretas del modelo y así conocer cuáles pueden ser las mayores debilidades del mismo para poder ponerles solución en el caso de que fuera necesario.

El análisis de métricas solo se ha llevado a cabo de manera exhaustiva en el modelo de MobileNetV2 puesto que este era el que principalmente nos importaba para la realización de este proyecto. Está realizado todo el análisis del mismo de forma detallada en el documento `03_metricasMobileNetV2.pdf`.

## 3. CUANTIZACIÓN

Este es uno de los pasos más importantes y necesarios para la implementación de un modelo IA en un microcontrolador, especialmente para un Raspberry Pi Pico. Esto es así debido a las limitaciones de espacio en la memoria Flash y RAM con las que contamos, por lo tanto va a ser necesario reducir el tamaño de cada modelo lo máximo posible. Adicionalmente existe otro problema, y es que la Pico no cuenta con una Unidad de Punto Flotante, lo cual es un impedimento ya que los modelos originales manejan datos decimales en lugar de enteros.

Dados esos motivos, es necesaria una cuantización a enteros de 8 bits empleando como referencia un conjunto representativo del dataset utilizado para el entreno del modelo, se realiza a partir del script `cuantizacion.py`. Los pasos seguidos para la cuantización se encuentran detallados en el documento `04_cuantizacion.pdf`, en él solo se encuentra documentada la cuantización del modelo MobileNetV2 pero la realización de este proceso para el resto de modelos sería análoga. 

Por otro lado, también se ha generado un script, `pruebaCuantizado.py`, en el que se prueba dicho modelo cuantizado para observar cuales son las consecuencias de este cambio, en el documento `04_cuantización.pdf` se muestran los cambios más significativos así como un justificación de las decisiones tomadas.

## 4. INFERENCIA EN LA RASPBERRY PI PICO

Para lograr ejecutar el modelo en la Pico, lo recomendable por los autores es tomar código implementado por ellos mismo y posteriormente ajustarlo a nuestro propósito.

La idea inicial del proyecto es la de emplear un pi cam o similar para tomar a tiempo real capturas de un entorno en el que se pueda colar un caracol. Sin embargo, actualmente no contamos con una cámara que pueda ser usada en la Pico, por lo tanto lo que se tratará de hacer en un primer momento es cargar una imagen preconfigurada (A partir de los Scripts `conversion_imagenes.py` y `conversion_imagenes_byn.py`) y, si en algún momento se cuenta con el material necesario, habrá que sustituir dicha imagen por un código que implemente el uso de la cámara.

Para esta implementación, serán únicamente los ficheros indicados, siendo el más importante `main_functions.cpp`, el cual se encarga de preparar la Pico reservando espacio para el Tensor Arena y posteriormente realiza la ejecución de la inferencia, indicando si hay caracol o no lo hay. 

Para la ejecución de la inferencia en la Pico, será necesario hacer:

`mkdir build && cd build`

`cmake .. -DPICO_SDK_PATH=/home/sandra/.pico-sdk/sdk/2.1.0 -DPICO_BOARD=pico`

`make -j4`

Una vez ejecutado lo anterior, será necesario arrastrar el ejecutable `.uf2` hasta la Raspberry. Y automáticamente se comenzará a ejecutar. Para poder observar lo que devuelve, será necesario abrir en el terminal `screen /dev/ttyACM0 115200`, donde se mostrará un valor entre -128 y 127. Siendo -128 la certeza de que no existe caracol y 127 la certeza de que le hay.

El proceso seguido para la implementación se encuentra detallado en `05_implementacionEnPico.pdf`, inicialmente solo se pretendía hacerlo para el modelo MobileNetV2, pero tras ciertas pruebas, se ha realizado también para el modelo3 de tal forma que así contamos con distintas implementaciones que nos pueden ayudar en un futuro.

## FASE EXTRA: PRUEBAS EN LA PICO

Antes de ponernos a ejecutar directamente el modelo de IA, se han implementado unos pequeños proyectos donde se pretendía familiarizarse con el entorno de desarrollo de la Pico, así como del uso de los pines (Por si se tenía que manejar la cámara, o algún tipo de pantalla que mostrase el resultado final). En `/code` se pueden encontrar dichos proyectos y es `06_pruebasRaspberry.pdf` hay una pequeña explicación del desarrollo.

## CONCLUSIONES

Tras todo el proceso de implementación, es importante destacar cuales son las posibilidades que hay a la hora de implementar un modelo de IA para un microcontrolador:

1º. Siempre que sea posible, se recomienda utilizar un modelo ya existente, preentrenado y optimizado para el uso que le queremos dar. Hay muchos, en nuestro caso se ha usado MobileNetV2 porque es el que más se puede adaptar a nuestro caso y porque es uno de los más usados.

2º. Cuantizar el modelo a int8, dado que esto el lo que más va a reducir el tamaño de nuestro modelos y además también va a hacer que el tiempo de ejecución disminuya (Ausencia de Unidad de Punto Flotante en la Pico).

3º. En caso de que un modelo ya existente no nos valga, habrá que generar uno manualmente, estableciendo el menor número de capas neuronales posibles, para hacer un modelo sencillo y probarlo hasta que funcione cómo deseamos. La implementación dependerá mucho del uso que le pretendamos dar al modelo (No es lo mismo detección de objetos en imágenes, que detección de sonidos o predicción de valores por ejemplo).

4º. Si es como en nuestro caso, que es necesario tratar imágenes, ajustar al máximo la resolución de las mismas. Esto es algo que puede afectar al tamaño de nuestra implementación así como al espacio de RAM requerido.

Es muy importante tener siempre en mente el tamaño limitado de memoria que tenemos puesto que este va a ser el principal impedimento del proyecto.

## LIBROS Y ENLACES DE AYUDA

Para la realización de este proyecto se han consultado las siguientes fuentes bibliográficas y recursos oficiales:

### Bibliografía

`TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers (Pete Warden & Daniel Situnayake)`: Considerado el libro de referencia ("la biblia") para aprender Machine Learning para Microcontroladores.

`Getting started with Raspberry Pi Pico-series (Raspberry Pi Foundation)`: Guía para la configuración del entorno y desarrollo en C/C++ sobre el microcontrolador RP2040.

### Recursos Digitales y Repositorios

`Guía oficial de TensorFlow Lite`(https://www.tensorflow.org/lite/guide): Documentación detallada sobre el proceso de conversión y cuantización de modelos.

`Repositorio oficial pico-tflmicro`(https://github.com/raspberrypi/pico-tflmicro): Repositorio de referencia que contiene las librerías necesarias para ejecutar la inferencia en la Raspberry Pi Pico.
