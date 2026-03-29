import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import os
import numpy as np
import tensorflow as tf
import pathlib


# Se cuantiza en modelo para que sea más ligero y 
# sea más sencillo de ejecutar en la Raspberry, que 
# no cuenta con una unidad de punto flotante.
# https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant

# Podríamos realizar las dos siguientes cuantizaciones también
# Convertimos a LiteRT
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()


# Convertimos con cuantizacion dinamica
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_model_quant = converter.convert()


# Cargamos el modelo original
model = tf.keras.models.load_model("models/modelo_caracoles_original.keras")


# PREPARADO DE UN DATASET REPRESENTATIVO
PATH_DATOS = 'datasets/dataset/'
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
CONJUNTO_VALIDACION = 0.2

train_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_DATOS,
    validation_split=CONJUNTO_VALIDACION,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE
)


# Convertimos con cuantizacion de solo enteros
def representative_data_gen():
    
    for images, labels in train_ds.take(4):
        for i in range(images.shape[0]):
            img = images[i]
            img = tf.cast(img, tf.float32)
            
            img = (img / 127.5) - 1.0 
            
            # Forma (1, 224, 224, 3)
            img = tf.expand_dims(img, axis=0)
            
            yield [img]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Para que me avise si hay operaciones que no pueden ser cuantizadas
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Entradas y salidas enteros
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convierte el modelo
tflite_model_quant = converter.convert()

# Comprueba que las entradas y salidas son enteros
interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

# Guardar el modelo
tflite_models_dir = pathlib.Path("models/")

tflite_model_quant_file = tflite_models_dir/"modelo_caracoles_int8.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)