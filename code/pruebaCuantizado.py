import numpy as np
import tensorflow as tf

# Preparamos el dataset de prueba
PATH_PRUEBA = 'datasets/tests'

test_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_PRUEBA,
    image_size=(96, 96),
    color_mode='rgb',
    batch_size=32,
    shuffle=False
)

# Realiza la inferencia con el dataset de prueba
def run_tflite_model(tflite_file, dataset):
    # Inicializamos el interprete
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Extraemos parámetros de cuantizacion directamente
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    predictions = []
    real_labels = []

    for imagenes, etiquetas in dataset:
        for i in range(imagenes.shape[0]):
            img = imagenes[i]
            real_labels.append(etiquetas[i].numpy())

            img = (img / 127.5) - 1.0

            img = np.round((img / input_scale) + input_zero_point)
            img = np.clip(img, -128, 127)

            img = tf.cast(img, tf.int8)
            img = tf.expand_dims(img, axis=0)
            
            interpreter.set_tensor(input_details["index"], img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])[0][0]

            # Hace la prediccion
            prediccion_real = (float(output) - output_zero_point) * output_scale
            etiqueta_predicha = 1 if prediccion_real > 0.5 else 0
            predictions.append(etiqueta_predicha)

    return np.array(predictions), np.array(real_labels)


# Obtiene accuracy para ver cuanto ha empeorado de modelo
def evaluate_model(tflite_file, dataset):
    
    predictions, real_labels = run_tflite_model(tflite_file, dataset)
    accuracy = (np.sum(real_labels == predictions) * 100) / len(real_labels)

    print('The model accuracy is %.4f%%' % (
        accuracy))


# Se evalua el modelo
tflite_model_quant_file = "models/modelo_caracoles_int8.tflite"
evaluate_model(tflite_model_quant_file, test_ds)