import tensorflow as tf
import matplotlib.pyplot as plt
# Necesesario para ver las gráficas con los resultados
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers
import numpy as np


# MODIFICAMOS EL DATASET PARA QUE ESTÉ FORMADO POR IMAGENES DE BAJA RESOLUCIÓN

# CONSTANTES
PATH_DATOS = 'datasets/dataset/'
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
CONJUNTO_VALIDACION = 0.2
#También

# ENTRENAMIENTO (80% de las fotos)
train_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_DATOS,
    validation_split=CONJUNTO_VALIDACION,
    subset="training",
    seed=123,               # Para que siempre elija las mismas fotos
    image_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE
)

# VALIDACIÓN (El 20%)
val_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_DATOS,
    validation_split=CONJUNTO_VALIDACION,
    subset="validation",
    seed=123,               # La misma semilla
    image_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE
)

## Se escalan píxeles a [-1, 1]
capa_normalizacion = tf.keras.layers.Rescaling(1./127.5, offset=-1)
train_ds = train_ds.map(lambda x, y: (capa_normalizacion(x), y))
val_ds = val_ds.map(lambda x, y: (capa_normalizacion(x), y))

# AUMENTO DE DATOS (Hace zoom, voltea, gira, etc.)
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.08),
  layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
  layers.RandomZoom(height_factor=(-0.3, 0.4)),
])

# DEFINICION DEL MODELO
# Tomamos MobileNet V2 y modificamos ciertos parámetros para adecuarlo a
# nuestro proyecto
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3), # De momento, mantenemos la deficicion original
                               # para la que está diseñado este modelo.
    alpha=0.35,                # alpha es el parámetro más importante a tener en cuenta para reducir el tamaño de nuesto modelo
                               # con él se determina el número de filtro que tendrá nuestra red neuronal.
                               # 0.35 hace que los filtros se reduzcan con respecto al modelo original
    include_top=False,         # Quitamos las 1000 clases para ahorrar Flash, quita la ultima capa
                               # en la que hay una neurona por cada una de las 1000 clases del modelo entrenado
    weights="imagenet"         # Cargamos el conocimiento de pre-entreno, imagenet le permite
                               # distinguir entre 1000 clases distintas.
                               # Esto es importante porque si analizamos esas clases,
                               # la 113 se corresponde con caracoles, por lo que el modelo
                               # sin ser entrenado ya sabe distinguir caracoles.
)

base_model.trainable = False # Para mantener sus conocimientos anteriores

# Crear el modelo secuencial
model = tf.keras.Sequential([

    tf.keras.layers.InputLayer(input_shape=(96, 96, 3)),

    base_model, # Añadimos el modelo MobileNet V2 modificado

    # Reducción de dimensiones
    tf.keras.layers.GlobalAveragePooling2D(),

    # Clasificación final
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 1. Le decimos al modelo que, además de la exactitud, queremos que mida esto:
model.compile(
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc', num_thresholds=10000),
        tf.keras.metrics.AUC(
            name='auc_precision_recall', curve='PR', num_thresholds=10000),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Entreno y guardo los datos para analizarlos posteriormente
tensorboard = TensorBoard(log_dir='logs/mobileNetV2')
model.fit(train_ds, batch_size=32,
                epochs=20,
                validation_data=val_ds,
                callbacks=[tensorboard])

# Mostramos el tamanho del modelo
model.summary()

#PROBAMOS CON IMAGENES QUE NO HA VISTO NUNCA ANTES
PATH_PRUEBA = 'datasets/tests'

# Cargamos un dataset que había dejado reservado para esto
test_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_PRUEBA,
    image_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=32,
    shuffle=True
)

test_ds = test_ds.map(lambda x, y: (capa_normalizacion(x), y))

# Prueba el modelo con el dataset de test
resultados = model.evaluate(test_ds)

# Codigo que muestra de forma visual cómo funciona el modelo
for images, labels in test_ds.take(1):

    predicciones = model.predict(images)

    plt.figure(figsize=(15, 15))
    for i in range(32):
        ax = plt.subplot(11, 3, i + 1)

        img_array = ((images[i].numpy() + 1.0) * 127.5).astype("uint8")
        plt.imshow(img_array)

        probabilidad = predicciones[i][0]
        if probabilidad > 0.5:
            resultado_ia = "Caracol"
            color = "green" if labels[i] == 1 else "red"
        else:
            resultado_ia = "No Caracol"
            color = "green" if labels[i] == 0 else "red"

        plt.title(f"IA dice: {resultado_ia}\n(Seguridad: {probabilidad*100:.1f}%)", color=color)
        plt.axis("off")

    plt.tight_layout()

    plt.savefig('resultado_entrenamiento.png')
    print("Gráfica guardada como resultado_entrenamiento.png")


# Se exporta el modelo en formato normal
model.save('models/modelo_caracoles_original.keras')

# Se exporta en formato tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('models/modelo_caracoles_base.tflite', 'wb') as f:
    f.write(tflite_model)

