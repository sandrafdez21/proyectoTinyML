import tensorflow as tf
import matplotlib.pyplot as plt
# Necesesario para ver las gráficas con los resultados
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers

# MODIFICAMOS EL DATASET PARA QUE ESTÉ FORMADO POR IMAGENES DE BAJA RESOLUCIÓN Y
# EN BLANCO Y NEGRO

# CONSTANTES
PATH_DATOS = 'datasets/dataset/'
PATH_PRUEBA = 'datasets/tests'
IMG_SIZE = (96, 96)
INPUT_SHAPE = (96, 96, 1)
BATCH_SIZE = 32
EPOCHS = 20
CONJUNTO_VALIDACION = 0.2
SEED = 123
COLOR_MODE = 'grayscale'

# ENTRENAMIENTO (80% de las fotos)
train_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_DATOS,
    validation_split=CONJUNTO_VALIDACION,
    subset="training",
    seed=SEED,               # Para que siempre elija las mismas fotos
    image_size=IMG_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE
)

# VALIDACIÓN (El 20% restante)
val_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_DATOS,
    validation_split=CONJUNTO_VALIDACION,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE
)

# Mapeo para que los pixeles en lugar de valores de 0 a 255, serán valores
# flotantes entre 0 y 1. Al parecer la IA funciona mejor así
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))


# Definimos el bloque de aumento, como ya sabemos sirve para
# que el modelo aprenda mejor
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),                      # Voltea en vertical y en horizontal
  layers.RandomRotation(0.08),                                       # Rota unos 30 grados
  layers.RandomTranslation(height_factor=0.2, width_factor=0.2),     # Traslada la imagen
  layers.RandomZoom(height_factor=(-0.3, 0.4)),                      # Amplia dicha imagen
])

#------------------------------------------------------------------------------------------------------------------#

# MODELO 1 (DENSO)
# Lo normal es usar 'softmax' pero usamos 'sigmoid' para que regrese 0 o 1.
# si se acerca a 1, es caracol, y si se acerca a 0, no es un caracol.
# 'softmax tendría más sentido si fuera para distinguir entre más de
# dos opciones'
modelo1 = tf.keras.models.Sequential([
    data_augmentation,
  tf.keras.layers.Flatten(input_shape=INPUT_SHAPE),
  tf.keras.layers.Dense(150, activation='relu'),
  tf.keras.layers.Dense(150, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilo
modelo1.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=[
                                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                tf.keras.metrics.AUC(name='auc', num_thresholds=10000),
                                tf.keras.metrics.AUC(
                                    name='auc_precision_recall', curve='PR', num_thresholds=10000),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall')
                            ])

# Entreno y guardo los datos para analizarlos posteriormente
tensorboard1 = TensorBoard(log_dir='logs/1')
modelo1.fit(train_ds, batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=val_ds,
                callbacks=[tensorboard1],
                verbose=0 # Borrar esta linea si queremos observar como van las epocas del entrenamiento
)


#------------------------------------------------------------------------------------------------------------------#

# MODELO 2 (Uso de redes convolucioneles)
# Emplea capas de convolución, que buscan patrones en las imágenes
# Las va alternando con capas de reducción, que simplifican la imagen
modelo2 = tf.keras.models.Sequential([
  data_augmentation,
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=INPUT_SHAPE),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilo
modelo2.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=[
                        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                        tf.keras.metrics.AUC(name='auc', num_thresholds=10000),
                        tf.keras.metrics.AUC(
                            name='auc_precision_recall', curve='PR', num_thresholds=10000),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')
                    ])

# Enreno y guardo los datos para analizarlos posteriormente
tensorboard2 = TensorBoard(log_dir='logs/2')
modelo2.fit(train_ds, batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=val_ds,
                callbacks=[tensorboard2],
                verbose=0 # Borrar esta linea si queremos observar como van las epocas del entrenamiento
)


#------------------------------------------------------------------------------------------------------------------#


# MODELO 3 (Red convolucional con dropout)
# Este cambio hace que se activen y desactiven neuronas aleatoriamente,
# de tal forma que las neuronas se ven obligadas a aprender distintas formas
# de identificar un caracol en este caso, de tal forma que no se vuelven dependientes
# las unas de las otras porque todas van a tener varias habilidades.
modelo3 = tf.keras.models.Sequential([
  data_augmentation,
  tf.keras.layers.Conv2D(16, (3,3), strides=(2,2), activation='relu', input_shape=INPUT_SHAPE),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3) ,activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Dropout(0.5),

  tf.keras.layers.GlobalAveragePooling2D(),

  tf.keras.layers.Dense(250, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilo
modelo3.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=[
                      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                      tf.keras.metrics.AUC(name='auc', num_thresholds=10000),
                      tf.keras.metrics.AUC(
                          name='auc_precision_recall', curve='PR', num_thresholds=10000),
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall')
                    ])

# Entreno y guardo los datos para analizarlos posteriormente
tensorboard3 = TensorBoard(log_dir='logs/3')
modelo3.fit(train_ds, batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=val_ds,
                callbacks=[tensorboard3],
                verbose=0 # Borrar esta linea si queremos observar como van las epocas del entrenamiento
)


#------------------------------------------------------------------------------------------------------------------#

#PROBAMOS EL MODELO
# Probaremos el modelo con imagenes que no ha visto nunca
# y veremos que tal funciona.
# Cargamos un dataset que había dejado reservado para esto

test_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_PRUEBA,
    image_size=IMG_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

MODELO_A_PROBAR = modelo3 # Cambiar para probar los demas

# Prueba el modelo con el dataset de test
resultados = MODELO_A_PROBAR.evaluate(test_ds)

#------------------------------------------------------------------------------------------------------------------#

# EXPORTAR EL MODELO
# Se puede exportar cualquier modelo, en este caso exportaremos:

MODELO_A_EXPORTAR = modelo3 #Cambiar para exportar lo demas

# Se exporta el modelo en formato normal.
MODELO_A_EXPORTAR.save('models/modelo_caracoles_original.keras')

# Se exporta en formato tflite.
converter = tf.lite.TFLiteConverter.from_keras_model(MODELO_A_EXPORTAR)
tflite_model = converter.convert()
with open('models/modelo_caracoles_base.tflite', 'wb') as f:
    f.write(tflite_model)
