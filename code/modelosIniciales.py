import tensorflow as tf
import matplotlib.pyplot as plt
# Necesesario para ver las gráficas con los resultados
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers

# MODIFICAMOS EL DATASET PARA QUE ESTÉ FORMADO POR IMAGENES DE BAJA RESOLUCIÓN Y
# EN BLANCO Y NEGRO

# CONSTANTES
PATH_DATOS = 'datasets/dataset/'
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
CONJUNTO_VALIDACION = 0.2

# ENTRENAMIENTO (80% de las fotos)
train_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_DATOS,
    validation_split=CONJUNTO_VALIDACION,
    subset="training",
    seed=123,               # Para que siempre elija las mismas fotos
    image_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE
)

# VALIDACIÓN (El 20% restante)
val_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_DATOS,
    validation_split=CONJUNTO_VALIDACION,
    subset="validation",
    seed=123,               # La misma semilla
    image_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE
)

# Mapeo para que los pixeles en lugar de valores de 0 a 255, serán valores
# flotantes entre 0 y 1. AL parecer la IA funciona mejor así
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))


# 1. Definimos el bloque de aumento (copiando tus parámetros)
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"), # horizontal_flip y vertical_flip
  layers.RandomRotation(0.08),  # Equivale a unos 30 grados (30/360)
  layers.RandomTranslation(height_factor=0.2, width_factor=0.2), # width/height_shift_range
  layers.RandomZoom(height_factor=(-0.3, 0.4)), # zoom_range [0.7, 1.4]
])

# MODELO 1 (DENSO)
# Lo normal es usar 'softmax' pero usamos 'sigmoid' para que regrese 0 o 1.
# si se acerca a 1, es caracol, y si se acerca a 0, no es un caracol.
# 'softmax tendría más sentido si fuera para distinguir entre más de
# dos opciones'
modelo1 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(100, 100, 1)),
  tf.keras.layers.Dense(150, activation='relu'),
  tf.keras.layers.Dense(150, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilo
modelo1.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# Entreno y guardo los datos para analizarlos posteriormente
tensorboard1 = TensorBoard(log_dir='logs/1')
modelo1.fit(train_ds, batch_size=32,
                epochs=20,
                validation_data=val_ds,
                callbacks=[tensorboard1])

# MODELO 2 (Uso de redes convolucioneles)
# Emplea capas de convolución, que buscan patrones en las imágenes
# Las va alternando con capas de reducción, que simplifican la imagen
modelo2 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
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
                    metrics=['accuracy'])

# Enreno y guardo los datos para analizarlos posteriormente
tensorboard2 = TensorBoard(log_dir='logs/2')
modelo2.fit(train_ds, batch_size=32,
                epochs=20,
                validation_data=val_ds,
                callbacks=[tensorboard2])

# MODELO 3 (Red convolucional con dropout)
# Este cambio hace que se activen y desactiven neuronas aleatoriamente,
# de tal forma que las neuronas se ven obligadas a aprender distintas formas
# de identificar un caracol en este caso, de tal forma que no se vuelven dependientes
# las unas de las otras porque todas van a tener varias habilidades.
modelo3 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(250, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilo
modelo3.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# Entreno y guardo los datos para analizarlos posteriormente
tensorboard3 = TensorBoard(log_dir='logs/3')
modelo3.fit(train_ds, batch_size=32,
                epochs=20,
                validation_data=val_ds,
                callbacks=[tensorboard3])
