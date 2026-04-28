from PIL import Image
import numpy as np

# Simplemente se encarga de transformar las imagenes al formato adecuado para que 
# puedan ser tratadas por la pico
IMG_SIZE = (96, 96)

# Cambiamos la direccion manualmente, solo es para probar alguna imagen y observar si funciona 
# correctamente el modelo
img = Image.open("/home/sandra/Escritorio/Prácticas/proyectoTinyML/datasets/tests/snail/480.jpg").convert("L").resize(IMG_SIZE)
data = np.array(img).flatten()
print(",".join(str(x) for x in data))