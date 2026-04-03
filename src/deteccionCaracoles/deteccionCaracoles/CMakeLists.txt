# 1. Creamos el ejecutable y le decimos qué archivos .cpp necesita compilar
add_executable(deteccionCaracol
  main.cpp
  main_functions.cpp
  model_data_96x96.cpp
  imagen_prueba.cpp
)

# 2. Le decimos que busque los archivos .h en esta misma carpeta
target_include_directories(deteccionCaracol
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
)

# 3. Enlazamos el cerebro de IA (pico-tflmicro) y el tiempo (pico_stdlib para el sleep)
target_link_libraries(deteccionCaracol
  PRIVATE
  pico-tflmicro
  pico_stdlib
)

# 4. ¡LA LÍNEA MÁGICA! 
# Esto le dice a la Pico que genere el archivo .uf2 arrastrable.
pico_add_extra_outputs(deteccionCaracol)

# Activamos la salida por el cable USB
pico_enable_stdio_usb(deteccionCaracol 1)

# Desactivamos la salida por los pines físicos (opcional, pero buena práctica)
pico_enable_stdio_uart(deteccionCaracol 0)