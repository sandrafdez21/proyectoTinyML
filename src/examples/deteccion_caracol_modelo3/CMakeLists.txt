# Los archivos que tiene que compilar.
add_executable(deteccion_caracol_modelo3
  main.cpp
  main_functions.cpp
  model_data.cpp
  imagen_prueba.cpp
)

target_include_directories(deteccion_caracol_modelo3
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
)

#Enlaza con las librerías
target_link_libraries(deteccion_caracol_modelo3
  PRIVATE
  pico-tflmicro
  pico_stdlib
)

# Genera el archivo .uf2.
pico_add_extra_outputs(deteccion_caracol_modelo3)

# Activamos la salida por el cable USB
pico_enable_stdio_usb(deteccion_caracol_modelo3 1)
pico_enable_stdio_uart(deteccion_caracol_modelo3 0)
