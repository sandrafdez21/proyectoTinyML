// Librerias que necesitamos
#include "constants.h"
#include "model_data_96x96.h"
#include "main_functions.h"
//#include "output_handler.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "pico/stdlib.h"

// Para la prueba inicial no usaremos la camara, sino que 
// directamente usaremos un imagen fija.
#include "imagen_prueba.h"

// Punteros al modelo, interprete y a la entrada y salida
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Tambien reservamos espacio en la RAM para el tensor arena
// Para saber cuanto espacio hay que reservar hay que ir probando
// empezaremos por un gran espacio y se ira reduciendo todo lo 
// que se pueda
constexpr int kTensorArenaSize = 100 * 1024; // 100KB
uint8_t tensor_arena[kTensorArenaSize];
} 

void setup() {
  tflite::InitializeTarget();

  // Toma nuestro modelo
  model = tflite::GetModel(g_model_data_96x96);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Carga las operaciones matematicas
  // Para optimizar al maximo el espacio, solo cargare las operaciones necesarias
  // WEB: netron.app, aqui se nos muestran todas las capas que 
  // contiene nuestro modelo
  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddMean();
  resolver.AddLogistic();
  resolver.AddAdd();
  //resolver.AddRelu6(); //En principio no se pone pq suele ir incluida

  // Creamos el interprete
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Se reserva la memoria
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtenemos la entrada y la salida de nuestro interprete
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  
    // Cargamos la imagen en el tensor de entrada
    for (int i = 0; i < 9216; i++) {
        // Importante que los valores de cada pixel sean entre 
        // (-128 a 127)
        input->data.int8[i] = g_imagen_caracol[i] - 128;
    }

    // Se realiza la inferencia
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("¡Fallo al ejecutar la inferencia!");
        return;
    }

  //Obtenemos el resultado
  int8_t resultado_bruto = output->data.int8[0];

  // Mostramos el resultado en el monitor serie
  MicroPrintf("Inferencia completada. Resultado bruto: %d", resultado_bruto);

  sleep_ms(1000);
}