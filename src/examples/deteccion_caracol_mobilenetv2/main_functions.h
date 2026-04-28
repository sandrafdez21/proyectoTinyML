// Avisa al compilador para evitar incluir varias veces
// las mismas funciones
#ifndef MAIN_FUNCTIONS_H
#define MAIN_FUNCTIONS_H

// Para que trate las funciones como si fueran C
#ifdef __cplusplus
extern "C" {
#endif

// Se inicializa todo lo necesario
void setup();

// Se realiza repetidamente el reconocimiento de las imagenes
void loop();

#ifdef __cplusplus
}
#endif

#endif