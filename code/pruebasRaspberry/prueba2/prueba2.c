#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/adc.h" // Para manejar el pin analogico a digital que va al sensor de temperatura


int main()
{
    stdio_init_all();
    adc_init();

    adc_set_temp_sensor_enabled(true);
    adc_select_input(4);

    while (true) {

        uint16_t valor_bruto = adc_read(); // Lee la senhal recibida, va de 0 a 4095

        float voltaje = valor_bruto * (3.3f / (1 << 12));   // Lo escala a los 3.3V a los que funciona la pico

        float temperatura = 27.0f - (voltaje - 0.706f) / 0.001721f;  // Formula proporcionada por Raspberry Pi para obtener los grados Celsius

        printf("Temperatura: %.2fºC\n", temperatura);

        // Esperamos 1 segundo antes de volver a medir
        sleep_ms(1000);
    }
}
