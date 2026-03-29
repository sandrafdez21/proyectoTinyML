#include <stdio.h>
#include "pico/stdlib.h" //Usada para gestionar los pines


int main()
{
    // Para que funcione el USB
    stdio_init_all();

    const uint PIN_LED = 25;
    gpio_init(PIN_LED); //Activa el led para el funcionamiento
    gpio_set_dir(PIN_LED, true); // Hace que el pin sea de salida

    while (true) {
        gpio_put(PIN_LED, true);
        printf("Se enciende el LED\n");
        sleep_ms(1000);

        gpio_put(PIN_LED, false);
        printf("Se apaga el LED\n");
        sleep_ms(1000);
    }

    return 0;
}
