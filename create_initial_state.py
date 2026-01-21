from pyboy import PyBoy
import os

def create_state():
    rom_path = "roms/PokemonYellow.gb"
    state_path = "states/init.state"
    
    # Verificar que existe la carpeta states
    if not os.path.exists("states"):
        os.makedirs("states")

    print("--- INSTRUCCIONES ---")
    print("1. Se abrirá el juego. JUEGA TÚ MANUALMENTE.")
    print("2. Controles por defecto de PyBoy:")
    print("   Flechas: Movimiento")
    print("   Z: Botón A")
    print("   X: Botón B")
    print("   Enter: Start")
    print("   Backspace: Select")
    print("3. Pasa la intro, ponle nombre a tu PJ y rival.")
    print("4. Cuando aparezcas en tu cuarto y PUEDAS MOVERTE, CIERRA LA VENTANA DEL JUEGO.")
    print("5. El estado se guardará automáticamente al cerrar.")
    print("---------------------")

    # Inicializamos PyBoy en modo ventana
    pyboy = PyBoy(rom_path, window="SDL2")
    pyboy.set_emulation_speed(1) # Velocidad normal para que puedas jugar

    try:
        while pyboy.tick():
            pass # Bucle infinito hasta que cierres la ventana
    except KeyboardInterrupt:
        pass
    
    # Al salir del bucle (cerrar ventana), guardamos estado
    print(f"\nGuardando estado en {state_path}...")
    with open(state_path, "wb") as f:
        pyboy.save_state(f)
    
    print("¡Estado guardado exitosamente!")
    pyboy.stop()

if __name__ == "__main__":
    create_state()