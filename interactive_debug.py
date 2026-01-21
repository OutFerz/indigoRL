from pyboy import PyBoy
import os
import time
from src.utils.memory_reader import MemoryReader

def main():
    print("--- DEBUGGER INTERACTIVO V2 ---")
    print("1. Haz clic en la ventana de 'PyBoy' para enfocarla.")
    print("2. CONTROLES (Teclado):")
    print("   - Flechas: Moverse")
    print("   - A: Botón A (Interactuar)")
    print("   - S: Botón B (Correr/Cancelar)")
    print("   - Enter: Start")
    print("   - Backspace: Select")
    print("3. Mira esta consola: Las coordenadas DEBEN cambiar al moverte.")
    print("-------------------------------")
    
    rom_path = "roms/PokemonYellow.gb"
    state_path = "states/init.state"
    
    # Iniciamos PyBoy
    pyboy = PyBoy(rom_path, window="SDL2")
    
    # Cargamos el estado si existe
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            pyboy.load_state(f)
            print("Estado cargado correctamente.")
    
    memory = MemoryReader(pyboy)
    pyboy.set_emulation_speed(1) # Velocidad normal

    try:
        # Bucle corregido: Mientras el juego siga abierto...
        while pyboy.tick():
            
            # Imprimir RAM cada 30 cuadros (0.5 segundos aprox)
            if pyboy.frame_count % 30 == 0:
                ram = memory.get_ram_state()
                print(f"RAM -> X: {ram['x']} | Y: {ram['y']} | Map: {ram['map_id']} | Badges: {ram['badges']}")
                
    except KeyboardInterrupt:
        pass
    
    print("\nCerrando debug.")
    pyboy.stop()

if __name__ == "__main__":
    main()