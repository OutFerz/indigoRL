import time
import random
from src.environment.pokemon_env import PokemonYellowEnv

def main():
    # Ruta relativa a la ROM
    rom_path = "roms/PokemonYellow.gb"
    
    print("Iniciando entorno Pokémon Yellow...")
    env = PokemonYellowEnv(rom_path, render_mode="human")
    
    obs, info = env.reset()
    
    print("Entorno cargado. Iniciando bucle aleatorio...")
    
    try:
        for i in range(1000): # Correr por 1000 pasos
            # Tomar acción aleatoria (0 a 5)
            action = env.action_space.sample()
            
            # Ejecutar paso
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Logs cada 20 pasos para no saturar la consola
            if i % 20 == 0:
                print(f"Paso: {i} | Acción: {action} | Coords: ({info['x']}, {info['y']}) | Mapa: {info['map_id']}")
            
            # Pequeña pausa para que puedas verlo (opcional, PyBoy ya limita FPS)
            # time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print("Detenido por el usuario.")
    finally:
        env.close()
        print("Emulación cerrada.")

if __name__ == "__main__":
    main()