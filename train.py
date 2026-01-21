import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from src.environment.pokemon_env import PokemonYellowEnv

# --- CONFIGURACIÓN ---
EXPERIMENT_NAME = "poke_ppo_v1"
TOTAL_TIMESTEPS = 7_000_000 
NUM_CPU = 2  # Ajustado a tu CPU
SAVE_FREQ_PER_ENV = 20000  # Guardará cada (20,000 * 2) = 40,000 pasos reales

# Directorios
LOG_DIR = f"experiments/{EXPERIMENT_NAME}/logs"
MODEL_DIR = f"experiments/{EXPERIMENT_NAME}/models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class SleepCallback(BaseCallback):
    """
    Frena el entrenamiento ligeramente para liberar CPU para el Stream.
    """
    def __init__(self, sleep_time=0.005, verbose=0):
        super(SleepCallback, self).__init__(verbose)
        self.sleep_time = sleep_time

    def _on_step(self) -> bool:
        # Dormir un poquito en cada paso
        time.sleep(self.sleep_time)
        return True

class ConsoleLogCallback(BaseCallback):
    """
    Callback personalizado para imprimir mensajes claros en la consola
    cuando se guarda un modelo.
    """
    def __init__(self, verbose=0):
        super(ConsoleLogCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # CheckpointCallback maneja el guardado, nosotros solo avisamos
        # Calculamos si estamos en el paso de guardado
        # n_calls es cuantas veces se ha llamado al step por ambiente
        if self.n_calls % SAVE_FREQ_PER_ENV == 0:
            total_steps = self.num_timesteps
            print(f"\n[INFO] --- GUARDANDO MODELO AUTOMÁTICO (Pasos: {total_steps}) ---")
            print(f"[INFO] Puedes ver el archivo en: {MODEL_DIR}")
        return True

def make_env(rank, seed=0):
    """Generador de entornos para multiprocesamiento."""
    def _init():
        # Usamos rgb_array para que no abra ventanas durante el entreno
        rom_path = "roms/PokemonYellow.gb"
        env = PokemonYellowEnv(rom_path, render_mode="rgb_array") 
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    print(f"--- INICIANDO ENTRENAMIENTO PRO: {EXPERIMENT_NAME} ---")
    print(f"CPUs: {NUM_CPU} | Meta: {TOTAL_TIMESTEPS} pasos")
    print(f"Guardado automático cada: {SAVE_FREQ_PER_ENV * NUM_CPU} pasos reales.")

    # 1. Entorno Vectorizado
    env = SubprocVecEnv([make_env(i) for i in range(NUM_CPU)])

    # 2. Callbacks (El Gestor de Eventos)
    # Callback oficial para guardar
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ_PER_ENV, 
        save_path=MODEL_DIR,
        name_prefix="ppo_poke"
    )

    sleep_callback = SleepCallback(sleep_time=0.005) # 5ms de pausa

    # Nuestro callback para avisar por consola
    log_callback = ConsoleLogCallback()
    
    # Unimos ambos
    callback_list = CallbackList([checkpoint_callback, sleep_callback, log_callback])

    # 3. Modelo PPO
    # Intentamos cargar un modelo previo si existe para CONTINUAR entrenando
    # (Opcional, pero útil si se te corta la luz)
    last_model_path = f"{MODEL_DIR}/interrupted_model.zip"
    
    if os.path.exists(last_model_path):
        print(f"\n¡ENCONTRADO MODELO PREVIO! Cargando {last_model_path}...")
        model = PPO.load(last_model_path, env=env, device="auto", tensorboard_log=LOG_DIR)
    else:
        print("\nCreando nuevo agente desde cero...")
        model = PPO(
            "MultiInputPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=LOG_DIR,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01, # Un poco de entropía para forzar exploración
            device="auto" # Usará CPU preferentemente para envs pequeños
        )

    # 4. Entrenamiento con SAFE EXIT (Ctrl+C)
    print("\nEntrenando... (Mira Tensorboard en localhost:6006)")
    print(">>> PRESIONA CTRL+C EN CUALQUIER MOMENTO PARA GUARDAR Y SALIR <<<")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=callback_list,
            progress_bar=True
        )
        # Si llega aquí, terminó el millón
        print("\n¡META ALCANZADA! Guardando modelo final...")
        model.save(f"{MODEL_DIR}/final_model")
        
    except KeyboardInterrupt:
        print("\n\n!!! DETECTADA INTERRUPCIÓN POR USUARIO (Ctrl+C) !!!")
        print("Guardando estado actual de emergencia...")
        model.save(f"{MODEL_DIR}/interrupted_model")
        print(f"Modelo guardado exitosamente en: {MODEL_DIR}/interrupted_model.zip")
        
    except Exception as e:
        print(f"\n\n!!! ERROR INESPERADO: {e} !!!")
        model.save(f"{MODEL_DIR}/crash_backup_model")
        
    finally:
        env.close()
        print("Entorno cerrado. Script finalizado.")

if __name__ == "__main__":
    main()