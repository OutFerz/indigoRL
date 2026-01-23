from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from src.environment.pokemon_env import PokemonYellowEnv
import os
from stream_agent_wrapper import StreamWrapper
import uuid

# --- CONFIGURATION ---
ROM_PATH = "roms/PokemonYellow.gb"
SESSION_NAME = "poke_lstm_v1"
CHECKPOINT_DIR = f"experiments/{SESSION_NAME}/models"
LOG_DIR = f"experiments/{SESSION_NAME}/logs"
TOTAL_TIMESTEPS = 10000000 
NUM_CPU = 6 
FINAL_MODEL_PATH = f"{CHECKPOINT_DIR}/final_model_optimized"

# Save every 20 network updates
SAVE_FREQ = (2048 * NUM_CPU * 20) // NUM_CPU 

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if __name__ == "__main__":
    # 1. Create Vectorized Environment
    env = make_vec_env(
        lambda: StreamWrapper(PokemonYellowEnv(ROM_PATH, render_mode='rgb_array'), 
                              stream_metadata={"user": "Pokemon_Yellow\n",
                                              "env_id": uuid.uuid4().hex[:8],
                                              "color": "#a200ff", # 
                                              "extra": ""}),
        n_envs=NUM_CPU,
        vec_env_cls=SubprocVecEnv
    )

    # 2. Callback for periodic saving
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="lstm_model_optimized"
    )

    # 3. Model Loading or Creation Logic
    # Check if previous final model exists to resume
    if os.path.exists(f"{FINAL_MODEL_PATH}.zip"):
        print(f"--- REANUDANDO ENTRENAMIENTO: Cargando {FINAL_MODEL_PATH} ---")
        model = RecurrentPPO.load(
            FINAL_MODEL_PATH, 
            env=env, 
            device="auto", # Automatically detects your 6600 XT
            tensorboard_log=LOG_DIR
        )
    else:
        print(f"--- INICIANDO ENTRENAMIENTO DESDE CERO: {SESSION_NAME} ---")
        model = RecurrentPPO(
            "MultiInputLstmPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=LOG_DIR,
            learning_rate=0.00025,
            n_steps=2048,          
            batch_size=1024,        
            n_epochs=15,
            gamma=0.998,           
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,         
            policy_kwargs=dict(
                enable_critic_lstm=False, 
                lstm_hidden_size=256,
            )
        )

    # 4. Training execution
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            tb_log_name="LSTM_Optimized_Heavy_Batch",
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False # Keeps global step count in TensorBoard
        )
    except KeyboardInterrupt:
        print("\n--- Pausa detectada. Guardando progreso... ---")
    finally:
        # Safety save always on close
        model.save(FINAL_MODEL_PATH)
        env.close()
        print(f"✅ Proceso guardado en: {FINAL_MODEL_PATH}")