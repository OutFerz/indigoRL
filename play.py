import os
import glob
import time
import cv2
import numpy as np
from stable_baselines3 import PPO
from src.environment.pokemon_env import PokemonYellowEnv

# --- CONFIGURACIÓN ---
MODEL_DIR = "experiments/poke_ppo_v1/models"
ROM_PATH = "roms/PokemonYellow.gb"
SCALE = 3

# --- PALETA DE COLORES GAME BOY ---
GB_CASE = (180, 180, 180)    # Gris plástico
GB_SCREEN_BORDER = (100, 100, 100) # Gris oscuro borde
GB_DPAD_DARK = (40, 40, 40)  # Cruceta oscura
GB_DPAD_LIGHT = (60, 60, 60) # Cruceta clara (centro)
GB_BTN_PURPLE = (100, 50, 100) # Botones A/B oscuros
GB_BTN_PURPLE_L = (130, 70, 130) # Botones A/B claros (centro)
COLOR_TEXT = (50, 50, 50)    # Texto oscuro
COLOR_ON_NEON = (0, 255, 255) # Amarillo neón para indicar presión

# --- CONFIGURACIÓN DE DISEÑO (Centrado y Grande) ---
PANEL_W = 300
DPAD_CENTER = (100, 150)
DPAD_SIZE = 35 # Tamaño de un brazo de la cruceta
BTN_A_CENTER = (240, 120)
BTN_B_CENTER = (190, 150)
BTN_RADIUS = 25

def get_latest_model():
    list_of_files = glob.glob(f'{MODEL_DIR}/*.zip')
    if not list_of_files: return None
    return max(list_of_files, key=os.path.getctime)

def draw_gb_button_circle(panel, center, radius, base_color, light_color, text, is_pressed):
    """Dibuja un botón circular estilo Game Boy con efecto 3D simple."""
    # Borde exterior (sombra)
    cv2.circle(panel, center, radius, base_color, -1)
    # Centro (luz) o neón si está presionado
    inner_color = COLOR_ON_NEON if is_pressed else light_color
    cv2.circle(panel, center, radius - 5, inner_color, -1)
    # Texto
    text_color = (0,0,0) if is_pressed else (200,200,200)
    cv2.putText(panel, text, (center[0]-10, center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

def draw_gb_dpad(panel, center, size, pressed_idx):
    """Dibuja la cruceta con efecto 3D y estado de presión."""
    cx, cy = center
    s = size
    # Base oscura
    cv2.rectangle(panel, (cx-s, cy-s//3), (cx+s, cy+s//3), GB_DPAD_DARK, -1) # Horizontal
    cv2.rectangle(panel, (cx-s//3, cy-s), (cx+s//3, cy+s), GB_DPAD_DARK, -1) # Vertical
    
    # Colores de estado
    c_up = COLOR_ON_NEON if pressed_idx == 3 else GB_DPAD_LIGHT
    c_down = COLOR_ON_NEON if pressed_idx == 0 else GB_DPAD_LIGHT
    c_left = COLOR_ON_NEON if pressed_idx == 1 else GB_DPAD_LIGHT
    c_right = COLOR_ON_NEON if pressed_idx == 2 else GB_DPAD_LIGHT
    
    # Tapas claras (botones)
    pad = 5
    cv2.rectangle(panel, (cx-s+pad, cy-s//3+pad), (cx-s//3, cy+s//3-pad), c_left, -1)
    cv2.rectangle(panel, (cx+s//3, cy-s//3+pad), (cx+s-pad, cy+s//3-pad), c_right, -1)
    cv2.rectangle(panel, (cx-s//3+pad, cy-s+pad), (cx+s//3-pad, cy-s//3), c_up, -1)
    cv2.rectangle(panel, (cx-s//3+pad, cy+s//3), (cx+s//3-pad, cy+s-pad), c_down, -1)
    # Centro
    cv2.rectangle(panel, (cx-s//3+pad, cy-s//3+pad), (cx+s//3-pad, cy+s//3-pad), GB_DPAD_LIGHT, -1)


def draw_gamepad_panel(pressed_btn_idx, height):
    """Renderiza el panel completo estilo Game Boy."""
    panel = np.full((height, PANEL_W, 3), GB_CASE, dtype=np.uint8)
    
    # Decoración: Borde de pantalla falso a la izquierda
    cv2.rectangle(panel, (0, 0), (20, height), GB_SCREEN_BORDER, -1)

    # Título
    cv2.putText(panel, "NEURAL INPUT", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    
    # DIBUJAR CONTROLES
    draw_gb_dpad(panel, DPAD_CENTER, DPAD_SIZE, pressed_btn_idx)
    draw_gb_button_circle(panel, BTN_A_CENTER, BTN_RADIUS, GB_BTN_PURPLE, GB_BTN_PURPLE_L, "A", pressed_btn_idx == 4)
    draw_gb_button_circle(panel, BTN_B_CENTER, BTN_RADIUS, GB_BTN_PURPLE, GB_BTN_PURPLE_L, "B", pressed_btn_idx == 5)

    # Decoración: Rejilla de altavoz falsa
    for i in range(5):
        cv2.line(panel, (220 + i*10, 250), (240 + i*10, 280), GB_SCREEN_BORDER, 2)

    return panel

def main():
    print("--- STREAM GAME BOY VISUALIZER ---")
    env = PokemonYellowEnv(ROM_PATH, render_mode="stream") 
    
    # --- CALLBACK DE RENDERIZADO EN TIEMPO REAL ---
    # Ahora recibe el índice del botón presionado físicamente
    def live_render(pressed_btn_idx):
        game_pixels = env.render()
        if game_pixels.shape[2] == 4: game_pixels = game_pixels[:, :, :3]
        game_bgr = cv2.cvtColor(game_pixels, cv2.COLOR_RGB2BGR)
        
        h, w, _ = game_bgr.shape
        game_view = cv2.resize(game_bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
        
        # Dibujamos el mando usando el estado real del botón
        gamepad_view = draw_gamepad_panel(pressed_btn_idx, height=game_view.shape[0])
        
        final_visual = np.hstack((game_view, gamepad_view))
        cv2.imshow("Project Red-RL | Live Feed", final_visual)
        cv2.waitKey(1)
    
    env.set_render_callback(live_render)
    # ----------------------------------------------

    current_model_path = None
    model = None
    
    try:
        while True:
            latest_model_path = get_latest_model()
            if not latest_model_path:
                time.sleep(2)
                continue
                
            if latest_model_path != current_model_path:
                print(f"[UPDATE] Cargando: {os.path.basename(latest_model_path)}")
                time.sleep(0.5)
                try:
                    model = PPO.load(latest_model_path, env=env)
                    current_model_path = latest_model_path
                except: continue
            
            obs, _ = env.reset()
            done = False
            
            while not done:
                if model:
                    action, _ = model.predict(obs, deterministic=False)
                else:
                    action = env.action_space.sample()

                # La animación suave y el renderizado ocurren dentro de step()
                obs, _, terminated, truncated, _ = env.step(action)
                
                if terminated or truncated: done = True

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()