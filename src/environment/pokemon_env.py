import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from src.utils.memory_reader import MemoryReader
import os
import time

class PokemonYellowEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'stream'], 'render_fps': 60}

    def __init__(self, rom_path, render_mode='rgb_array'):
        super(PokemonYellowEnv, self).__init__()
        
        self.rom_path = rom_path
        self.state_path = "states/init.state"
        self.render_mode = render_mode
        
        window_type = "SDL2" if render_mode == "human" else "null"
        self.pyboy = PyBoy(rom_path, window=window_type)
        
        if render_mode == "human":
            self.pyboy.set_emulation_speed(1)
        else:
            self.pyboy.set_emulation_speed(0)
            
        self.memory_reader = MemoryReader(self.pyboy)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8),
            "ram": spaces.Box(low=0, high=255, shape=(4,), dtype=np.uint8)
        })

        self.visited_locations = set()
        self.total_reward = 0
        
        # Callback para renderizado y estado del botón actual (-1 = ninguno)
        self.render_callback = None
        self.currently_pressed_btn_idx = -1

    def set_render_callback(self, callback_func):
        self.render_callback = callback_func

    def step(self, action):
        self._perform_action(action)
        
        info = self.memory_reader.get_ram_state()
        reward = 0
        current_loc = (info['map_id'], info['x'], info['y'])
        
        if current_loc not in self.visited_locations:
            reward = 1.0
            self.visited_locations.add(current_loc)
        
        self.total_reward += reward
        observation = self._get_obs()
        
        return observation, reward, False, False, info

    def _perform_action(self, action):
        btn = 'none'
        btn_idx = -1
        if action == 0: btn, btn_idx = 'down', 0
        elif action == 1: btn, btn_idx = 'left', 1
        elif action == 2: btn, btn_idx = 'right', 2
        elif action == 3: btn, btn_idx = 'up', 3
        elif action == 4: btn, btn_idx = 'a', 4
        elif action == 5: btn, btn_idx = 'b', 5
        
        use_smooth = self.render_mode in ["human", "stream"]
        
        if btn != 'none':
            # PRESIONAR
            self.pyboy.button_press(btn)
            if use_smooth:
                # Marcamos que el botón está físicamente presionado ahora
                self.currently_pressed_btn_idx = btn_idx 
                self._smooth_tick(2)
            else:
                self.pyboy.tick(2)
            
            # SOLTAR
            self.pyboy.button_release(btn)
            if use_smooth:
                # Marcamos que ya no hay botón presionado
                self.currently_pressed_btn_idx = -1 
                self._smooth_tick(20) # Animación de espera
            else:
                self.pyboy.tick(20)
        else:
            if use_smooth: self._smooth_tick(22)
            else: self.pyboy.tick(22)

    def _smooth_tick(self, frames):
        """Avanza frame a frame llamando al renderizador con el estado actual."""
        target_frame_time = 1.0 / 60.0
        for _ in range(frames):
            start = time.perf_counter()
            self.pyboy.tick()
            
            # Pasamos el estado actual del botón al visualizador
            if self.render_callback:
                self.render_callback(self.currently_pressed_btn_idx)

            elapsed = time.perf_counter() - start
            wait = target_frame_time - elapsed
            if wait > 0:
                time.sleep(wait)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.visited_locations = set()
        self.total_reward = 0
        self.currently_pressed_btn_idx = -1
        
        if os.path.exists(self.state_path):
            with open(self.state_path, "rb") as f:
                self.pyboy.load_state(f)
            self.pyboy.tick(5)
        else:
            if self.pyboy.frame_count > 0:
                 self.pyboy.load_rom(self.rom_path)

        info = self.memory_reader.get_ram_state()
        self.visited_locations.add((info['map_id'], info['x'], info['y']))
        
        # Render inicial
        if self.render_callback: self.render_callback(-1)

        return self._get_obs(), {}

    def render(self):
        return self.pyboy.screen.ndarray

    def close(self):
        self.pyboy.stop()

    def _get_obs(self):
        raw_screen = self.pyboy.screen.ndarray 
        if raw_screen.shape[2] == 4:
            screen = raw_screen[:, :, :3]
        else:
            screen = raw_screen
        return {
            "screen": screen.astype(np.uint8),
            "ram": np.array([0,0,0,0], dtype=np.uint8)
        }