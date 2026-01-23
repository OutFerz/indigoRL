import io
import os
import random
from gymnasium import Env, spaces
import numpy as np
from pyboy import PyBoy
from skimage.transform import resize

class PokemonYellowEnv(Env):
    def __init__(self, rom_path, render_mode='rgb_array', observation_type='multi'):
        super().__init__()
        self.rom_path = rom_path
        self.render_mode = render_mode
        self.observation_type = observation_type
        self.upload_interval = 300 # num of coords captured before sent to stream. needs adjusted based off training speed.

        # --- MEMORY ADDRESSES (Extracted from wram.asm) ---
        self.MEM_EVENT_FLAGS_START = 0xD747
        self.MEM_EVENT_FLAGS_END = 0xD747 + 320 
        self.MEM_MAP_ID = 0xD35D
        self.MEM_IS_IN_BATTLE = 0xD057
        self.MEM_ENEMY_HP_HIGH = 0xCFE6
        self.MEM_ENEMY_HP_LOW = 0xCFE7
        self.MEM_MY_HP_HIGH = 0xD16C
        self.MEM_MY_HP_LOW = 0xD16D
        self.MEM_PARTY_LEVELS = 0xD18C
        self.MEM_X_COORD = 0xD361
        self.MEM_Y_COORD = 0xD360
        self.MEM_PARTY_SPECIES = 0xD164 # List of species in the party
        self.MEM_POKEDEX_OWNED = 0xD2F7 # Start of capture flags (19 bytes)
        self.MEM_EVENT_FLAGS_START = 0xD747
        self.MEM_POKEDEX_OWNED = 0xD2F7 
        self.MEM_PARTY_SPECIES = 0xD164
        self.MEM_IS_IN_BATTLE = 0xD057
        
        
        # PyBoy 2.0 Configuration
        window_type = "null" if render_mode == 'rgb_array' else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window_type)
        if render_mode == 'rgb_array':
            self.pyboy.set_emulation_speed(0) 

        self.screen_width = 160
        self.screen_height = 144
        self.render_callback = None 

        self.valid_actions = ['down', 'left', 'right', 'up', 'a', 'b', 'start']
        self.action_space = spaces.Discrete(len(self.valid_actions))

        # --- OBSERVATION (FLOAT32 FOR STABILITY) ---
        self.output_shape = (3, self.screen_height, self.screen_width)
        screen_space = spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)
        
        # Normalized RAM: [X, Y, MapID, MyHP, EnemyHP, Levels, InBattle]
        ram_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            'screen': screen_space,
            'ram': ram_space
        })

        # Internal state variables
        self.visited_maps = set()
        self.visited_coords = set()
        self.coords = ()
        self.last_event_count = 0
        self.last_hp = 1.0
        self.last_party_levels = 0
        self.last_enemy_hp = 0.0
        self.last_dex_count = 0
        self.has_anti_rock_bonus = False # Flag for one-time Nidoran/Mankey bonus
        self.step_count = 0
        
        self.max_steps = 2048 * 8 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if hasattr(self, 'pyboy'): self.pyboy.stop()

        window_type = "null" if self.render_mode == 'rgb_array' else "SDL2"
        self.pyboy = PyBoy(self.rom_path, window=window_type)
        if self.render_mode == 'rgb_array': self.pyboy.set_emulation_speed(0)

        # Load state to skip intro
        state_path = "states/start.state"
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                self.pyboy.load_state(f)
        else:
            print("⚠️ Iniciando desde el principio (No se encontró start.state)")

        # Reset metrics
        self.visited_maps = set()
        self.visited_coords = set()
        self.step_count = 0
        self.has_anti_rock_bonus = False
        
        # Initial normalized readings
        self.last_hp = self._read_hp() / 700.0
        self.last_party_levels = self._read_party_levels()
        self.last_event_count = self._read_event_count()
        self.last_enemy_hp = self._read_enemy_hp() / 700.0
        self.last_dex_count = self._read_dex_count()
        
        self.visited_maps.add(self.pyboy.memory[self.MEM_MAP_ID])

        return self._get_obs(), {}

    def step(self, action_idx):
        self.step_count += 1
        
        action = self.valid_actions[action_idx]
        self.pyboy.button(action)
        self.pyboy.tick(24) 

        if self.render_callback: self.render_callback(action_idx)

        obs = self._get_obs()
        reward = self._compute_reward()

        terminated = False
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Screen processing
        screen = self.pyboy.screen.ndarray 
        screen = resize(screen, (self.screen_height, self.screen_width), anti_aliasing=False, preserve_range=True)
        screen = screen.astype(np.uint8)
        if screen.shape[2] == 4: screen = screen[:, :, :3]
        screen = np.moveaxis(screen, 2, 0)

        # RAM normalization for the AI brain
        ram_data = np.array([
            np.clip(self.pyboy.memory[self.MEM_X_COORD] / 255.0, 0.0, 1.0),
            np.clip(self.pyboy.memory[self.MEM_Y_COORD] / 255.0, 0.0, 1.0),
            np.clip(self.pyboy.memory[self.MEM_MAP_ID] / 255.0, 0.0, 1.0),
            np.clip(self._read_hp() / 700.0, 0.0, 1.0),
            np.clip(self._read_enemy_hp() / 700.0, 0.0, 1.0),
            np.clip(self._read_party_levels() / 100.0, 0.0, 1.0),
            1.0 if self.pyboy.memory[self.MEM_IS_IN_BATTLE] > 0 else 0.0
        ], dtype=np.float32)

        return {'screen': screen, 'ram': ram_data}

    def _compute_reward(self):
        reward = 0
        
        # 1. STORY PROGRESS (Event Flags)
        current_event_count = self._read_event_count()
        if current_event_count > self.last_event_count:
            reward += (current_event_count - self.last_event_count) * 20.0
            self.last_event_count = current_event_count

        # 2. MAP EXPLORATION (New areas)
        map_id = self.pyboy.memory[self.MEM_MAP_ID]
        if map_id not in self.visited_maps:
            self.visited_maps.add(map_id)
            reward += 5.0

        # 3. CAPTURE AND POKEDEX (Encourages party diversity)
        current_dex = self._read_dex_count()
        if current_dex > self.last_dex_count:
            reward += 15.0 # Reward for catching any Pokemon
            self.last_dex_count = current_dex

        # 4. KEY PARTY REWARD (Nidoran M=03, Mankey=57/0x39)
        # This guides the AI to find solutions for Brock subtly
        if not self.has_anti_rock_bonus:
            party = self.pyboy.memory[self.MEM_PARTY_SPECIES : self.MEM_PARTY_SPECIES + 6]
            if 3 in party or 57 in party:
                reward += 25.0
                self.has_anti_rock_bonus = True
                print("💎 Bonus de equipo 'Anti-Roca' detectado!")

        # 5. COMBAT (Damage to enemy)
        curr_enemy_hp = self._read_enemy_hp()
        last_enemy_hp_raw = self.last_enemy_hp * 700.0
        if self.pyboy.memory[self.MEM_IS_IN_BATTLE]:
            if last_enemy_hp_raw > curr_enemy_hp:
                reward += (last_enemy_hp_raw - curr_enemy_hp) * 0.2
            self.last_enemy_hp = np.clip(curr_enemy_hp / 700.0, 0.0, 1.0)
        else:
            self.last_enemy_hp = 0.0

        # 6. SURVIVAL AND LOCAL EXPLORATION
        # Soft penalty for standing still (loops)
        coord = (self.pyboy.memory[self.MEM_X_COORD], self.pyboy.memory[self.MEM_Y_COORD], map_id)
        self.coords = coord
        if coord not in self.visited_coords:
            self.visited_coords.add(coord)
            reward += 0.02
        else:
            reward -= 0.001
            
        return reward

    # --- MEMORY READING FUNCTIONS ---
    def _read_hp(self):
        return (self.pyboy.memory[self.MEM_MY_HP_HIGH] << 8) + self.pyboy.memory[self.MEM_MY_HP_LOW]

    def _read_enemy_hp(self):
        return (self.pyboy.memory[self.MEM_ENEMY_HP_HIGH] << 8) + self.pyboy.memory[self.MEM_ENEMY_HP_LOW]
    
    def _read_party_levels(self):
        return self.pyboy.memory[self.MEM_PARTY_LEVELS] 

    def _read_event_count(self):
        event_bytes = self.pyboy.memory[self.MEM_EVENT_FLAGS_START : self.MEM_EVENT_FLAGS_END]
        return sum(bin(byte).count('1') for byte in event_bytes)

    def _read_dex_count(self):
        # Counts Pokemon owned in Pokedex
        dex_bytes = self.pyboy.memory[self.MEM_POKEDEX_OWNED : self.MEM_POKEDEX_OWNED + 19]
        return sum(bin(byte).count('1') for byte in dex_bytes)

    def render(self):
        return self.pyboy.screen.ndarray
    
    def set_render_callback(self, callback):
        self.render_callback = callback

    def close(self):
        if hasattr(self, 'pyboy') and self.pyboy:
            self.pyboy.stop()