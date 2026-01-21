import sys

# Direcciones de memoria RAM para Pokémon Yellow (Versión EN)
# Nota: Estas direcciones son específicas de Yellow. Red/Blue usan otras ligeramente distintas.
MEM_X_POS = 0xD362
MEM_Y_POS = 0xD361
MEM_MAP_ID = 0xD35E
MEM_BADGES = 0xD356  # Bitmask de medallas

class MemoryReader:
    """
    Clase encargada de extraer información estructurada directamente de la RAM
    de la Game Boy a través de PyBoy.
    """
    def __init__(self, pyboy_instance):
        self.pyboy = pyboy_instance

    def get_coordinate_x(self):
        return self.pyboy.memory[MEM_X_POS]

    def get_coordinate_y(self):
        return self.pyboy.memory[MEM_Y_POS]

    def get_map_id(self):
        return self.pyboy.memory[MEM_MAP_ID]
    
    def get_badges(self):
        return self.pyboy.memory[MEM_BADGES]

    def get_ram_state(self):
        """Devuelve un diccionario con el estado crítico actual."""
        return {
            "x": self.get_coordinate_x(),
            "y": self.get_coordinate_y(),
            "map_id": self.get_map_id(),
            "badges": self.get_badges()
        }