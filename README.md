# 🧠 Project Red-RL: Autonomous Pokémon Yellow Agent

![Status](https://img.shields.io/badge/Status-Active_Development-success)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Framework](https://img.shields.io/badge/RL-Stable--Baselines3-orange)
![Emulator](https://img.shields.io/badge/Emulator-PyBoy-purple)

> **Arquitectura:** PPO (Proximal Policy Optimization) + Decodificación de Estado Neuro-Simbólica.
> **Objetivo:** Entrenar un agente de Inteligencia Artificial capaz de completar *Pokémon Edición Amarilla* desde cero, sin conocimiento previo (Tabula Rasa).

## 📋 Descripción Técnica

Este proyecto implementa una arquitectura de **Aprendizaje por Refuerzo Profundo (Deep RL)** diseñada para resolver entornos de RPG complejos con un horizonte temporal extremadamente largo. 

A diferencia de los enfoques puramente visuales (que solo "ven" píxeles), este sistema utiliza un **Espacio de Observación Híbrido** que combina:
1.  **Visión (CNN):** Procesamiento de la pantalla para entender la geometría local y obstáculos.
2.  **Memoria (RAM):** Lectura directa de la memoria del sistema emulado para obtener contexto global (coordenadas, mapa ID, medallas).

### ✨ Características Clave

* **⚡ Emulación Acelerada:** Utiliza `PyBoy` como entorno base sin interfaz gráfica durante el entrenamiento, permitiendo velocidades superiores a **1000 FPS**.
* **👁️ Observación Híbrida:** El agente no solo "ve", sino que "sabe" dónde está gracias a la inyección de datos hexadecimales de la RAM en la red neuronal.
* **🗺️ Exploración Eficiente:** Sistema de recompensas densas basado en coordenadas únicas visitadas $(x, y)$ para mitigar el problema de recompensas dispersas (Sparse Rewards).
* **🎥 Streamer-Ready Architecture:** Infraestructura asimétrica que permite entrenar a máxima velocidad en segundo plano mientras se visualiza una instancia clonada a 60 FPS fluidos para transmisión en vivo.
* **⚙️ Optimización de Hardware:** Implementación de `SleepCallback` y gestión de hilos (`OMP_NUM_THREADS=1`) para permitir entrenamiento y streaming simultáneo en CPUs de consumo (ej. i5/Ryzen 5) sin congelar el sistema.

## 🛠️ Stack Tecnológico

| Componente | Tecnología | Uso |
| :--- | :--- | :--- |
| **Lenguaje** | Python 3.11 | Lógica del núcleo |
| **RL Framework** | Stable-Baselines3 | Implementación de PPO y Vectorización de Entornos |
| **Emulador** | PyBoy | Interfaz de bajo nivel con la ROM de Game Boy |
| **Visión** | OpenCV / NumPy | Preprocesamiento de frames y renderizado |
| **Logging** | TensorBoard | Monitoreo de métricas (Loss, Reward, Entropy) en tiempo real |

## 🚀 Instalación y Configuración

### Prerrequisitos
* **Python 3.11** (Se recomienda usar Conda).
* **ROM de Pokémon Yellow:** Debe nombrarse exactamente `PokemonYellow.gb` y colocarse en la carpeta `roms/`.

### Guía Paso a Paso

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/pokemon-rl.git](https://github.com/tu-usuario/pokemon-rl.git)
    cd pokemon-rl
    ```

2.  **Crear entorno virtual:**
    ```bash
    conda create -n poke-rl python=3.11
    conda activate poke-rl
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install gymnasium pyboy shimmy stable-baselines3[extra] opencv-python torch-directml
    ```

4.  **Generar Estado Inicial (Skip Intro):**
    Para evitar que el agente pierda horas de entrenamiento en el menú de "Nueva Partida", generamos un estado guardado justo después de la intro.
    ```bash
    python src/utils/create_initial_state.py
    ```
    *Instrucción: Juega manualmente hasta tener el control del personaje en la habitación de Ash y cierra la ventana.*

## 🏃‍♂️ Ejecución y Flujo de Trabajo

Este proyecto está diseñado para funcionar en dos terminales simultáneas: una para el "Cerebro" (Entrenamiento) y otra para los "Ojos" (Streaming).

### 1. Entrenamiento (The Brain) 🧠
Inicia el bucle de entrenamiento masivo. El sistema es "headless" (sin ventana) para maximizar velocidad.
* **Uso de CPU:** Optimizado para usar 1-2 núcleos de forma intensiva.
* **Guardado:** Genera checkpoints automáticos en `experiments/`.

```bash
python train.py

Nota: Usa Ctrl + C en cualquier momento para pausar y realizar un "Guardado de Emergencia" seguro.

2. Visualización
Muestra al agente jugando en tiempo real a 60 FPS. Este script detecta automáticamente cuando train.py guarda un nuevo modelo "más inteligente" y lo carga en caliente ("Hot-Reload") sin cerrar la ventana.

Bash
python watch_continuous.py

3. Monitoreo (Analytics) 📊
Para ver gráficas de recompensa, pérdida (loss) y entropía:

Bash
tensorboard --logdir experiments/poke_ppo_v1/logs

🧠 Arquitectura del Agente
Espacio de Acción (Action Space)
Discreto (6): [DOWN, LEFT, RIGHT, UP, A, B].

Optimización: Se deshabilitaron Start y Select para reducir el ruido estocástico y evitar que el agente se quede atascado en menús.

Sistema de Recompensa (Reward Shaping)
La función de recompensa actual incentiva la curiosidad pura:
$$R_t = R_{exploración} + R_{eventos}$$
Exploración: +1.0 punto por cada coordenada única $(x, y)$ visitada por mapa. Esto empuja al agente a recorrer todo el mapa disponible.
Penalización de Inactividad: (Implícita) Al no haber recompensas por quedarse quieto, el algoritmo de maximización fuerza el movimiento.

📂 Estructura del Proyecto

pokemon-rl/
├── config/                 # Hiperparámetros y configuraciones
├── experiments/            # Checkpoints (.zip) y Logs de TensorBoard
├── roms/                   # Archivos del juego (.gb)
├── states/                 # Archivos .state (Save States de PyBoy)
├── src/
│   ├── environment/
│   │   ├── pokemon_env.py  # Wrapper Gym (Lógica de RAM, Visión y Smooth Ticking)
│   │   └── ...
│   ├── utils/
│   │   ├── memory_reader.py # Extracción de direcciones Hex de la RAM
│   │   └── ...
├── train.py                # Script de entrenamiento (Backend)
├── watch_continuous.py     # Script de visualización para Stream (Frontend)
└── README.md               # Documentación

🔮 Roadmap
[ ] Implementar HippoTorch (S4) para memoria a largo plazo.

[ ] Integrar un VLM (Vision Language Model) para lectura de diálogos en pantalla.