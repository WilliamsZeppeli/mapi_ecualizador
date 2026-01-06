"""
Archivo de Configuración
Parámetros globales del proyecto
"""

# Parámetros de audio
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_DURATION = 3.0  # segundos

# Parámetros del ecualizador
N_BANDS = 5  # Número de bandas de frecuencia
FREQ_MIN = 20  # Hz - Límite inferior audible
FREQ_MAX = 20000  # Hz - Límite superior audible

# Parámetros de STFT
WINDOW_SIZE = 2048  # Muestras
HOP_LENGTH = 512  # Muestras
WINDOW_TYPE = 'hann'  # 'hann', 'hamming', 'blackman'

# Parámetros de visualización
FIGSIZE_LARGE = (15, 10)
FIGSIZE_MEDIUM = (12, 6)
FIGSIZE_SMALL = (10, 4)
DPI = 300

# Colores para visualizaciones
COLOR_ORIGINAL = 'steelblue'
COLOR_EQUALIZED = 'coral'
COLOR_SPECTRUM = 'darkgreen'

# Rangos de ganancia
GAIN_MIN_DB = -20
GAIN_MAX_DB = +20

# Tolerancia para verificaciones numéricas
PARSEVAL_TOLERANCE = 1e-6
RECONSTRUCTION_TOLERANCE = 1e-10

# Nombres de archivos de salida
OUTPUT_COMPARISON = 'comparison.png'
OUTPUT_SPECTROGRAMS = 'spectrograms_comparison.png'
OUTPUT_BAND_ENERGIES = 'band_energies.png'
OUTPUT_EQUALIZER_RESPONSE = 'equalizer_response. png'
OUTPUT_PARSEVAL = 'parseval_verification.png'

# Configuraciones de audio de prueba
TEST_FREQUENCIES = [100, 500, 1000, 3000, 8000]  # Hz
TEST_AMPLITUDES = [0.5, 0.3, 0.4, 0.3, 0.2]  # Amplitudes relativas

# Presets de ecualización
EQ_PRESETS = {
    'flat': [0, 0, 0, 0, 0],
    'bass_boost': [+6, +3, 0, -3, -3],
    'vocal_clarity': [-3, -3, +3, +6, 0],
    'treble_boost': [-3, -3, 0, +3, +6],
    'v_shape': [+6, 0, -3, 0, +6],
    'radio':  [-6, +3, +6, +3, -6],
    'rock': [+5, +3, -1, +2, +3],
    'classical': [0, 0, 0, +2, +3],
    'jazz': [+3, +2, 0, +2, +3],
    'electronic': [+6, +4, 0, +2, +4]
}

# Descripciones de presets
EQ_PRESET_DESCRIPTIONS = {
    'flat': 'Sin modificación - respuesta plana',
    'bass_boost': 'Realza graves - Rock/EDM',
    'vocal_clarity': 'Claridad vocal - Podcasts',
    'treble_boost': 'Realza agudos - Brillo',
    'v_shape':  'Curva en V - Graves y agudos',
    'radio': 'Simulación de radio FM',
    'rock': 'Optimizado para rock',
    'classical':  'Optimizado para música clásica',
    'jazz': 'Optimizado para jazz',
    'electronic': 'Optimizado para música electrónica'
}