# 🎵 Ecualizador de Audio Multibanda + Espectrograma

**Proyecto de Matemáticas Avanzadas**: Análisis de Fourier y Procesamiento de Señales

---

## 📋 Descripción

Este proyecto implementa un ecualizador de audio multibanda profesional con análisis espectral completo. Aplica conceptos fundamentales de:

- **Transformada de Fourier (DFT/FFT)**
- **Series de Fourier**
- **Teorema de Parseval**
- **Análisis Tiempo-Frecuencia (STFT)**
- **Filtrado y Convolución**
- **Propiedades de la Transformada de Fourier**

---

## 🎯 Características Principales

### ✅ Funcionalidades Implementadas

1. **Carga de Audio**: Soporte para archivos `.wav`
2. **Ecualización Multibanda**: División del espectro en 5 bandas de frecuencia
3. **Análisis de Fourier**: FFT, IFFT y verificación de Parseval
4. **Espectrogramas**: Visualización tiempo-frecuencia mediante STFT
5. **Visualizaciones**: Gráficas en dominios de tiempo y frecuencia
6. **Exportación**: Guardado de audio procesado y gráficas

---

## 🧮 Conceptos Matemáticos Implementados

### 1. Transformada Discreta de Fourier (DFT)

```
X[k] = Σ(n=0 to N-1) x[n] · e^(-j·2π·k·n/N)
```

**Implementación**: `fourier_analysis.py` - método `compute_fft()`

### 2. Transformada Inversa (IFFT)

```
x[n] = (1/N) · Σ(k=0 to N-1) X[k] · e^(j·2π·k·n/N)
```

**Propiedad**: IFFT(FFT(x)) = x (reconstrucción perfecta)

**Implementación**: `fourier_analysis.py` - método `compute_ifft()`

### 3. Teorema de Parseval

```
Σ|x[n]|² = (1/N) · Σ|X[k]|²
```

**Significado**: La energía total se conserva en ambos dominios.

**Implementación**: `fourier_analysis.py` - método `verify_parseval()`

### 4. Propiedad de Convolución

```
F{x * h} = F{x} · F{h}
```

**Aplicación**: Filtrado en frecuencia mediante multiplicación.

**Implementación**: `equalizer.py` - método `equalize()`

### 5. STFT (Short-Time Fourier Transform)

```
STFT{x[n]}(m,ω) = Σ x[n] · w[n-m] · e^(-jωn)
```

**Análisis tiempo-frecuencia**: Ventanas deslizantes con FFT.

**Implementación**: `spectrogram. py` - método `compute_spectrogram()`

---

## 📦 Instalación

### Requisitos

- Python 3.8+
- pip

### Pasos

```bash
# Clonar o descargar el proyecto
cd audio_equalizer

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🚀 Uso

### Modo Demostración (Recomendado para Primera Ejecución)

```bash
python main.py
```

Este modo:

- Genera una señal de prueba con 5 componentes frecuenciales (100Hz, 500Hz, 1kHz, 3kHz, 8kHz)
- Aplica ecualización predefinida
- Genera todos los análisis y visualizaciones
- Verifica el Teorema de Parseval

### Modo Interactivo

```bash
python main.py -i
```

Permite:

- Cargar tu propio archivo de audio `.wav`
- Configurar ganancias personalizadas para cada banda
- Procesar y analizar el audio

---

## 📊 Salidas Generadas

El programa genera los siguientes archivos:

### 🎵 Audio

- `test_signal.wav` - Señal original de prueba
- `test_signal_equalized.wav` - Señal ecualizada
- `[nombre]_equalized.wav` - Audio procesado (modo interactivo)

### 📈 Visualizaciones

1. **comparison.png**

   - Formas de onda (tiempo)
   - Espectros de frecuencia (dB)
   - Comparación antes/después

2. **spectrograms_comparison.png**

   - Espectrograma original
   - Espectrograma ecualizado
   - Análisis tiempo-frecuencia (STFT)

3. **band_energies.png**

   - Distribución de energía por banda
   - Comparación original vs ecualizado

4. **equalizer_response.png**

   - Curva de respuesta en frecuencia
   - Bandas marcadas con colores
   - Ganancias aplicadas en dB

5. **parseval_verification.png**
   - Verificación del Teorema de Parseval
   - Energías en dominio del tiempo y frecuencia
   - Error relativo

---

## 🎚️ Configuración de Bandas

El ecualizador divide el espectro audible en 5 bandas logarítmicas:

| Banda | Rango Frecuencial | Descripción                    |
| ----- | ----------------- | ------------------------------ |
| 1     | 20 - 160 Hz       | Sub-Bass/Bass (patada, bajo)   |
| 2     | 160 - 630 Hz      | Bass/Mid-Low (cuerpo, calidez) |
| 3     | 630 - 2. 5 kHz    | Midrange (voz, melodía)        |
| 4     | 2.5 - 10 kHz      | Mid-High/Presence (claridad)   |
| 5     | 10 - 20 kHz       | Brilliance (aire, brillo)      |

### Ejemplos de Configuraciones

#### 🎸 Realce de Bajos (Rock/EDM)

```
+6 +3 0 -3 -3
```

#### 🎙️ Claridad Vocal

```
-3 -3 +3 +6 0
```

#### 📻 Radio FM

```
-6 +3 +6 +3 -6
```

#### 🎼 Plano/Neutro

```
0 0 0 0 0
```

---

## 🔬 Análisis Matemático Detallado

### Series de Fourier

Una señal periódica se descompone en suma de sinusoides:

```
x(t) = a₀ + Σ[aₙ·cos(nω₀t) + bₙ·sin(nω₀t)]
```

**Forma compleja**:

```
x(t) = Σ cₙ · e^(jnω₀t)
```

donde:

```
cₙ = (1/T) ∫ x(t) · e^(-jnω₀t) dt
```

### Propiedades de la Transformada de Fourier

#### Linealidad

```
F{a·x₁(t) + b·x₂(t)} = a·X₁(f) + b·X₂(f)
```

#### Desplazamiento en Frecuencia

```
x(t) · e^(j2πf₀t) ↔ X(f - f₀)
```

**Implementación**: `fourier_analysis.py` - método `apply_frequency_shift()`

#### Escalamiento Temporal

```
x(at) ↔ (1/|a|) · X(f/a)
```

**Implementación**: `fourier_analysis.py` - método `apply_time_scaling()`

#### Convolución

```
y(t) = x(t) * h(t) ↔ Y(f) = X(f) · H(f)
```

**Aplicación**: Filtrado eficiente en frecuencia.

---

## 🎓 Casos de Uso Educativos

### Experimento 1: Verificación de Parseval

```python
from fourier_analysis import FourierAnalyzer
import numpy as np

# Crear señal
signal = np.random.randn(1000)
analyzer = FourierAnalyzer(44100)

# FFT
fft_signal = np.fft.fft(signal)

# Verificar Parseval
is_valid, time_energy, freq_energy = analyzer.verify_parseval(signal, fft_signal)

print(f"Energía (tiempo): {time_energy}")
print(f"Energía (frecuencia): {freq_energy}")
print(f"Parseval verificado: {is_valid}")
```

### Experimento 2: Análisis de Voz vs Música

Procesar dos archivos diferentes:

```bash
# Voz
python main.py -i
# Archivo:  voice.wav
# Ganancias: -3 0 +6 +3 -3

# Música
python main. py -i
# Archivo:  music.wav
# Ganancias: +6 +3 0 -3 -3
```

**Observaciones esperadas**:

- **Voz**: Mayor energía en mid-high (2-4 kHz)
- **Música**: Energía distribuida en todo el espectro

---

## 📚 Estructura del Código

```
audio_equalizer/
├── fourier_analysis.py    # Análisis de Fourier (FFT, IFFT, Parseval)
├── equalizer.py           # Ecualización multibanda
├── spectrogram.py         # STFT y espectrogramas
├── audio_processor.py     # I/O de audio
├── visualizer.py          # Generación de gráficas
├── main.py                # Aplicación principal
├── requirements.txt       # Dependencias
└── README.md             # Este archivo
```

---

## 🧪 Validación Matemática

### Test 1: Reconstrucción Perfecta

```python
# FFT seguida de IFFT debe recuperar la señal original
signal_reconstructed = np.fft.ifft(np.fft.fft(signal)).real
assert np.allclose(signal, signal_reconstructed)
```

### Test 2: Teorema de Parseval

```python
# Energía en tiempo = energía en frecuencia
time_energy = np.sum(np.abs(signal)**2)
freq_energy = np.sum(np.abs(fft_signal)**2) / len(signal)
assert np.isclose(time_energy, freq_energy)
```

### Test 3: Linealidad de la Transformada

```python
# F{a·x₁ + b·x₂} = a·F{x₁} + b·F{x₂}
a, b = 2.0, 3.0
lhs = np.fft.fft(a*x1 + b*x2)
rhs = a*np.fft.fft(x1) + b*np.fft.fft(x2)
assert np.allclose(lhs, rhs)
```

---

## 🎨 Ejemplos de Resultados

### Señal Original vs Ecualizada

![Comparison](comparison.png)

**Interpretación**:

- **Tiempo**: Cambios en amplitud y forma
- **Frecuencia**: Bandas realzadas/atenuadas según configuración

### Espectrogramas

![Spectrograms](spectrograms_comparison.png)

**Interpretación**:

- **Eje X**: Tiempo
- **Eje Y**: Frecuencia (escala logarítmica)
- **Color**: Intensidad (dB)

---

## 🔧 Personalización Avanzada

### Cambiar Número de Bandas

```python
# En main.py, modificar:
self.n_bands = 10  # Por defecto:  5
```

### Ajustar Resolución del Espectrograma

```python
# Mayor resolución frecuencial (ventana más grande)
self.window_size = 4096  # Por defecto: 2048

# Mayor resolución temporal (salto más pequeño)
self.hop_length = 256  # Por defecto: 512
```

---

## 📖 Referencias Matemáticas

1. **Transformada de Fourier**

   - Oppenheim & Schafer. "Discrete-Time Signal Processing"

2. **Teorema de Parseval**

   - Parseval des Chênes, M.-A. (1806)

3. **STFT**

   - Allen, J. (1977). "Short term spectral analysis"

4. **Procesamiento Digital de Señales**
   - Proakis & Manolakis. "Digital Signal Processing"

---

## 🤝 Contribuciones

Este proyecto es de código abierto para fines educativos. Conceptos implementados:

✅ Series de Fourier (trigonométricas y complejas)  
✅ DFT/FFT  
✅ Teorema de Parseval  
✅ Reconstrucción de señales  
✅ Propiedades de la Transformada de Fourier  
✅ Convolución y filtrado

---

## 📝 Licencia

Proyecto educativo - Matemáticas Avanzadas

---

## 👨‍💻 Autor

Proyecto desarrollado para demostrar conceptos de:

- Transformadas de Fourier
- Análisis de señales
- Procesamiento digital de audio

---

## 🆘 Soporte

Para problemas o preguntas:

1. Verificar que todas las dependencias están instaladas
2. Usar Python 3.8 o superior
3. Probar primero el modo demostración

---

**¡Disfruta experimentando con Fourier!** 🎵📊🔬
