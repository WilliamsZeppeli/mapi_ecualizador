"""
Módulo de Análisis de Fourier
Implementa DFT, FFT, propiedades de la Transformada de Fourier y Teorema de Parseval
"""

import numpy as np
from typing import Tuple, Optional

class FourierAnalyzer:
    """
    Clase para realizar análisis de Fourier sobre señales de audio.
    Implementa conceptos fundamentales de Series y Transformadas de Fourier.
    """
    
    def __init__(self, sample_rate: int):
        """
        Inicializa el analizador de Fourier.
        
        Args:
            sample_rate: Frecuencia de muestreo en Hz
        """
        self. sample_rate = sample_rate
        
    def compute_fft(self, signal: np. ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula la FFT de una señal.
        
        CONCEPTO MATEMÁTICO:
        La Transformada Rápida de Fourier (FFT) es una implementación eficiente
        de la Transformada Discreta de Fourier (DFT):
        
        X[k] = Σ(n=0 to N-1) x[n] * e^(-j*2π*k*n/N)
        
        donde: 
        - X[k]:  Componente de frecuencia k
        - x[n]: Muestra temporal n
        - N: Número total de muestras
        - j: Unidad imaginaria
        
        Args:
            signal:  Señal en dominio del tiempo
            
        Returns:
            Tuple (frecuencias, magnitudes de la FFT)
        """
        # Aplicar FFT
        fft_result = np.fft.fft(signal)
        
        # Calcular magnitud (espectro de amplitud)
        magnitude = np. abs(fft_result)
        
        # Calcular frecuencias correspondientes
        n_samples = len(signal)
        frequencies = np.fft.fftfreq(n_samples, d=1/self.sample_rate)
        
        # Retornar solo la mitad positiva (espectro simétrico)
        positive_freq_idx = frequencies >= 0
        
        return frequencies[positive_freq_idx], magnitude[positive_freq_idx]
    
    def compute_ifft(self, fft_signal: np.ndarray) -> np.ndarray:
        """
        Calcula la IFFT (Transformada Inversa de Fourier).
        
        CONCEPTO MATEMÁTICO:
        La Transformada Inversa permite reconstruir la señal temporal: 
        
        x[n] = (1/N) * Σ(k=0 to N-1) X[k] * e^(j*2π*k*n/N)
        
        Propiedad de INVERSIÓN:  IFFT(FFT(x)) = x
        
        Args:
            fft_signal: Señal en dominio de la frecuencia
            
        Returns:
            Señal reconstruida en dominio del tiempo
        """
        return np.fft.ifft(fft_signal).real
    
    def verify_parseval(self, signal: np. ndarray, fft_signal: np.ndarray, 
                       tolerance: float = 1e-6) -> Tuple[bool, float, float]:
        """
        Verifica el Teorema de Parseval. 
        
        CONCEPTO MATEMÁTICO - TEOREMA DE PARSEVAL:
        La energía total de una señal es igual en el dominio del tiempo
        y en el dominio de la frecuencia:
        
        Σ|x[n]|² = (1/N) * Σ|X[k]|²
        
        Este teorema garantiza que la transformación preserva la energía.
        
        Args:
            signal:  Señal en dominio del tiempo
            fft_signal: Señal en dominio de la frecuencia
            tolerance: Tolerancia para la comparación
            
        Returns: 
            Tuple (cumple_parseval, energía_temporal, energía_frecuencial)
        """
        # Energía en dominio del tiempo
        time_energy = np.sum(np.abs(signal)**2)
        
        # Energía en dominio de la frecuencia
        freq_energy = np.sum(np.abs(fft_signal)**2) / len(signal)
        
        # Verificar si se cumple Parseval
        is_valid = np.abs(time_energy - freq_energy) < tolerance
        
        return is_valid, time_energy, freq_energy
    
    def apply_frequency_shift(self, signal: np.ndarray, 
                             shift_hz: float) -> np.ndarray:
        """
        Aplica desplazamiento en frecuencia.
        
        CONCEPTO MATEMÁTICO - PROPIEDAD DE DESPLAZAMIENTO:
        Un desplazamiento en frecuencia corresponde a modulación: 
        
        x[n] * e^(j*2π*f₀*n/fs) ↔ X(f - f₀)
        
        donde f₀ es el desplazamiento de frecuencia. 
        
        Args:
            signal: Señal original
            shift_hz: Desplazamiento en Hz
            
        Returns:
            Señal con frecuencia desplazada
        """
        n_samples = len(signal)
        t = np.arange(n_samples) / self.sample_rate
        
        # Modulación compleja
        modulation = np.exp(1j * 2 * np.pi * shift_hz * t)
        
        # Aplicar desplazamiento
        shifted = signal * modulation
        
        return shifted. real
    
    def apply_time_scaling(self, signal: np.ndarray, 
                          scale_factor: float) -> np.ndarray:
        """
        Aplica escalamiento temporal.
        
        CONCEPTO MATEMÁTICO - PROPIEDAD DE ESCALAMIENTO:
        Un escalamiento en el tiempo produce escalamiento inverso en frecuencia:
        
        x(a*t) ↔ (1/|a|) * X(f/a)
        
        Args:
            signal: Señal original
            scale_factor: Factor de escalamiento (>1 acelera, <1 desacelera)
            
        Returns: 
            Señal escalada en tiempo
        """
        from scipy import interpolate
        
        n_samples = len(signal)
        original_time = np.arange(n_samples)
        new_time = np.linspace(0, n_samples-1, int(n_samples/scale_factor))
        
        # Interpolación para remuestreo
        f = interpolate.interp1d(original_time, signal, kind='cubic', 
                                fill_value='extrapolate')
        
        return f(new_time)
    
    def compute_stft(self, signal:  np.ndarray, 
                    window_size: int = 2048, 
                    hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula la Transformada de Fourier de Tiempo Corto (STFT).
        
        CONCEPTO MATEMÁTICO - STFT:
        Análisis tiempo-frecuencia mediante ventanas deslizantes:
        
        STFT{x[n]}(m,ω) = Σ x[n] * w[n-m] * e^(-jωn)
        
        donde w[n] es una función de ventana (ej:  Hamming, Hann)
        
        Args:
            signal: Señal de audio
            window_size: Tamaño de ventana en muestras
            hop_length:  Salto entre ventanas
            
        Returns: 
            Tuple (STFT compleja, tiempos, frecuencias)
        """
        # Ventana de Hann (suave en los extremos)
        window = np.hanning(window_size)
        
        # Número de frames
        n_frames = 1 + (len(signal) - window_size) // hop_length
        
        # Inicializar matriz STFT
        stft_matrix = np.zeros((window_size // 2 + 1, n_frames), 
                              dtype=complex)
        
        # Calcular STFT para cada ventana
        for i in range(n_frames):
            start = i * hop_length
            end = start + window_size
            
            if end > len(signal):
                break
                
            # Extraer segmento y aplicar ventana
            segment = signal[start:end] * window
            
            # FFT del segmento
            fft_segment = np.fft.fft(segment)
            
            # Guardar solo frecuencias positivas
            stft_matrix[: , i] = fft_segment[:window_size // 2 + 1]
        
        # Calcular ejes de tiempo y frecuencia
        times = np.arange(n_frames) * hop_length / self.sample_rate
        frequencies = np.fft.fftfreq(window_size, 1/self.sample_rate)[:window_size // 2 + 1]
        
        return stft_matrix, times, frequencies
    
    def compute_phase_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el espectro de fase.
        
        CONCEPTO MATEMÁTICO: 
        La FFT produce números complejos X[k] = A[k] * e^(jφ[k])
        donde φ[k] es la fase. 
        
        Args:
            signal: Señal de audio
            
        Returns:
            Tuple (frecuencias, fases)
        """
        fft_result = np.fft.fft(signal)
        n_samples = len(signal)
        frequencies = np.fft.fftfreq(n_samples, d=1/self.sample_rate)
        
        # Calcular fase
        phase = np. angle(fft_result)
        
        # Solo frecuencias positivas
        positive_freq_idx = frequencies >= 0
        
        return frequencies[positive_freq_idx], phase[positive_freq_idx]