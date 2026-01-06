"""
Módulo de Espectrogramas
Implementa análisis tiempo-frecuencia mediante STFT
"""

import numpy as np
from typing import Tuple, Optional

class SpectrogramGenerator:
    """
    Generador de espectrogramas usando STFT (Short-Time Fourier Transform).
    Permite visualizar cómo evoluciona el contenido frecuencial en el tiempo.
    """
    
    def __init__(self, sample_rate: int):
        """
        Inicializa el generador de espectrogramas.
        
        Args:
            sample_rate: Frecuencia de muestreo en Hz
        """
        self. sample_rate = sample_rate
    
    def compute_spectrogram(self, signal: np.ndarray,
                           window_size: int = 2048,
                           hop_length: int = 512,
                           window_type: str = 'hann') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula el espectrograma de una señal.
        
        CONCEPTO MATEMÁTICO - STFT:
        El espectrograma es |STFT|², donde STFT se calcula como:
        
        X(m,ω) = Σ x[n] · w[n-m] · e^(-jωn)
        
        Ventaneo: 
        - Resuelve la incertidumbre tiempo-frecuencia
        - Ventanas comunes: Hann, Hamming, Blackman
        
        Args:
            signal:  Señal de audio
            window_size: Tamaño de ventana (afecta resolución frecuencial)
            hop_length: Salto entre ventanas (afecta resolución temporal)
            window_type: Tipo de ventana ('hann', 'hamming', 'blackman')
            
        Returns:
            Tuple (espectrograma, tiempos, frecuencias)
        """
        # Seleccionar función de ventana
        if window_type == 'hann': 
            window = np.hanning(window_size)
        elif window_type == 'hamming': 
            window = np.hamming(window_size)
        elif window_type == 'blackman':
            window = np.blackman(window_size)
        else:
            window = np.ones(window_size)
        
        # Número de frames
        n_frames = 1 + (len(signal) - window_size) // hop_length
        n_freqs = window_size // 2 + 1
        
        # Inicializar matriz de espectrograma
        spectrogram = np.zeros((n_freqs, n_frames))
        
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
            
            # Magnitud al cuadrado (espectro de potencia)
            power_spectrum = np.abs(fft_segment[: n_freqs])**2
            
            # Guardar en espectrograma
            spectrogram[:, i] = power_spectrum
        
        # Calcular ejes
        times = np.arange(n_frames) * hop_length / self.sample_rate
        frequencies = np.linspace(0, self.sample_rate/2, n_freqs)
        
        return spectrogram, times, frequencies
    
    def spectrogram_to_db(self, spectrogram: np.ndarray, 
                         ref:  float = 1.0, 
                         amin: float = 1e-10) -> np.ndarray:
        """
        Convierte espectrograma a escala de decibelios.
        
        CONCEPTO MATEMÁTICO: 
        La escala logarítmica (dB) es más adecuada para percepción: 
        
        S_dB = 10 · log₁₀(S / S_ref)
        
        Args:
            spectrogram:  Espectrograma en escala lineal
            ref: Valor de referencia
            amin: Valor mínimo para evitar log(0)
            
        Returns: 
            Espectrograma en dB
        """
        magnitude = np.maximum(spectrogram, amin)
        return 10 * np.log10(magnitude / ref)
    
    def compute_mel_spectrogram(self, signal: np.ndarray,
                               n_mels: int = 128,
                               window_size: int = 2048,
                               hop_length:  int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula el espectrograma Mel. 
        
        CONCEPTO MATEMÁTICO - ESCALA MEL:
        La escala Mel refleja mejor la percepción humana de frecuencias:
        
        mel(f) = 2595 · log₁₀(1 + f/700)
        
        Args:
            signal:  Señal de audio
            n_mels: Número de bandas Mel
            window_size: Tamaño de ventana
            hop_length: Salto entre ventanas
            
        Returns:
            Tuple (mel_spectrogram, tiempos, frecuencias_mel)
        """
        # Calcular espectrograma normal
        spec, times, freqs = self.compute_spectrogram(signal, window_size, hop_length)
        
        # Crear banco de filtros Mel
        mel_filters = self._create_mel_filterbank(n_mels, window_size, self.sample_rate)
        
        # Aplicar filtros Mel
        mel_spec = np.dot(mel_filters, spec)
        
        # Frecuencias Mel
        mel_freqs = self._hz_to_mel(freqs)
        
        return mel_spec, times, mel_freqs
    
    def _hz_to_mel(self, hz: np.ndarray) -> np.ndarray:
        """Convierte Hz a escala Mel."""
        return 2595 * np.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel: np.ndarray) -> np.ndarray:
        """Convierte escala Mel a Hz."""
        return 700 * (10**(mel / 2595) - 1)
    
    def _create_mel_filterbank(self, n_mels: int, n_fft: int, 
                              sample_rate: int) -> np.ndarray:
        """
        Crea banco de filtros Mel triangulares.
        
        Args:
            n_mels:  Número de filtros
            n_fft:  Tamaño FFT
            sample_rate:  Frecuencia de muestreo
            
        Returns:
            Matriz de filtros Mel (n_mels x (n_fft//2 + 1))
        """
        # Límites de frecuencia
        fmin = 0
        fmax = sample_rate / 2
        
        # Convertir a Mel
        mel_min = self._hz_to_mel(np.array([fmin]))[0]
        mel_max = self._hz_to_mel(np.array([fmax]))[0]
        
        # Crear puntos equiespaciados en escala Mel
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        
        # Convertir de vuelta a Hz
        hz_points = self._mel_to_hz(mel_points)
        
        # Convertir a bins de FFT
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
        
        # Crear filtros triangulares
        n_freqs = n_fft // 2 + 1
        filters = np.zeros((n_mels, n_freqs))
        
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            # Rampa ascendente
            for j in range(left, center):
                filters[i, j] = (j - left) / (center - left)
            
            # Rampa descendente
            for j in range(center, right):
                filters[i, j] = (right - j) / (right - center)
        
        return filters