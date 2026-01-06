"""
Módulo de Procesamiento de Audio
Maneja carga, guardado y operaciones básicas sobre archivos de audio
"""

import numpy as np
import soundfile as sf
from typing import Tuple, Optional

class AudioProcessor:
    """
    Procesador de archivos de audio.
    Soporta lectura y escritura de archivos WAV. 
    """
    
    @staticmethod
    def load_audio(filepath: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Carga un archivo de audio.
        
        Args:
            filepath: Ruta al archivo de audio
            target_sr:  Frecuencia de muestreo objetivo (None = usar original)
            
        Returns: 
            Tuple (señal_audio, sample_rate)
        """
        try:
            # Cargar audio
            signal, sample_rate = sf.read(filepath, dtype='float32')
            
            # Si es estéreo, convertir a mono
            if len(signal.shape) > 1:
                signal = np.mean(signal, axis=1)
            
            # Remuestrear si es necesario
            if target_sr is not None and target_sr != sample_rate: 
                signal = AudioProcessor. resample(signal, sample_rate, target_sr)
                sample_rate = target_sr
            
            return signal, sample_rate
            
        except Exception as e:
            raise IOError(f"Error al cargar audio: {e}")
    
    @staticmethod
    def save_audio(filepath: str, signal: np.ndarray, sample_rate: int):
        """
        Guarda una señal de audio en archivo WAV.
        
        Args:
            filepath: Ruta donde guardar el archivo
            signal:  Señal de audio
            sample_rate: Frecuencia de muestreo
        """
        try:
            # Normalizar señal a rango [-1, 1]
            signal_normalized = signal / np.max(np.abs(signal))
            
            # Guardar archivo
            sf. write(filepath, signal_normalized, sample_rate)
            
        except Exception as e:
            raise IOError(f"Error al guardar audio: {e}")
    
    @staticmethod
    def resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Remuestrea una señal a diferente frecuencia de muestreo.
        
        Args:
            signal:  Señal original
            orig_sr: Frecuencia de muestreo original
            target_sr: Frecuencia de muestreo objetivo
            
        Returns:
            Señal remuestreada
        """
        from scipy import signal as scipy_signal
        
        # Calcular número de muestras objetivo
        n_samples = int(len(signal) * target_sr / orig_sr)
        
        # Remuestrear
        resampled = scipy_signal.resample(signal, n_samples)
        
        return resampled
    
    @staticmethod
    def normalize_audio(signal:  np.ndarray, target_level: float = 0.9) -> np.ndarray:
        """
        Normaliza el nivel de audio.
        
        Args:
            signal: Señal de audio
            target_level: Nivel objetivo (0 a 1)
            
        Returns:
            Señal normalizada
        """
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            return signal * (target_level / max_val)
        return signal
    
    @staticmethod
    def compute_rms(signal: np.ndarray) -> float:
        """
        Calcula el RMS (Root Mean Square) de una señal.
        
        CONCEPTO MATEMÁTICO:
        RMS = sqrt(mean(x²))
        
        Representa el nivel promedio de energía. 
        
        Args:
            signal: Señal de audio
            
        Returns:
            Valor RMS
        """
        return np.sqrt(np.mean(signal**2))
    
    @staticmethod
    def compute_peak_level(signal: np.ndarray) -> float:
        """
        Calcula el nivel de pico en dB.
        
        Args:
            signal: Señal de audio
            
        Returns: 
            Nivel de pico en dB
        """
        peak = np.max(np.abs(signal))
        if peak > 0:
            return 20 * np.log10(peak)
        return -np.inf
    
    @staticmethod
    def generate_test_tone(frequency: float, duration: float, 
                          sample_rate: int = 44100) -> np.ndarray:
        """
        Genera un tono de prueba senoidal.
        
        CONCEPTO MATEMÁTICO - SERIE DE FOURIER:
        Un tono puro es un sinusoide: 
        
        x(t) = A · sin(2πft + φ)
        
        Su transformada de Fourier es un impulso en f. 
        
        Args:
            frequency: Frecuencia en Hz
            duration: Duración en segundos
            sample_rate: Frecuencia de muestreo
            
        Returns: 
            Señal de audio
        """
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np. sin(2 * np.pi * frequency * t)
        return signal