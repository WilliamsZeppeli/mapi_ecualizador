"""
Motor de Ecualización Multibanda
Implementa filtrado en frecuencia y ecualización por bandas
"""

import numpy as np
from typing import List, Tuple, Dict

class MultibandEqualizer:
    """
    Ecualizador multibanda que divide el espectro en K bandas
    y permite ajustar la ganancia de cada banda independientemente.
    """
    
    def __init__(self, sample_rate: int, n_bands: int = 5):
        """
        Inicializa el ecualizador.
        
        Args:
            sample_rate: Frecuencia de muestreo en Hz
            n_bands:  Número de bandas de frecuencia
        """
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self.bands = self._create_frequency_bands()
        self.gains = np.ones(n_bands)  # Ganancia inicial:  1 (sin cambio)
        
    def _create_frequency_bands(self) -> List[Tuple[float, float]]: 
        """
        Crea bandas de frecuencia logarítmicas.
        
        CONCEPTO MATEMÁTICO: 
        Dividimos el espectro audible (20Hz - 20kHz) en bandas logarítmicas
        porque la percepción humana del sonido es logarítmica.
        
        Para K bandas:
        f[i] = f_min * (f_max/f_min)^(i/K)
        
        Returns:
            Lista de tuplas (freq_inicio, freq_fin) para cada banda
        """
        f_min = 20  # Hz - límite inferior audible
        f_max = min(20000, self.sample_rate / 2)  # Hz - límite superior (Nyquist)
        
        # Crear límites logarítmicos
        band_edges = np.logspace(np. log10(f_min), np.log10(f_max), 
                                self.n_bands + 1)
        
        # Crear tuplas de bandas
        bands = [(band_edges[i], band_edges[i+1]) 
                for i in range(self.n_bands)]
        
        return bands
    
    def get_band_names(self) -> List[str]:
        """
        Retorna nombres descriptivos para cada banda.
        
        Returns:
            Lista de nombres de bandas
        """
        names = []
        for i, (f_low, f_high) in enumerate(self.bands):
            if f_low < 250: 
                name = "Sub-Bass/Bass"
            elif f_low < 500:
                name = "Bass"
            elif f_low < 2000:
                name = "Mid-Low"
            elif f_low < 4000:
                name = "Mid-High"
            elif f_low < 6000:
                name = "Presence"
            else:
                name = "Brilliance"
            
            names.append(f"Banda {i+1}: {name}\n({f_low:.0f}-{f_high:.0f} Hz)")
        
        return names
    
    def set_gain(self, band_index: int, gain_db: float):
        """
        Establece la ganancia para una banda específica.
        
        CONCEPTO MATEMÁTICO: 
        La ganancia en dB se convierte a escala lineal:
        
        ganancia_lineal = 10^(ganancia_dB / 20)
        
        Args:
            band_index:  Índice de la banda (0 a n_bands-1)
            gain_db:  Ganancia en decibelios (-20 a +20 típicamente)
        """
        if 0 <= band_index < self.n_bands:
            self.gains[band_index] = 10 ** (gain_db / 20)
    
    def set_all_gains(self, gains_db: List[float]):
        """
        Establece ganancias para todas las bandas. 
        
        Args:
            gains_db: Lista de ganancias en dB para cada banda
        """
        for i, gain_db in enumerate(gains_db):
            self.set_gain(i, gain_db)
    
    def create_filter_mask(self, fft_length: int) -> np.ndarray:
        """
        Crea una máscara de filtro en frecuencia para la ecualización.
        
        CONCEPTO MATEMÁTICO - FILTRADO EN FRECUENCIA:
        El filtrado en frecuencia se realiza mediante multiplicación: 
        
        Y(f) = X(f) * H(f)
        
        donde H(f) es la función de transferencia del filtro.
        
        En dominio del tiempo esto equivale a convolución:
        y(t) = x(t) * h(t)
        
        PROPIEDAD DE CONVOLUCIÓN: 
        F{x * h} = F{x} · F{h}
        
        Args:
            fft_length: Longitud de la FFT
            
        Returns: 
            Máscara de filtro (array complejo)
        """
        # Inicializar máscara
        mask = np. ones(fft_length, dtype=complex)
        
        # Calcular frecuencias
        frequencies = np.fft. fftfreq(fft_length, d=1/self.sample_rate)
        
        # Aplicar ganancia a cada banda
        for i, (f_low, f_high) in enumerate(self.bands):
            # Crear máscara para esta banda
            band_mask = (np.abs(frequencies) >= f_low) & (np.abs(frequencies) < f_high)
            
            # Aplicar ganancia
            mask[band_mask] *= self.gains[i]
        
        return mask
    
    def equalize(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aplica ecualización a la señal.
        
        PROCESO:
        1. FFT:  señal tiempo → frecuencia
        2. Filtrado: multiplicación por máscara
        3. IFFT: señal frecuencia → tiempo
        
        Args:
            signal: Señal de audio en dominio del tiempo
            
        Returns:
            Tuple (señal_ecualizada, fft_original, fft_ecualizada)
        """
        # 1. Transformar a dominio de la frecuencia
        fft_original = np.fft. fft(signal)
        
        # 2. Crear y aplicar máscara de filtro
        filter_mask = self.create_filter_mask(len(signal))
        fft_equalized = fft_original * filter_mask
        
        # 3. Transformar de vuelta a dominio del tiempo
        signal_equalized = np.fft.ifft(fft_equalized).real
        
        # Normalizar para evitar clipping
        max_val = np.max(np.abs(signal_equalized))
        if max_val > 1.0:
            signal_equalized /= max_val
        
        return signal_equalized, fft_original, fft_equalized
    
    def get_band_energies(self, fft_signal: np.ndarray) -> np.ndarray:
        """
        Calcula la energía en cada banda de frecuencia.
        
        CONCEPTO MATEMÁTICO: 
        La energía en una banda es:
        
        E_banda = Σ |X(f)|² para f en [f_low, f_high]
        
        Args: 
            fft_signal: Señal en dominio de la frecuencia
            
        Returns:
            Array con energías por banda
        """
        frequencies = np.fft.fftfreq(len(fft_signal), d=1/self.sample_rate)
        energies = np.zeros(self.n_bands)
        
        for i, (f_low, f_high) in enumerate(self.bands):
            band_mask = (np.abs(frequencies) >= f_low) & (np.abs(frequencies) < f_high)
            energies[i] = np.sum(np.abs(fft_signal[band_mask])**2)
        
        return energies
    
    def analyze_frequency_response(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula la respuesta en frecuencia del ecualizador.
        
        Returns:
            Tuple (frecuencias, respuesta_magnitud_dB)
        """
        # Crear impulso
        impulse = np. zeros(self.sample_rate)
        impulse[0] = 1
        
        # Aplicar ecualización
        response, _, _ = self.equalize(impulse)
        
        # FFT de la respuesta
        fft_response = np.fft.fft(response)
        frequencies = np.fft.fftfreq(len(response), d=1/self.sample_rate)
        
        # Convertir a dB
        magnitude_db = 20 * np.log10(np.abs(fft_response) + 1e-10)
        
        # Solo frecuencias positivas
        positive_idx = frequencies >= 0
        
        return frequencies[positive_idx], magnitude_db[positive_idx]