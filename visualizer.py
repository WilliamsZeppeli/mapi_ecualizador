"""
Módulo de Visualización
Genera gráficas para análisis de señales de audio en tiempo y frecuencia
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, List

class AudioVisualizer:
    """
    Clase para generar visualizaciones de señales de audio. 
    Incluye gráficas en dominio del tiempo, frecuencia y espectrogramas.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10), output_dir: str = "."):
        """
        Inicializa el visualizador. 
        
        Args:
            figsize: Tamaño de las figuras (ancho, alto)
            output_dir: Directorio donde guardar las visualizaciones
        """
        self.figsize = figsize
        self.output_dir = output_dir
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_waveform(self, signal: np.ndarray, sample_rate: int, 
                     title: str = "Forma de Onda", ax: Optional[plt.Axes] = None):
        """
        Grafica la forma de onda en dominio del tiempo.
        
        Args:
            signal: Señal de audio
            sample_rate: Frecuencia de muestreo
            title: Título de la gráfica
            ax: Eje de matplotlib (opcional)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        
        # Crear eje de tiempo
        time = np.arange(len(signal)) / sample_rate
        
        # Graficar
        ax.plot(time, signal, linewidth=0.5, color='blue', alpha=0.7)
        ax.set_xlabel('Tiempo (s)', fontsize=12)
        ax.set_ylabel('Amplitud', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time[-1])
        
    def plot_spectrum(self, frequencies: np.ndarray, magnitudes: np.ndarray,
                     title: str = "Espectro de Frecuencia", 
                     ax: Optional[plt.Axes] = None,
                     db_scale: bool = True):
        """
        Grafica el espectro de frecuencia.
        
        Args:
            frequencies:  Array de frecuencias
            magnitudes: Array de magnitudes
            title: Título de la gráfica
            ax: Eje de matplotlib (opcional)
            db_scale: Si True, usa escala de dB
        """
        if ax is None: 
            fig, ax = plt.subplots(figsize=(12, 4))
        
        # Convertir a dB si es necesario
        if db_scale: 
            magnitudes_plot = 20 * np.log10(magnitudes + 1e-10)
            ylabel = 'Magnitud (dB)'
        else:
            magnitudes_plot = magnitudes
            ylabel = 'Magnitud'
        
        # Graficar
        ax. plot(frequencies, magnitudes_plot, linewidth=1, color='red', alpha=0.8)
        ax.set_xlabel('Frecuencia (Hz)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xlim(20, frequencies[-1])
        
    def plot_spectrogram(self, spectrogram: np.ndarray, times: np.ndarray,
                        frequencies: np.ndarray, title: str = "Espectrograma",
                        ax:  Optional[plt.Axes] = None, db_scale: bool = True):
        """
        Grafica un espectrograma.
        
        Args:
            spectrogram: Matriz de espectrograma
            times: Array de tiempos
            frequencies:  Array de frecuencias
            title: Título de la gráfica
            ax: Eje de matplotlib (opcional)
            db_scale: Si True, usa escala de dB
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convertir a dB si es necesario
        if db_scale: 
            spec_plot = 10 * np.log10(spectrogram + 1e-10)
            vmin, vmax = np.percentile(spec_plot, [5, 95])
        else:
            spec_plot = spectrogram
            vmin, vmax = None, None
        
        # Graficar espectrograma
        im = ax.imshow(spec_plot, aspect='auto', origin='lower',
                      extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
                      cmap='viridis', vmin=vmin, vmax=vmax)
        
        ax.set_xlabel('Tiempo (s)', fontsize=12)
        ax.set_ylabel('Frecuencia (Hz)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.set_ylim(20, frequencies[-1])
        
        # Agregar barra de color
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Potencia (dB)' if db_scale else 'Potencia', fontsize=10)
        
    def plot_comparison(self, signal_original: np.ndarray, 
                       signal_processed: np.ndarray,
                       sample_rate: int, 
                       fft_original: np.ndarray,
                       fft_processed:  np.ndarray):
        """
        Crea una comparación completa entre señal original y procesada.
        
        Args:
            signal_original: Señal original
            signal_processed: Señal procesada
            sample_rate: Frecuencia de muestreo
            fft_original: FFT de la señal original
            fft_processed: FFT de la señal procesada
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Calcular frecuencias
        n_samples = len(signal_original)
        frequencies = np.fft.fftfreq(n_samples, d=1/sample_rate)
        positive_freq_idx = frequencies >= 0
        freqs = frequencies[positive_freq_idx]
        
        # Magnitudes
        mag_original = np.abs(fft_original[positive_freq_idx])
        mag_processed = np.abs(fft_processed[positive_freq_idx])
        
        # Forma de onda original
        self.plot_waveform(signal_original, sample_rate, 
                          "Señal Original - Dominio del Tiempo", axes[0, 0])
        
        # Forma de onda procesada
        self.plot_waveform(signal_processed, sample_rate,
                          "Señal Ecualizada - Dominio del Tiempo", axes[0, 1])
        
        # Espectro original
        self.plot_spectrum(freqs, mag_original,
                          "Espectro Original - Dominio de la Frecuencia", 
                          axes[1, 0])
        
        # Espectro procesado
        self.plot_spectrum(freqs, mag_processed,
                          "Espectro Ecualizado - Dominio de la Frecuencia",
                          axes[1, 1])
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparación guardada en '{output_path}'")
        
    def plot_band_energies(self, energies_before: np.ndarray,
                          energies_after:  np.ndarray,
                          band_names: List[str]):
        """
        Grafica las energías por banda antes y después de ecualizar.
        
        Args:
            energies_before:  Energías originales
            energies_after:  Energías después de ecualización
            band_names:  Nombres de las bandas
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(band_names))
        width = 0.35
        
        # Normalizar energías para mejor visualización
        energies_before_norm = energies_before / np.max(energies_before)
        energies_after_norm = energies_after / np.max(energies_after)
        
        # Graficar barras
        bars1 = ax.bar(x - width/2, energies_before_norm, width, 
                      label='Original', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, energies_after_norm, width,
                      label='Ecualizado', color='coral', alpha=0.8)
        
        ax.set_xlabel('Bandas de Frecuencia', fontsize=12)
        ax.set_ylabel('Energía Normalizada', fontsize=12)
        ax.set_title('Distribución de Energía por Banda', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(band_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/band_energies.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Energías por banda guardadas en '{output_path}'")
        
    def plot_equalizer_curve(self, frequencies: np.ndarray, 
                            response_db: np.ndarray,
                            bands: List[Tuple[float, float]],
                            gains: np.ndarray):
        """
        Grafica la curva de respuesta del ecualizador.
        
        Args:
            frequencies: Array de frecuencias
            response_db:  Respuesta en dB
            bands: Lista de bandas (freq_min, freq_max)
            gains: Ganancias aplicadas a cada banda
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graficar respuesta
        ax.plot(frequencies, response_db, linewidth=2, color='darkgreen', 
               label='Respuesta del Ecualizador')
        
        # Marcar bandas
        colors = plt.cm.rainbow(np.linspace(0, 1, len(bands)))
        for i, ((f_low, f_high), gain) in enumerate(zip(bands, gains)):
            gain_db = 20 * np.log10(gain)
            ax.axvspan(f_low, f_high, alpha=0.2, color=colors[i], 
                      label=f'Banda {i+1}: {gain_db:+.1f} dB')
        
        ax.set_xlabel('Frecuencia (Hz)', fontsize=12)
        ax.set_ylabel('Ganancia (dB)', fontsize=12)
        ax.set_title('Curva de Respuesta del Ecualizador', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        ax.set_xlim(20, frequencies[-1])
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/equalizer_response.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Respuesta del ecualizador guardada en '{output_path}'")
        
    def plot_spectrograms_comparison(self, spec_original: np.ndarray,
                                    spec_processed: np. ndarray,
                                    times: np.ndarray,
                                    frequencies: np.ndarray):
        """
        Compara espectrogramas original y procesado lado a lado.
        
        Args:
            spec_original:  Espectrograma original
            spec_processed: Espectrograma procesado
            times: Array de tiempos
            frequencies: Array de frecuencias
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Espectrograma original
        self.plot_spectrogram(spec_original, times, frequencies,
                             "Espectrograma Original", axes[0])
        
        # Espectrograma procesado
        self.plot_spectrogram(spec_processed, times, frequencies,
                             "Espectrograma Ecualizado", axes[1])
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/spectrograms_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparación de espectrogramas guardada en '{output_path}'")
        
    def plot_parseval_verification(self, time_energy: float, 
                                   freq_energy: float,
                                   signal_name: str = "Señal"):
        """
        Visualiza la verificación del Teorema de Parseval.
        
        Args:
            time_energy: Energía en dominio del tiempo
            freq_energy: Energía en dominio de la frecuencia
            signal_name: Nombre de la señal
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        energies = [time_energy, freq_energy]
        labels = ['Dominio del Tiempo\n∑|x[n]|²', 
                 'Dominio de la Frecuencia\n(1/N)∑|X[k]|²']
        colors = ['dodgerblue', 'orangered']
        
        bars = ax.bar(labels, energies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Añadir valores sobre las barras
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{energy:.2e}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Calcular error relativo
        error = abs(time_energy - freq_energy) / time_energy * 100
        
        ax.set_ylabel('Energía Total', fontsize=12)
        ax.set_title(f'Verificación del Teorema de Parseval - {signal_name}\n' + 
                    f'Error Relativo: {error:.6f}%',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir anotación del teorema
        ax.text(0.5, 0.95, 'Teorema de Parseval:  La energía se conserva en ambos dominios',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/parseval_verification.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Verificación de Parseval guardada en '{output_path}'")