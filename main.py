"""
Aplicación Principal - Ecualizador de Audio Multibanda con Espectrograma
Proyecto de Matemáticas Avanzadas:  Series y Transformadas de Fourier
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

from fourier_analysis import FourierAnalyzer
from equalizer import MultibandEqualizer
from spectrogram import SpectrogramGenerator
from audio_processor import AudioProcessor
from visualizer import AudioVisualizer

class AudioEqualizerApp:
    """
    Aplicación principal del ecualizador de audio multibanda.
    """
    
    def __init__(self):
        """Inicializa la aplicación."""
        self.audio_processor = AudioProcessor()
        
        # Crear carpeta de resultados si no existe
        self.results_dir = Path("resultados")
        self.results_dir.mkdir(exist_ok=True)
        
        self.visualizer = AudioVisualizer(output_dir=str(self.results_dir))
        
        # Parámetros configurables
        self.n_bands = 5
        self.window_size = 2048
        self.hop_length = 512
        
    def print_header(self):
        """Imprime el encabezado de la aplicación."""
        print("=" * 70)
        print("🎵  ECUALIZADOR DE AUDIO MULTIBANDA + ESPECTROGRAMA  🎵")
        print("=" * 70)
        print("Conceptos de Matemáticas Avanzadas:")
        print("  • Transformada de Fourier (DFT/FFT)")
        print("  • Teorema de Parseval")
        print("  • Análisis Tiempo-Frecuencia (STFT)")
        print("  • Filtrado y Convolución")
        print("  • Series de Fourier y Reconstrucción")
        print("=" * 70)
        print()
        
    def load_and_analyze_audio(self, filepath: str) -> dict:
        """
        Carga y realiza análisis inicial del audio.
        
        Args:
            filepath: Ruta al archivo de audio
            
        Returns:
            Diccionario con información del audio
        """
        print(f"📂 Cargando audio:  {filepath}")
        
        # Cargar audio
        signal, sample_rate = self.audio_processor.load_audio(filepath)
        
        print(f"   ✓ Duración: {len(signal)/sample_rate:.2f} segundos")
        print(f"   ✓ Sample rate: {sample_rate} Hz")
        print(f"   ✓ Número de muestras: {len(signal)}")
        print(f"   ✓ Nivel RMS: {self.audio_processor.compute_rms(signal):.4f}")
        print(f"   ✓ Nivel de pico: {self.audio_processor.compute_peak_level(signal):.2f} dB")
        print()
        
        return {
            'signal': signal,
            'sample_rate': sample_rate,
            'filepath': filepath
        }
    
    def apply_equalization(self, audio_data: dict, gains_db: list) -> dict:
        """
        Aplica ecualización al audio.
        
        Args:
            audio_data: Diccionario con datos del audio
            gains_db: Lista de ganancias en dB para cada banda
            
        Returns: 
            Diccionario con audio ecualizado y análisis
        """
        signal = audio_data['signal']
        sample_rate = audio_data['sample_rate']
        
        print(f"🎚️  Aplicando ecualización con {self.n_bands} bandas...")
        
        # Crear ecualizador
        equalizer = MultibandEqualizer(sample_rate, self.n_bands)
        
        # Configurar ganancias
        equalizer.set_all_gains(gains_db)
        
        # Mostrar configuración
        band_names = equalizer.get_band_names()
        print("\n   Configuración de bandas:")
        for i, (name, gain) in enumerate(zip(band_names, gains_db)):
            print(f"   {name}: {gain:+.1f} dB")
        print()
        
        # Aplicar ecualización
        signal_eq, fft_orig, fft_eq = equalizer.equalize(signal)
        
        print("   ✓ Ecualización aplicada")
        print(f"   ✓ Nivel RMS (original): {self.audio_processor.compute_rms(signal):.4f}")
        print(f"   ✓ Nivel RMS (ecualizado): {self.audio_processor.compute_rms(signal_eq):.4f}")
        print()
        
        return {
            'signal_original': signal,
            'signal_equalized': signal_eq,
            'fft_original': fft_orig,
            'fft_equalized': fft_eq,
            'equalizer': equalizer,
            'sample_rate': sample_rate
        }
    
    def perform_fourier_analysis(self, eq_data: dict):
        """
        Realiza análisis de Fourier completo.
        
        Args:
            eq_data: Diccionario con datos de ecualización
        """
        print("📊 Análisis de Fourier...")
        
        signal_orig = eq_data['signal_original']
        signal_eq = eq_data['signal_equalized']
        sample_rate = eq_data['sample_rate']
        
        # Crear analizador
        analyzer = FourierAnalyzer(sample_rate)
        
        # Verificar Teorema de Parseval para señal original
        print("\n   🔬 Verificación del Teorema de Parseval (señal original):")
        is_valid, time_energy, freq_energy = analyzer.verify_parseval(
            signal_orig, eq_data['fft_original']
        )
        print(f"   Energía (tiempo): {time_energy:.6e}")
        print(f"   Energía (frecuencia): {freq_energy:.6e}")
        print(f"   Error relativo: {abs(time_energy - freq_energy)/time_energy * 100:.8f}%")
        print(f"   ✓ Parseval {'VERIFICADO' if is_valid else 'NO verificado'}")
        
        # Visualizar Parseval
        self.visualizer.plot_parseval_verification(time_energy, freq_energy, "Original")
        
        # Verificar Parseval para señal ecualizada
        print("\n   🔬 Verificación del Teorema de Parseval (señal ecualizada):")
        is_valid_eq, time_energy_eq, freq_energy_eq = analyzer.verify_parseval(
            signal_eq, eq_data['fft_equalized']
        )
        print(f"   Energía (tiempo): {time_energy_eq:.6e}")
        print(f"   Energía (frecuencia): {freq_energy_eq:.6e}")
        print(f"   Error relativo: {abs(time_energy_eq - freq_energy_eq)/time_energy_eq * 100:.8f}%")
        print(f"   ✓ Parseval {'VERIFICADO' if is_valid_eq else 'NO verificado'}")
        print()
        
    def generate_spectrograms(self, eq_data: dict):
        """
        Genera espectrogramas para análisis tiempo-frecuencia.
        
        Args:
            eq_data: Diccionario con datos de ecualización
        """
        print("📈 Generando espectrogramas (STFT)...")
        
        signal_orig = eq_data['signal_original']
        signal_eq = eq_data['signal_equalized']
        sample_rate = eq_data['sample_rate']
        
        # Crear generador de espectrogramas
        spec_gen = SpectrogramGenerator(sample_rate)
        
        # Generar espectrograma original
        spec_orig, times, freqs = spec_gen.compute_spectrogram(
            signal_orig, self.window_size, self.hop_length
        )
        
        # Generar espectrograma ecualizado
        spec_eq, _, _ = spec_gen.compute_spectrogram(
            signal_eq, self.window_size, self.hop_length
        )
        
        print(f"   ✓ Resolución temporal: {times[1] - times[0]:.4f} s")
        print(f"   ✓ Resolución frecuencial: {freqs[1] - freqs[0]:.2f} Hz")
        print(f"   ✓ Número de frames: {len(times)}")
        print()
        
        # Visualizar espectrogramas
        self.visualizer.plot_spectrograms_comparison(spec_orig, spec_eq, times, freqs)
        
    def analyze_band_energies(self, eq_data: dict):
        """
        Analiza energías por banda de frecuencia.
        
        Args:
            eq_data:  Diccionario con datos de ecualización
        """
        print("🔍 Análisis de energía por banda...")
        
        equalizer = eq_data['equalizer']
        fft_orig = eq_data['fft_original']
        fft_eq = eq_data['fft_equalized']
        
        # Calcular energías
        energies_orig = equalizer.get_band_energies(fft_orig)
        energies_eq = equalizer.get_band_energies(fft_eq)
        
        # Mostrar resultados
        band_names = equalizer.get_band_names()
        print("\n   Energías por banda:")
        for name, e_orig, e_eq in zip(band_names, energies_orig, energies_eq):
            change = (e_eq / e_orig - 1) * 100 if e_orig > 0 else 0
            print(f"   {name}")
            print(f"      Original: {e_orig:.2e} | Ecualizado: {e_eq:.2e} | Cambio: {change:+.1f}%")
        print()
        
        # Visualizar energías
        self.visualizer.plot_band_energies(energies_orig, energies_eq, band_names)
        
    def create_visualizations(self, eq_data: dict):
        """
        Crea todas las visualizaciones. 
        
        Args:
            eq_data: Diccionario con datos de ecualización
        """
        print("🎨 Generando visualizaciones...")
        
        # Comparación tiempo-frecuencia
        self.visualizer.plot_comparison(
            eq_data['signal_original'],
            eq_data['signal_equalized'],
            eq_data['sample_rate'],
            eq_data['fft_original'],
            eq_data['fft_equalized']
        )
        
        # Curva de respuesta del ecualizador
        equalizer = eq_data['equalizer']
        freqs, response = equalizer.analyze_frequency_response()
        self.visualizer.plot_equalizer_curve(
            freqs, response, equalizer.bands, equalizer.gains
        )
        
        print()
        
    def save_equalized_audio(self, eq_data: dict, output_path: str):
        """
        Guarda el audio ecualizado. 
        
        Args:
            eq_data: Diccionario con datos de ecualización
            output_path: Ruta de salida
        """
        print(f"💾 Guardando audio ecualizado:  {output_path}")
        
        self.audio_processor.save_audio(
            output_path,
            eq_data['signal_equalized'],
            eq_data['sample_rate']
        )
        
        print("   ✓ Audio guardado exitosamente")
        print()
        
    def run_demo(self):
        """
        Ejecuta una demostración con audio de prueba.
        """
        self.print_header()
        
        print("🎬 MODO DEMOSTRACIÓN")
        print("Generando señales de prueba...\n")
        
        # Generar señal de prueba (mezcla de frecuencias)
        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Crear señal compleja con múltiples componentes frecuenciales
        # Esto demuestra la descomposición en Series de Fourier
        signal = (
            0.5 * np.sin(2 * np.pi * 100 * t) +    # Bajo (100 Hz)
            0.3 * np.sin(2 * np.pi * 500 * t) +    # Mid-bajo (500 Hz)
            0.4 * np.sin(2 * np.pi * 1000 * t) +   # Mid (1 kHz)
            0.3 * np.sin(2 * np.pi * 3000 * t) +   # Mid-alto (3 kHz)
            0.2 * np.sin(2 * np.pi * 8000 * t)     # Agudo (8 kHz)
        )
        
        # Normalizar
        signal = self.audio_processor.normalize_audio(signal)
        
        # Guardar señal de prueba
        test_file = self.results_dir / "test_signal.wav"
        self.audio_processor.save_audio(str(test_file), signal, sample_rate)
        
        print(f"✓ Señal de prueba generada: {test_file}")
        print("  Componentes frecuenciales: 100Hz, 500Hz, 1kHz, 3kHz, 8kHz\n")
        
        # Cargar y analizar
        audio_data = self.load_and_analyze_audio(str(test_file))
        
        # Configurar ecualización de ejemplo
        # Formato: [banda1, banda2, banda3, banda4, banda5] en dB
        gains_db = [-6, +3, 0, +6, -3]  # Curva en V modificada
        
        print("📝 Ecualización de ejemplo:")
        print("   • Atenuar graves (-6 dB)")
        print("   • Realzar mid-bajos (+3 dB)")
        print("   • Mantener medios (0 dB)")
        print("   • Realzar presencia (+6 dB)")
        print("   • Atenuar brillantez (-3 dB)\n")
        
        # Aplicar ecualización
        eq_data = self.apply_equalization(audio_data, gains_db)
        
        # Análisis de Fourier
        self.perform_fourier_analysis(eq_data)
        
        # Generar espectrogramas
        self.generate_spectrograms(eq_data)
        
        # Analizar energías por banda
        self.analyze_band_energies(eq_data)
        
        # Crear visualizaciones
        self.create_visualizations(eq_data)
        
        # Guardar resultado
        output_file = self.results_dir / "test_signal_equalized.wav"
        self.save_equalized_audio(eq_data, str(output_file))
        
        print("=" * 70)
        print("✅ DEMOSTRACIÓN COMPLETADA")
        print("=" * 70)
        print("\nArchivos generados en carpeta 'resultados':")
        print("  📄 test_signal.wav - Señal original")
        print("  📄 test_signal_equalized.wav - Señal ecualizada")
        print("  📊 comparison.png - Comparación tiempo-frecuencia")
        print("  📊 spectrograms_comparison.png - Espectrogramas")
        print("  📊 band_energies.png - Energías por banda")
        print("  📊 equalizer_response.png - Respuesta del ecualizador")
        print("  📊 parseval_verification.png - Verificación de Parseval")
        print()
        
    def run_interactive(self):
        """
        Ejecuta modo interactivo con archivo proporcionado por el usuario.
        """
        self.print_header()
        
        print("🎯 MODO INTERACTIVO\n")
        
        # Solicitar archivo
        filepath = input("Ingrese la ruta del archivo de audio (. wav): ").strip()
        
        if not Path(filepath).exists():
            print(f"❌ Error: El archivo '{filepath}' no existe.")
            return
        
        print()
        
        # Cargar y analizar
        audio_data = self.load_and_analyze_audio(filepath)
        
        # Solicitar ganancias
        print(f"Ingrese las ganancias para {self.n_bands} bandas (en dB, separadas por espacios)")
        print("Ejemplo: -3 0 +6 +3 -6")
        print(f"Rango recomendado: -12 dB a +12 dB\n")
        
        gains_input = input("Ganancias: ").strip()
        
        try:
            gains_db = [float(g) for g in gains_input.split()]
            if len(gains_db) != self.n_bands:
                print(f"❌ Error: Se requieren exactamente {self.n_bands} valores.")
                return
        except ValueError:
            print("❌ Error: Valores inválidos. Use números separados por espacios.")
            return
        
        print()
        
        # Aplicar ecualización
        eq_data = self.apply_equalization(audio_data, gains_db)
        
        # Análisis completo
        self.perform_fourier_analysis(eq_data)
        self.generate_spectrograms(eq_data)
        self.analyze_band_energies(eq_data)
        self.create_visualizations(eq_data)
        
        # Guardar resultado
        output_filename = str(Path(filepath).stem) + "_equalized.wav"
        output_path = self.results_dir / output_filename
        self.save_equalized_audio(eq_data, str(output_path))
        
        print("=" * 70)
        print("✅ PROCESAMIENTO COMPLETADO")
        print("=" * 70)
        print(f"\n🎵 Audio ecualizado guardado en: {output_path}\n")


def main():
    """Función principal."""
    app = AudioEqualizerApp()
    
    if len(sys.argv) > 1:
        # Modo interactivo si se proporciona argumento
        if sys.argv[1] == '--interactive' or sys.argv[1] == '-i':
            app.run_interactive()
        else:
            print("Uso:")
            print("  python main.py           # Ejecuta demostración")
            print("  python main.py -i        # Modo interactivo")
    else:
        # Modo demostración por defecto
        app.run_demo()


if __name__ == "__main__":
    main()