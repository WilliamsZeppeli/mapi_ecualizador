"""
Script de Pruebas Avanzadas
Demuestra conceptos matemáticos adicionales
"""

import numpy as np
import matplotlib. pyplot as plt
from fourier_analysis import FourierAnalyzer
from equalizer import MultibandEqualizer
from audio_processor import AudioProcessor

def test_fourier_properties():
    """
    Prueba las propiedades fundamentales de la Transformada de Fourier. 
    """
    print("=" * 60)
    print("PRUEBAS DE PROPIEDADES DE LA TRANSFORMADA DE FOURIER")
    print("=" * 60)
    
    sample_rate = 44100
    analyzer = FourierAnalyzer(sample_rate)
    
    # Generar señales de prueba
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    x1 = np.sin(2 * np.pi * 440 * t)  # La (440 Hz)
    x2 = np.sin(2 * np.pi * 880 * t)  # La (880 Hz, octava superior)
    
    print("\n1️⃣  TEST:   LINEALIDAD")
    print("   Propiedad:  F{a·x₁ + b·x₂} = a·F{x₁} + b·F{x₂}")
    
    a, b = 2.0, 3.0
    combined = a * x1 + b * x2
    
    fft_combined = np.fft.fft(combined)
    fft_separate = a * np.fft.fft(x1) + b * np.fft.fft(x2)
    
    max_error = np.max(np.abs(fft_combined - fft_separate))
    print(f"   Error máximo: {max_error:.2e}")
    print(f"   ✓ Linealidad {'VERIFICADA' if max_error < 1e-10 else 'FALLÓ'}")
    
    print("\n2️⃣  TEST:  DESPLAZAMIENTO EN FRECUENCIA")
    print("   Propiedad: x(t)·e^(j2πf₀t) ↔ X(f - f₀)")
    
    shift_hz = 100
    shifted = analyzer.apply_frequency_shift(x1, shift_hz)
    
    fft_orig = np.fft.fft(x1)
    fft_shifted = np.fft. fft(shifted)
    
    freqs = np.fft.fftfreq(len(x1), 1/sample_rate)
    
    # Encontrar picos
    orig_peak_idx = np.argmax(np.abs(fft_orig[: len(fft_orig)//2]))
    shifted_peak_idx = np.argmax(np.abs(fft_shifted[: len(fft_shifted)//2]))
    
    orig_peak_freq = freqs[orig_peak_idx]
    shifted_peak_freq = freqs[shifted_peak_idx]
    
    print(f"   Frecuencia original: {orig_peak_freq:.1f} Hz")
    print(f"   Frecuencia desplazada: {shifted_peak_freq:. 1f} Hz")
    print(f"   Desplazamiento esperado: {shift_hz} Hz")
    print(f"   Desplazamiento real: {shifted_peak_freq - orig_peak_freq:.1f} Hz")
    
    print("\n3️⃣  TEST: PARSEVAL (Conservación de Energía)")
    print("   Teorema:  Σ|x[n]|² = (1/N)·Σ|X[k]|²")
    
    is_valid, time_energy, freq_energy = analyzer.verify_parseval(x1, fft_orig)
    error_percent = abs(time_energy - freq_energy) / time_energy * 100
    
    print(f"   Energía (dominio del tiempo): {time_energy:.6f}")
    print(f"   Energía (dominio de frecuencia): {freq_energy:.6f}")
    print(f"   Error relativo: {error_percent:. 8f}%")
    print(f"   ✓ Parseval {'VERIFICADO' if is_valid else 'FALLÓ'}")
    
    print("\n4️⃣  TEST:  RECONSTRUCCIÓN (FFT + IFFT)")
    print("   Propiedad:  IFFT(FFT(x)) = x")
    
    reconstructed = analyzer.compute_ifft(fft_orig)
    reconstruction_error = np.max(np.abs(x1 - reconstructed))
    
    print(f"   Error máximo de reconstrucción: {reconstruction_error:.2e}")
    print(f"   ✓ Reconstrucción {'PERFECTA' if reconstruction_error < 1e-10 else 'CON ERROR'}")
    
    print("\n" + "=" * 60)
    print("TODAS LAS PRUEBAS COMPLETADAS")
    print("=" * 60 + "\n")


def test_series_de_fourier():
    """
    Demuestra la reconstrucción de señales mediante Series de Fourier.
    """
    print("=" * 60)
    print("DEMOSTRACIÓN:  SERIES DE FOURIER")
    print("=" * 60)
    
    # Crear señal cuadrada
    sample_rate = 44100
    duration = 0.1
    freq = 100  # Hz
    t = np. linspace(0, duration, int(sample_rate * duration))
    
    # Señal cuadrada ideal
    square_wave = np.sign(np.sin(2 * np.pi * freq * t))
    
    print(f"\nReconstruyendo onda cuadrada de {freq} Hz usando Series de Fourier")
    print("Fórmula: x(t) = (4/π) · Σ[sin((2k-1)ωt) / (2k-1)] para k=1,2,3,...")
    
    # Reconstruir con diferentes números de armónicos
    harmonics_list = [1, 3, 5, 10, 20]
    
    fig, axes = plt.subplots(len(harmonics_list), 1, figsize=(12, 12))
    
    for idx, n_harmonics in enumerate(harmonics_list):
        reconstructed = np.zeros_like(t)
        
        # Sumar armónicos impares
        for k in range(1, n_harmonics + 1):
            harmonic = (2*k - 1)
            reconstructed += (4 / np.pi) * np.sin(2 * np.pi * freq * harmonic * t) / harmonic
        
        # Calcular error
        error = np. mean((square_wave - reconstructed)**2)
        
        # Graficar
        axes[idx]. plot(t[: 500], square_wave[:500], 'k--', label='Original', linewidth=2, alpha=0.5)
        axes[idx].plot(t[:500], reconstructed[: 500], 'r-', label=f'{n_harmonics} armónicos', linewidth=1.5)
        axes[idx].set_ylabel('Amplitud')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_title(f'Reconstrucción con {n_harmonics} armónicos (MSE:  {error:.4f})')
        
        print(f"   {n_harmonics} armónicos: MSE = {error:. 6f}")
    
    axes[-1].set_xlabel('Tiempo (s)')
    plt.tight_layout()
    plt.savefig('fourier_series_reconstruction.png', dpi=300)
    print("\n✓ Gráfica guardada:  fourier_series_reconstruction.png")
    
    print("\n📊 OBSERVACIÓN:")
    print("   A mayor número de armónicos, mejor aproximación (fenómeno de Gibbs en discontinuidades)")
    print("=" * 60 + "\n")


def test_convolucion():
    """
    Demuestra la propiedad de convolución en procesamiento de audio.
    """
    print("=" * 60)
    print("DEMOSTRACIÓN:  CONVOLUCIÓN Y FILTRADO")
    print("=" * 60)
    
    sample_rate = 44100
    
    # Crear señal con ruido
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Señal limpia
    signal_clean = np.sin(2 * np.pi * 440 * t)
    
    # Agregar ruido de alta frecuencia
    noise = 0.3 * np.sin(2 * np. pi * 5000 * t)
    signal_noisy = signal_clean + noise
    
    print("\n1️⃣  Señal original:  440 Hz (La)")
    print("   Ruido agregado: 5000 Hz")
    
    # Crear filtro paso-bajo simple (promedio móvil)
    window_size = 50
    filter_kernel = np.ones(window_size) / window_size
    
    print(f"\n2️⃣  Aplicando filtro paso-bajo (ventana de {window_size} muestras)")
    
    # Método 1: Convolución en dominio del tiempo
    print("   Método A: Convolución directa en tiempo")
    signal_filtered_time = np. convolve(signal_noisy, filter_kernel, mode='same')
    
    # Método 2: Multiplicación en dominio de la frecuencia
    print("   Método B:  Multiplicación en frecuencia (FFT → × → IFFT)")
    
    # Zero-pad para evitar convolución circular
    n = len(signal_noisy) + len(filter_kernel) - 1
    signal_fft = np.fft. fft(signal_noisy, n)
    filter_fft = np.fft.fft(filter_kernel, n)
    
    filtered_fft = signal_fft * filter_fft
    signal_filtered_freq = np.fft.ifft(filtered_fft).real[:len(signal_noisy)]
    
    # Comparar resultados
    difference = np.max(np.abs(signal_filtered_time - signal_filtered_freq))
    
    print(f"\n3️⃣  VERIFICACIÓN DE LA PROPIEDAD DE CONVOLUCIÓN")
    print("   Propiedad:  y = x * h ↔ Y = X · H")
    print(f"   Diferencia máxima entre métodos:  {difference:.2e}")
    print(f"   ✓ Propiedad {'VERIFICADA' if difference < 1e-5 else 'CON ERROR'}")
    
    # Visualizar
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    plot_samples = 500
    
    axes[0].plot(t[:plot_samples], signal_clean[:plot_samples], label='Señal limpia')
    axes[0].set_title('Señal Original (440 Hz)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t[:plot_samples], signal_noisy[:plot_samples], label='Señal con ruido', color='orange')
    axes[1].set_title('Señal con Ruido (440 Hz + 5000 Hz)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t[:plot_samples], signal_filtered_time[:plot_samples], label='Señal filtrada', color='green')
    axes[2].set_title('Señal Filtrada (paso-bajo)')
    axes[2].set_xlabel('Tiempo (s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convolution_demo.png', dpi=300)
    print("\n✓ Gráfica guardada: convolution_demo.png")
    
    print("=" * 60 + "\n")


def test_windowing():
    """
    Demuestra el efecto del ventaneo en análisis espectral.
    """
    print("=" * 60)
    print("DEMOSTRACIÓN:  VENTANEO Y ANÁLISIS ESPECTRAL")
    print("=" * 60)
    
    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Señal de prueba:  dos tonos cercanos en frecuencia
    freq1, freq2 = 440, 460  # Hz
    signal = np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t)
    
    print(f"\n📊 Señal de prueba: dos tonos ({freq1} Hz y {freq2} Hz)")
    print("   Separación: 20 Hz")
    
    # Diferentes tipos de ventanas
    windows = {
        'Rectangular': np.ones(len(signal)),
        'Hann': np.hanning(len(signal)),
        'Hamming': np.hamming(len(signal)),
        'Blackman': np.blackman(len(signal))
    }
    
    fig, axes = plt.subplots(len(windows), 2, figsize=(14, 12))
    
    for idx, (window_name, window) in enumerate(windows. items()):
        # Aplicar ventana
        windowed_signal = signal * window
        
        # FFT
        fft_result = np.fft.fft(windowed_signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # Solo frecuencias positivas
        positive_mask = freqs >= 0
        freqs_plot = freqs[positive_mask]
        mag_plot = magnitude_db[positive_mask]
        
        # Graficar ventana
        axes[idx, 0].plot(t, window, linewidth=2, color='blue')
        axes[idx, 0].set_title(f'Ventana {window_name}')
        axes[idx, 0].set_ylabel('Amplitud')
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].set_ylim(-0.1, 1.1)
        
        if idx == len(windows) - 1:
            axes[idx, 0]. set_xlabel('Tiempo (s)')
        
        # Graficar espectro
        axes[idx, 1].plot(freqs_plot, mag_plot, linewidth=1, color='red')
        axes[idx, 1].set_title(f'Espectro con Ventana {window_name}')
        axes[idx, 1].set_ylabel('Magnitud (dB)')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_xlim(400, 500)
        axes[idx, 1].axvline(freq1, color='green', linestyle='--', alpha=0.5, label=f'{freq1} Hz')
        axes[idx, 1].axvline(freq2, color='purple', linestyle='--', alpha=0.5, label=f'{freq2} Hz')
        axes[idx, 1].legend(fontsize=8)
        
        if idx == len(windows) - 1:
            axes[idx, 1].set_xlabel('Frecuencia (Hz)')
        
        print(f"\n   {window_name}:")
        print(f"      - Lóbulo principal: {'Ancho' if window_name == 'Rectangular' else 'Estrecho'}")
        print(f"      - Lóbulos laterales: {'Altos' if window_name == 'Rectangular' else 'Bajos'}")
    
    plt.tight_layout()
    plt.savefig('windowing_comparison.png', dpi=300)
    print("\n✓ Gráfica guardada: windowing_comparison.png")
    
    print("\n📊 OBSERVACIONES:")
    print("   • Rectangular:  Mejor resolución frecuencial, pero más fuga espectral")
    print("   • Hann/Hamming: Balance entre resolución y fuga")
    print("   • Blackman: Menor fuga espectral, pero menor resolución")
    print("=" * 60 + "\n")


def test_nyquist_aliasing():
    """
    Demuestra el teorema de muestreo de Nyquist y aliasing.
    """
    print("=" * 60)
    print("DEMOSTRACIÓN:  TEOREMA DE NYQUIST Y ALIASING")
    print("=" * 60)
    
    # Crear señal continua
    duration = 0.05
    t_continuous = np.linspace(0, duration, 100000)
    freq = 440  # Hz
    signal_continuous = np.sin(2 * np.pi * freq * t_continuous)
    
    print(f"\n📊 Señal original: {freq} Hz")
    print(f"   Frecuencia de Nyquist requerida: {2 * freq} Hz")
    
    # Diferentes tasas de muestreo
    sample_rates = [
        (200, "Submuestreo (< Nyquist)"),
        (880, "En Nyquist"),
        (2000, "Sobremuestreo (> Nyquist)"),
        (8000, "Sobremuestreo alto")
    ]
    
    fig, axes = plt.subplots(len(sample_rates), 1, figsize=(12, 10))
    
    for idx, (fs, description) in enumerate(sample_rates):
        # Muestrear
        t_sampled = np.arange(0, duration, 1/fs)
        signal_sampled = np.sin(2 * np.pi * freq * t_sampled)
        
        # Graficar
        axes[idx].plot(t_continuous, signal_continuous, 'b-', alpha=0.3, 
                      linewidth=1, label='Señal continua')
        axes[idx].plot(t_sampled, signal_sampled, 'ro-', markersize=4, 
                      linewidth=1, label=f'Muestreada a {fs} Hz')
        axes[idx].set_ylabel('Amplitud')
        axes[idx].set_title(f'{description} (fs = {fs} Hz, Nyquist = {2*freq} Hz)')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, 0.02)
        
        if idx == len(sample_rates) - 1:
            axes[idx].set_xlabel('Tiempo (s)')
        
        # Análisis
        if fs < 2 * freq:
            print(f"\n   ⚠️  {description} (fs = {fs} Hz):")
            print(f"      ALIASING: La señal aparecerá como {abs(freq - fs)} Hz")
        elif fs == 2 * freq:
            print(f"\n   ⚡ {description} (fs = {fs} Hz):")
            print(f"      Justo en el límite de Nyquist (no recomendado en práctica)")
        else:
            print(f"\n   ✓ {description} (fs = {fs} Hz):")
            print(f"      Muestreo adecuado - reconstrucción posible")
    
    plt.tight_layout()
    plt.savefig('nyquist_aliasing.png', dpi=300)
    print("\n✓ Gráfica guardada: nyquist_aliasing. png")
    
    print("\n📊 TEOREMA DE NYQUIST:")
    print("   Para reconstruir perfectamente una señal, la frecuencia de muestreo")
    print("   debe ser al menos el doble de la frecuencia máxima de la señal:")
    print("   fs ≥ 2 · fmax")
    print("=" * 60 + "\n")


def test_filter_types():
    """
    Demuestra diferentes tipos de filtros digitales.
    """
    print("=" * 60)
    print("DEMOSTRACIÓN:  TIPOS DE FILTROS DIGITALES")
    print("=" * 60)
    
    sample_rate = 44100
    duration = 1.0
    t = np. linspace(0, duration, int(sample_rate * duration))
    
    # Crear señal con múltiples componentes
    signal = (
        np.sin(2 * np.pi * 100 * t) +    # Bajo
        np.sin(2 * np.pi * 1000 * t) +   # Medio
        np.sin(2 * np.pi * 5000 * t)     # Alto
    )
    
    print("\n📊 Señal de prueba: 100 Hz + 1000 Hz + 5000 Hz")
    
    # FFT de la señal original
    fft_original = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    
    # Crear diferentes filtros
    filters = {}
    
    # 1. Paso bajo (deja pasar < 500 Hz)
    lowpass = np.ones(len(fft_original), dtype=complex)
    lowpass[np.abs(freqs) > 500] = 0
    filters['Paso Bajo (< 500 Hz)'] = lowpass
    
    # 2. Paso alto (deja pasar > 2000 Hz)
    highpass = np.ones(len(fft_original), dtype=complex)
    highpass[np.abs(freqs) < 2000] = 0
    filters['Paso Alto (> 2000 Hz)'] = highpass
    
    # 3. Paso banda (deja pasar 500-2000 Hz)
    bandpass = np.zeros(len(fft_original), dtype=complex)
    bandpass[(np.abs(freqs) >= 500) & (np.abs(freqs) <= 2000)] = 1
    filters['Paso Banda (500-2000 Hz)'] = bandpass
    
    # 4. Rechazo de banda (rechaza 500-2000 Hz)
    bandstop = np.ones(len(fft_original), dtype=complex)
    bandstop[(np.abs(freqs) >= 500) & (np.abs(freqs) <= 2000)] = 0
    filters['Rechazo de Banda (500-2000 Hz)'] = bandstop
    
    # Aplicar filtros y visualizar
    fig, axes = plt.subplots(len(filters) + 1, 2, figsize=(14, 14))
    
    # Gráfica de señal original
    positive_mask = freqs >= 0
    freqs_positive = freqs[positive_mask]
    mag_original = np.abs(fft_original[positive_mask])
    mag_original_db = 20 * np.log10(mag_original + 1e-10)
    
    axes[0, 0].plot(t[: 500], signal[:500], linewidth=1, color='blue')
    axes[0, 0].set_title('Señal Original - Dominio del Tiempo')
    axes[0, 0].set_ylabel('Amplitud')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(freqs_positive, mag_original_db, linewidth=1, color='blue')
    axes[0, 1].set_title('Señal Original - Dominio de Frecuencia')
    axes[0, 1].set_ylabel('Magnitud (dB)')
    axes[0, 1].set_xlim(0, 10000)
    axes[0, 1].grid(True, alpha=0.3)
    
    for idx, (filter_name, filter_mask) in enumerate(filters.items(), start=1):
        # Aplicar filtro
        fft_filtered = fft_original * filter_mask
        signal_filtered = np.fft.ifft(fft_filtered).real
        
        # Magnitud filtrada
        mag_filtered = np.abs(fft_filtered[positive_mask])
        mag_filtered_db = 20 * np.log10(mag_filtered + 1e-10)
        
        # Graficar tiempo
        axes[idx, 0].plot(t[:500], signal_filtered[:500], linewidth=1, color='red')
        axes[idx, 0].set_title(f'{filter_name} - Tiempo')
        axes[idx, 0].set_ylabel('Amplitud')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Graficar frecuencia
        axes[idx, 1].plot(freqs_positive, mag_filtered_db, linewidth=1, color='red')
        axes[idx, 1].set_title(f'{filter_name} - Frecuencia')
        axes[idx, 1].set_ylabel('Magnitud (dB)')
        axes[idx, 1].set_xlim(0, 10000)
        axes[idx, 1].grid(True, alpha=0.3)
        
        print(f"\n   {filter_name}:")
        # Análisis de componentes preservadas
        if 'Bajo' in filter_name:
            print("      ✓ Preserva:  100 Hz")
            print("      ✗ Elimina: 1000 Hz, 5000 Hz")
        elif 'Alto' in filter_name:
            print("      ✗ Elimina: 100 Hz, 1000 Hz")
            print("      ✓ Preserva: 5000 Hz")
        elif 'Paso Banda' in filter_name: 
            print("      ✗ Elimina: 100 Hz, 5000 Hz")
            print("      ✓ Preserva: 1000 Hz")
        else:  # Rechazo de banda
            print("      ✓ Preserva: 100 Hz, 5000 Hz")
            print("      ✗ Elimina: 1000 Hz")
    
    axes[-1, 0].set_xlabel('Tiempo (s)')
    axes[-1, 1].set_xlabel('Frecuencia (Hz)')
    
    plt.tight_layout()
    plt.savefig('filter_types.png', dpi=300)
    print("\n✓ Gráfica guardada: filter_types.png")
    
    print("\n📊 TIPOS DE FILTROS:")
    print("   • Paso Bajo: Deja pasar frecuencias bajas, atenúa altas")
    print("   • Paso Alto: Deja pasar frecuencias altas, atenúa bajas")
    print("   • Paso Banda: Deja pasar rango específico de frecuencias")
    print("   • Rechazo de Banda: Atenúa rango específico (notch filter)")
    print("=" * 60 + "\n")


if __name__ == "__main__": 
    print("\n🔬 SUITE DE PRUEBAS AVANZADAS - CONCEPTOS MATEMÁTICOS\n")
    
    test_fourier_properties()
    test_series_de_fourier()
    test_convolucion()
    test_windowing()
    test_nyquist_aliasing()
    test_filter_types()
    
    print("=" * 60)
    print("✅ TODAS LAS DEMOSTRACIONES COMPLETADAS")
    print("=" * 60)
    print("\n📁 Archivos generados:")
    print("  📊 fourier_series_reconstruction.png - Series de Fourier")
    print("  📊 convolution_demo.png - Convolución y filtrado")
    print("  📊 windowing_comparison.png - Efecto del ventaneo")
    print("  📊 nyquist_aliasing.png - Teorema de Nyquist")
    print("  📊 filter_types. png - Tipos de filtros digitales")
    print("\n🎓 Conceptos demostrados:")
    print("  ✅ Linealidad de la Transformada de Fourier")
    print("  ✅ Desplazamiento en frecuencia")
    print("  ✅ Teorema de Parseval")
    print("  ✅ Reconstrucción mediante FFT/IFFT")
    print("  ✅ Series de Fourier y convergencia")
    print("  ✅ Propiedad de convolución")
    print("  ✅ Ventaneo y fuga espectral")
    print("  ✅ Teorema de muestreo de Nyquist")
    print("  ✅ Aliasing")
    print("  ✅ Filtros digitales (paso bajo, alto, banda, rechazo)")
    print("\n🚀 Para ejecutar:")
    print("   python advanced_test.py")
    print()