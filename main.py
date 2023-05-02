import matplotlib.pyplot as plt
import numpy as np
import time


def fourier_term(x, k, N):
    angle = 2 * np.pi * k * np.arange(N) / N
    cos_term = np.sum(x * np.cos(angle))
    sin_term = np.sum(x * np.sin(angle))
    return cos_term, sin_term


def fourier_coefficients(x, N):
    coefficients = np.zeros(N, dtype=complex)

    for k in range(N):
        a_k, b_k = fourier_term(x, k, N)
        coefficients[k] = a_k + 1j * b_k

    return coefficients


def time_and_operation_count(x, N):
    start_time = time.time()

    cos_operations = N * (N - 1)
    sin_operations = N * (N - 1)
    add_operations = 2 * N * (N - 1)
    mult_operations = 2 * N * (N - 1)

    total_operations = cos_operations + sin_operations + add_operations + mult_operations

    coeffs = fourier_coefficients(x, N)

    elapsed_time = time.time() - start_time

    return elapsed_time, total_operations, coeffs


def plot_spectra(coeffs, N):
    amplitude_spectrum = np.abs(coeffs)
    phase_spectrum = np.angle(coeffs)

    plt.figure()
    plt.stem(np.arange(N), amplitude_spectrum, 'b', )
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Amplitude Spectrum')

    plt.figure()
    plt.stem(np.arange(N), phase_spectrum, 'b', )
    plt.xlabel('Frequency')
    plt.ylabel('Phase')
    plt.title('Phase Spectrum')

    plt.show()


def main():
    N = 30
    np.random.seed(42)
    x = np.random.rand(N)

    elapsed_time, total_operations, coeffs = time_and_operation_count(x, N)

    print(f"Час обчислення: {elapsed_time:.6f} секунд")
    print(f"Всього операцій: {total_operations}")

    print("Коефіцієнти Фур'є:")
    for i, coeff in enumerate(coeffs):
        print(f"C_{i} = {coeff:.4f}")

    plot_spectra(coeffs, N)


if __name__ == '__main__':
    main()
