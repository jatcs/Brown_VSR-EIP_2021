import numpy as np
import matplotlib.pyplot as plt


def fun(x):
    return np.sin(2 * np.pi * x) + 1.5 * np.sin(4 * np.pi * x) + 2.5 * np.sin(6 * np.pi * x)


if __name__ == '__main__':
    X = np.linspace(-1, 1, 128)
    Y = [fun(x) for x in X]

    amplitudes = np.fft.fft(Y)
    frequencies = np.fft.fftfreq(128, 2 * np.pi / (128))
    plt.plot(X, Y, 'ro', 'Data set')
    plt.show()
    # fourier transform
    plt.plot(frequencies[0:64], abs(amplitudes[0:64]), 'g*', 'Fourier transform')
    plt.show()
    pass

