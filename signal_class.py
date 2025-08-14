import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis, entropy, gaussian_kde


class Signal:
    def __init__(self, filename, wavelengths, intensities, int_time=None, averages=None, smoothing=None):
        """
        A class to store and process signal (spectrum) data.

        Parameters:
        filename: str - name of the source file
        wavelengths: array of wavelengths (nm)
        intensities: array of intensities
        int_time: optional, integration time
        averages: optional, number of averages in measurement
        smoothing: optional, smoothing params
        """
        self.filename = filename
        self.wavelengths = np.array(wavelengths, dtype=float)
        self.intensities = np.array(intensities, dtype=float)
        self.original_intensities = self.intensities.copy()
        self.int_time = int_time
        self.averages = averages
        self.smoothing = smoothing

    def info(self):
        """Prints basic information about the signal."""
        print(f"File: {self.filename}")
        print(f"Wavelength range: {self.wavelengths[0]:.2f}–{self.wavelengths[-1]:.2f} nm")
        print(f"Points: {len(self.wavelengths)}")
        print(f"Integration time: {self.int_time}")
        print(f"Averages: {self.averages}")

    def plot_intensities(self, show = True):
        """Plots the current intensities vs. wavelength."""
        plt.plot(self.wavelengths, self.intensities)
        plt.title(f'{self.__class__.__name__}: {self.filename}')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        if show: plt.show()

    def reset_intensities(self):
        """Restores intensities to their original state."""
        self.intensities = self.original_intensities.copy()

    def normalize(self):
        """Normalizes intensities so the maximum becomes 1."""
        if self.intensities.max() == 0:
            raise ValueError("Max intensity is zero, normalization impossible")
        self.intensities = self.intensities / self.intensities.max()

    def fourier_transform(self, show = False):
        """
        Performs FFT (Fast Fourier Transform) on intensities.
        Returns: (fft array, frequency array)
        """
        N = len(self.intensities)
        T = np.mean(np.diff(self.wavelengths))
        fft = np.fft.fft(self.intensities)
        frequency = np.fft.fftfreq(N, T)

        if show:
            plt.plot(frequency[:N//2], np.abs(fft)[:N//2])
            plt.title(f'Fast Fourier Transform of {self.filename}')
            plt.xlabel('Frequency, $nm^{-1}$')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.show()

        return fft, frequency

    def wavelet_transform(self, wavelet = 'db4', level = 4, show = False):
        """
        Performs wavelet decomposition.
        Returns: list of coefficient arrays.
        """
        coefficients = pywt.wavedec(self.intensities, wavelet, level=level)
        if show:
            fig, axes = plt.subplots(len(coefficients),1 , figsize = (10, 2*len(coefficients)))
            for i, coeff in enumerate(coefficients):
                axes[i].plot(coeff)
                axes[i].set_title(f'Parameters of {i} coefficient')
            plt.tight_layout()
            plt.show()
        return coefficients

    def savgol_filter(self, window_length, polyorder):
        """Applies Savitzky–Golay smoothing filter."""
        self.intensities = savgol_filter(self.intensities, window_length=window_length, polyorder=polyorder)

    def butter_filter(self, cutoff, order = 5, btype ='low'):
        """Applies Butterworth filter."""
        fs = 1 / np.mean(np.diff(self.wavelengths))
        half_fs = fs/2
        normal_cutoff = cutoff / half_fs
        if not 0 < normal_cutoff < 1:
            raise ValueError(f"Cutoff must be between 0 and {half_fs:.3f}, got {cutoff:.3f}")
        b, a = butter(order, normal_cutoff, btype = btype, analog = False)
        self.intensities = filtfilt(b, a, self.intensities)

    def modality(self, bandwidth=None, grid_points=2048, prominence=None, pad=0.05):
        """
        Calculates number of modes (peaks) in the intensity distribution using KDE.
        """

        x = self.intensities
        n = len(x)

        if n < 3:
            return 1
        if np.std(x) == 0:
            return 1

        kde = gaussian_kde(x)

        if bandwidth is not None:
            kde.set_bandwidth(bw_method=bandwidth / np.std(x))

        xmin, xmax = x.min(), x.max()
        span = xmax - xmin if xmax > xmin else 1.0
        xmin -= pad * span
        xmax += pad * span
        grid = np.linspace(xmin, xmax, grid_points)

        kde_values = kde(grid)

        if prominence is None:
            prominence = 0.05 * np.max(kde_values)

        peaks, _ = find_peaks(kde_values, prominence=prominence)

        return len(peaks)

    @property
    def mean(self):
        return np.mean(self.intensities)

    @property
    def std(self):
        return np.std(self.intensities)

    @property
    def skewness(self):
        return skew(self.intensities)

    @property
    def kurtosis(self):
        return kurtosis(self.intensities)

    @property
    def iqr(self):
        return np.percentile(self.intensities, 75) - np.percentile(self.intensities, 25)

    @property
    def snr(self):
        numerator = np.sqrt(np.sum(self.intensities ** 2) / len(self.intensities))
        denominator = np.sqrt(np.sum((self.intensities - self.mean) ** 2) / (len(self.intensities) - 1))

        if denominator == 0:
            raise ValueError("Denominator equals 0")
        return numerator / denominator

    @property
    def max_i(self):
        return np.max(self.intensities)

    @property
    def sum_i(self):
        return np.sum(self.intensities)

    @property
    def entropy(self):
        hist, _ = np.histogram(self.intensities, bins=512, density=True)
        hist = hist[hist > 0]
        return entropy(hist)  # Уточнить формулу

    @property
    def freq_magnitude(self):
        freq_spectrum = self.fourier_transform(show = False)[0]
        return np.abs(freq_spectrum).max()

    @property
    def get_features_list(self):
        return [
            self.mean,
            self.std,
            self.modality(),
            self.skewness,
            self.kurtosis,
            self.iqr,
            self.snr,
            self.max_i,
            self.sum_i,
            self.entropy,
            self.freq_magnitude
        ]

    @classmethod
    def average(cls, signal_list, normalize=True):
        """
        Creates an average signal from a list of Signal objects.
        """
        if not signal_list:
            raise ValueError(f"{cls.__name__} list is empty")

        wavelengths = signal_list[0].wavelengths

        for sign in signal_list:
            if not np.array_equal(sign.wavelengths, wavelengths):
                raise ValueError("Wavelengths are not equal")

        if normalize:
            for sign in signal_list:
                sign.normalize()

        intensities_stack = np.array([sign.intensities for sign in signal_list])
        mean_intensities = intensities_stack.mean(axis=0)
        average_signal = cls(filename=f"Average {cls.__name__}", wavelengths=wavelengths, intensities=mean_intensities)

        return average_signal
