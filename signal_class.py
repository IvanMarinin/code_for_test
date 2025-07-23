import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.stats import skew, kurtosis

class Signal:
    def __init__(self, filename, wavelengths, intensities, int_time=None, averages=None, smoothing=None):
        self.filename = filename
        self.wavelengths = np.array(wavelengths, dtype=float)
        self.intensities = np.array(intensities, dtype=float)
        self.original_intensities = self.intensities.copy()
        self.int_time = int_time
        self.averages = averages
        self.smoothing = smoothing

    def info(self):
        print(f"File: {self.filename}")
        print(f"Wavelength range: {self.wavelengths[0]:.2f}–{self.wavelengths[-1]:.2f} nm")
        print(f"Points: {len(self.wavelengths)}")
        print(f"Integration time: {self.int_time}")
        print(f"Averages: {self.averages}")

    def plot_intensities(self, show = True):
        plt.plot(self.wavelengths, self.intensities)
        plt.title(f'{self.__class__.__name__}: {self.filename}')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        if show: plt.show()

    def reset_intensities(self):
        self.intensities = self.original_intensities.copy()

    def normalize(self):
        if self.intensities.max() == 0:
            raise ValueError("Max intensity is zero, normalization impossible")
        self.intensities = self.intensities / self.intensities.max()

    def fourier_transform(self, show = False):
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
        self.intensities = savgol_filter(self.intensities, window_length=window_length, polyorder=polyorder)

    def butter_filter(self, cutoff, order = 5, btype ='low'):
        fs = 1 / np.mean(np.diff(self.wavelengths))
        half_fs = fs/2
        normal_cutoff = cutoff / half_fs
        if not 0 < normal_cutoff < 1:
            raise ValueError(f"Cutoff must be between 0 and {half_fs:.3f}, got {cutoff:.3f}")
        b, a = butter(order, normal_cutoff, btype = btype, analog = False)
        self.intensities = filtfilt(b, a, self.intensities)

    def get_features(self, normalize = True):
        if normalize: self.normalize()
        features = []
        features.append(np.mean(self.intensities))
        features.append(np.std(self.intensities))
        features.append(np.max(self.intensities)/np.mean(self.intensities))
        features.append(np.sum(self.intensities ** 2))
        features.append(skew(self.intensities))
        features.append(kurtosis(self.intensities))
        features.append(np.argmax(self.intensities))
        return features

    @classmethod
    def average(cls, signal_list, normalize=True):
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
