import numpy as np
from scipy.stats import skew, kurtosis, entropy, gaussian_kde
from signal_class import Signal


class Spectrum(Signal):
    def __init__(self, filename, wavelengths, intensities,
                 h_u=None, h_g=None, h_e=None, h_p=None,
                 h_s=None, h_m=None, h_i=None):
        super().__init__(filename, wavelengths, intensities)
        self.h_u = h_u
        self.h_g = h_g
        self.h_e = h_e
        self.h_p = h_p
        self.h_s = h_s
        self.h_m = h_m
        self.h_i = h_i

    @property
    def mean(self):
        return np.mean(self.intensities)

    @property
    def std(self):
        return np.std(self.intensities)

    @property
    def Modality(self):
        return 0  # Надо дописать нормально

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
        freq_spectrum = np.fft.fft(self.intensities)
        return np.abs(freq_spectrum).max()

    @property
    def get_features_list(self):
        return [
            self.mean,
            self.std,
            self.Modality,
            self.skewness,
            self.kurtosis,
            self.iqr,
            self.snr,
            self.max_i,
            self.sum_i,
            self.entropy,
            self.freq_magnitude
        ]

    @property
    def get_defect_params_list(self):
        return [
            self.h_i,
            self.h_u,
            self.h_g,
            self.h_e,
            self.h_p,
            self.h_s,
            self.h_m
        ]
