import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from signal_class import Signal
from joblib import Parallel, delayed

from spectrum_class import Spectrum


class SignalAnalyzer:
    """
    A class for analyzing a set of Signal or Spectrum or Noise objects, including:
    - computing averages
    - applying Savitzky-Golay smoothing
    - computing Pearson correlations between spectral features and defect parameters
    - searching for optimal smoothing parameters in parallel
    """

    def __init__(self, signals):
        """
        Initialize the analyzer with a list of Signal objects.

        Parameters:
        - signals: list of Signal objects
        """
        self.signals = signals
        self.correlation_matrices = []         # stores correlation matrices for different smoothing params
        self.best_correlation_matrix = None    # stores the result with highest RMS

    def average(self, normalize=True):
        """
        Compute the average spectrum from all signals.

        Parameters:
        - normalize: bool, whether to normalize each signal before averaging

        Returns:
        - Signal object representing the averaged spectrum
        """
        if not self.signals:
            raise ValueError("Signal list is empty")

        wavelengths = self.signals[0].wavelengths

        # Ensure all signals have the same wavelengths
        for s in self.signals:
            if not np.array_equal(s.wavelengths, wavelengths):
                raise ValueError("Wavelengths are not equal across signals")

        if normalize:
            for s in self.signals:
                s.normalize()

        intensities_stack = np.array([s.intensities for s in self.signals])
        mean_intensities = intensities_stack.mean(axis=0)

        return Signal(filename="Average signal", wavelengths=wavelengths, intensities=mean_intensities)

    def correlation(self, show=False):
        """
        Compute Pearson correlation between spectral features and defect parameters.

        Parameters:
        - show: bool, whether to display a heatmap

        Returns:
        - DataFrame of correlations (features x defect parameters)
        """
        # Collect feature and defect parameter lists
        features_data = [s.get_features_list for s in self.signals]
        params_data = [s.get_defect_params_list for s in self.signals]

        feature_columns = ["mean", "std", "modality", "skewness", "kurtosis",
                           "iqr", "snr", "max_i", "sum_i", "entropy", "freq_magnitude"]
        param_columns = ["h_i", "h_u", "h_g", "h_e", "h_p", "h_s", "h_m"]

        df_features = pd.DataFrame(features_data, columns=feature_columns)
        df_params = pd.DataFrame(params_data, columns=param_columns)
        df = pd.concat([df_features, df_params], axis=1)

        corr_matrix = df.corr(method='pearson')

        if show:
            # Only show correlation between features and defect parameters
            corr_subset = corr_matrix.loc[feature_columns, param_columns]
            plt.figure(figsize=(11, 7))
            sns.heatmap(corr_subset, annot=True, cmap="coolwarm", center=0, linewidths=0.5)
            plt.title("Correlation between spectrum features and defect parameters")
            plt.xlabel("Defect parameters")
            plt.ylabel("Spectrum features")
            plt.show()

        return corr_matrix.loc[feature_columns, param_columns]

    def apply_savgol(self, window_length, polyorder, normalize_before=True, normalize_after=True):
        """
        Apply Savitzky-Golay smoothing to all signals with optional normalization.

        Parameters:
        - window_length: int, length of filter window (must be odd and > polyorder)
        - polyorder: int, polynomial order
        - normalize_before: bool, normalize before smoothing
        - normalize_after: bool, normalize after smoothing
        """
        for s in self.signals:
            if normalize_before:
                s.normalize()
            s.savgol_filter(window_length, polyorder)
            if normalize_after:
                s.normalize()

    def remove_baseline(self, normalize_before=True, normalize_after=True, method='mpls', **params):
        """
        Remove baseline from all spectra in the analyzer.

        Parameters:
        - normalize_before: bool, normalize before smoothing
        - normalize_after: bool, normalize after smoothing
        - method: baseline removal method ('asls', 'airpls', etc.)
        - **params: parameters for the baseline method
        """
        for s in self.signals:
            if normalize_before:
                s.normalize()
            s.flatten(method=method, **params)
            if normalize_after:
                s.normalize()

    def reset_all_signals(self):
        """Restore all signals to their original intensities."""
        for s in self.signals:
            s.reset_intensities()

    def smooth_and_correlate(self, window_length, polyorder):
        """
        Apply smoothing, compute correlation, and RMS.

        Returns a dictionary with window_length, polyorder, RMS, and correlation matrix.
        """
        self.apply_savgol(window_length, polyorder)
        corr_matrix = self.correlation(show=False)
        self.reset_all_signals()
        rms = np.sqrt(np.mean(np.nan_to_num(corr_matrix.to_numpy()) ** 2))

        return {
            "window_length": window_length,
            "polyorder": polyorder,
            "rms": rms,
            "correlation": corr_matrix
        }

    def different_smoothing_parallel(self, window_length_max, polyorder_max, n_jobs=-1):
        """
        Run smoothing for multiple combinations of window_length and polyorder in parallel.

        Parameters:
        - window_length_max: max window length to try
        - polyorder_max: max polynomial order to try
        - n_jobs: number of parallel jobs

        Returns:
        - List of dictionaries with smoothing results
        """
        combinations = [(w, p) for w in range(3, window_length_max, 2) for p in range(1, min(polyorder_max, w))]
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.smooth_and_correlate)(w, p) for w, p in combinations
        )
        self.correlation_matrices = results
        return results

    def find_best_rms(self):
        """Select the smoothing parameters with the highest RMS correlation."""
        if not self.correlation_matrices:
            raise ValueError("No correlation results. Run different_smoothing_parallel first.")
        self.best_correlation_matrix = max(self.correlation_matrices, key=lambda x: x["rms"])
        return self.best_correlation_matrix

    def apply_best_smoothing(self):
        """Apply the best smoothing parameters to all signals."""
        if not self.best_correlation_matrix:
            raise ValueError("Best result not set. Run find_best_rms first.")
        w = self.best_correlation_matrix["window_length"]
        p = self.best_correlation_matrix["polyorder"]
        self.apply_savgol(w, p)

    @classmethod
    def merge_spectra(cls, spectra1, spectra2):
        """
        Merge two lists of spectra by concatenating their wavelengths and intensities.
        Spectra must have matching parameters.
        """
        spectra = []
        # Check if spectra lists have the same length
        if len(spectra1) != len(spectra2):
            raise ValueError(f"Cannot merge spectra lists of different lengths: {len(spectra1)} != {len(spectra2)}")

        # Merge each corresponding pair of spectra
        for i in range(len(spectra1)):
            # List of spectral parameters to check for equality
            attributes = ['h_u', 'h_g', 'h_e', 'h_p', 'h_s', 'h_m', 'h_i']
            mismatches = []

            # Check if all parameters match between the two spectra
            for attr in attributes:
                val1 = getattr(spectra1[i], attr)
                val2 = getattr(spectra2[i], attr)
                if math.isnan(val1) and math.isnan(val2):
                    # Both are NaN - consider them equal
                    continue
                elif val1 != val2:
                    mismatches.append(f"{attr}: {val1} != {val2}")

            # If all parameters match, merge the spectra
            if not mismatches:
                # Concatenate wavelengths and intensities from both spectra
                merged_wavelengths = np.concatenate([spectra1[i].wavelengths, spectra2[i].wavelengths])
                merged_intensities = np.concatenate([spectra1[i].intensities, spectra2[i].intensities])

                # Get sorting indices based on wavelengths
                sort_indices = np.argsort(merged_wavelengths)

                # Apply the same sorting to both arrays to maintain wavelength-intensity correspondence
                sorted_wavelengths = merged_wavelengths[sort_indices]
                sorted_intensities = merged_intensities[sort_indices]
                merged_spectrum = Spectrum(
                    filename=spectra1[i].filename,
                    wavelengths=sorted_wavelengths,
                    intensities=sorted_intensities,
                    h_u=spectra1[i].h_u,
                    h_g=spectra1[i].h_g,
                    h_e=spectra1[i].h_e,
                    h_p=spectra1[i].h_p,
                    h_s=spectra1[i].h_s,
                    h_m=spectra1[i].h_m,
                    h_i=spectra1[i].h_i
                )
                spectra.append(merged_spectrum)
            else:
                # Raise detailed error if parameters don't match
                raise ValueError(
                    f'Merging parts of different spectra is impossible for spectrum {i}. '
                    f'Mismatches: {", ".join(mismatches)}'
                )

        return spectra