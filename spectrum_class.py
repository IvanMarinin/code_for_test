import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from signal_class import Signal


class Spectrum(Signal):
    def __init__(self, filename, wavelengths, intensities,
                 h_u=None, h_g=None, h_e=None, h_p=None,
                 h_s=None, h_m=None, h_i=None):
        """
        Spectrum extends Signal by adding defect-related parameters.
        Each spectrum has optional defect parameters h_u, h_g, h_e, h_p, h_s, h_m, h_i.
        """
        super().__init__(filename, wavelengths, intensities)
        self.h_u = h_u
        self.h_g = h_g
        self.h_e = h_e
        self.h_p = h_p
        self.h_s = h_s
        self.h_m = h_m
        self.h_i = h_i

    @property
    def get_defect_params_list(self):
        """
        Returns defect parameters as a list in a fixed order.
        Useful for building DataFrames or correlation analysis.
        """
        return [
            self.h_i,
            self.h_u,
            self.h_g,
            self.h_e,
            self.h_p,
            self.h_s,
            self.h_m
        ]

    @classmethod
    def correlation(cls, spectra, draw=False):
        """
        Computes Pearson correlation between spectrum features and defect parameters.

        Parameters:
        - spectra: list of Spectrum objects
        - draw: bool, if True shows a heatmap of correlations

        Returns:
        - DataFrame of correlations (features x defect parameters)
        """
        # Collect feature values and defect parameters for each spectrum
        features_data = [s.get_features_list for s in spectra]
        params_data = [s.get_defect_params_list for s in spectra]

        # Define column names
        feature_columns = ["mean", "std", "modality", "skewness", "kurtosis",
                           "iqr", "snr", "max_i", "sum_i", "entropy", "freq_magnitude"]
        param_columns = ["h_i", "h_u", "h_g", "h_e", "h_p", "h_s", "h_m"]

        # Create DataFrames
        df_features = pd.DataFrame(features_data, columns=feature_columns)
        df_params = pd.DataFrame(params_data, columns=param_columns)
        df = pd.concat([df_features, df_params], axis=1)

        # Compute Pearson correlation
        corr_matrix = df.corr(method='pearson')

        # Optional heatmap visualization
        if draw:
            corr_subset = corr_matrix.loc[feature_columns, param_columns]
            plt.figure(figsize=(11, 7))
            sns.heatmap(corr_subset, annot=True, cmap="coolwarm", center=0, linewidths=0.5)
            plt.title("Correlation between spectrum features and defect parameters")
            plt.xlabel("Defect parameters")
            plt.ylabel("Spectrum features")
            plt.show()

        # Return only the correlation between features and defect parameters
        return corr_matrix.loc[feature_columns, param_columns]