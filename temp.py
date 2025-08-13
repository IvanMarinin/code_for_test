import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import struct
from signal_class import Signal
from read_data import load_data_from_excel_with_defect


file_path2 = r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\Пробная задача про корреляции\Data_2Channel.xlsx"
file_path3 = r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\Пробная задача про корреляции\Data_3channel.xlsx"

spectra2 = load_data_from_excel_with_defect(file_path2)
spectra3 = load_data_from_excel_with_defect(file_path3)

def correlation(spectra, draw=False):
    features_data = [s.get_features_list for s in spectra]
    params_data = [s.get_defect_params_list for s in spectra]

    feature_columns = ["mean", "std", "modality", "skewness", "kurtosis", "iqr", "snr", "max_i", "sum_i", "entropy", "freq_magnitude"]
    param_columns = ["h_i", "h_u", "h_g", "h_e", "h_p", "h_s", "h_m"]

    df_features = pd.DataFrame(features_data, columns=feature_columns)
    df_params = pd.DataFrame(params_data, columns=param_columns)
    df = pd.concat([df_features, df_params], axis=1)
    corr_matrix = df.corr(method='pearson')

    if draw:
        corr_subset = corr_matrix.loc[feature_columns, param_columns]

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            corr_subset,
            annot=True,  # отображать значения коэффициентов на клетках
            cmap="coolwarm",  # красно-синяя цветовая схема
            center=0,  # белый цвет для нуля
            linewidths=0.5
        )
        plt.title("Корреляция признаков спектров и параметров дефектов")
        plt.xlabel("Параметры дефектов")
        plt.ylabel("Признаки спектров")
        plt.show()

    return corr_matrix.loc[feature_columns, param_columns]


correlation(spectra2, True)
correlation(spectra3, True)