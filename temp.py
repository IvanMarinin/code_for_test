import numpy as np
import struct
from signal_class import Signal
from read_data import load_data_from_excel_with_defect


file_path = r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\Пробная задача про корреляции\Data_2Channel.xlsx"
spectra = load_data_from_excel_with_defect(file_path)

print(spectra[2].mean)