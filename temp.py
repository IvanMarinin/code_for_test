from read_data import load_data_from_excel_with_defect


file_path2 = r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\Пробная задача про корреляции\Data_2Channel.xlsx"
file_path3 = r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\Пробная задача про корреляции\Data_3channel.xlsx"

spectra2 = load_data_from_excel_with_defect(file_path2)
spectra3 = load_data_from_excel_with_defect(file_path3)

# Normalization
for s in spectra2:
    s.normalize()
for s in spectra3:
    s.normalize()


# Before smoothing
spectra2[0].correlation(spectra2, True)
spectra3[0].correlation(spectra3, True)

# After smoothing
for s in spectra2:
    s.savgol_filter(21, 2)
    s.normalize()

for s in spectra3:
    s.savgol_filter(21, 2)
    s.normalize()

spectra2[0].correlation(spectra2, True)
spectra3[0].correlation(spectra3, True)