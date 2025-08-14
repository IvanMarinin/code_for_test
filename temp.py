from read_data import load_data_from_excel_with_defect


file_path2 = r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\Пробная задача про корреляции\Data_2Channel.xlsx"
file_path3 = r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\Пробная задача про корреляции\Data_3channel.xlsx"

spectra2 = load_data_from_excel_with_defect(file_path2)
spectra3 = load_data_from_excel_with_defect(file_path3)

spectra3[0].plot_intensities()
spectra3[1].plot_intensities()
spectra3[10].plot_intensities()



print("SSSSS")
for s in spectra3:
    print(s.modality())

spectra2[0].correlation(spectra2, True)
spectra3[0].correlation(spectra3, True)