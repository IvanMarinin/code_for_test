from read_data import load_data_from_excel_with_defect
from signal_analyzer import SignalAnalyzer


file_path2 = r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\Пробная задача про корреляции\Data_2Channel.xlsx"
file_path3 = r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\Пробная задача про корреляции\Data_3channel.xlsx"

spectra2 = load_data_from_excel_with_defect(file_path2)
spectra3 = load_data_from_excel_with_defect(file_path3)

analyser = SignalAnalyzer(spectra3)

test = analyser.different_smoothing_parallel(31,4)
best = analyser.find_best_rms()
analyser.apply_best_smoothing()
best = analyser.correlation(True)