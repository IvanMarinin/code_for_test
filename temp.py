from read_data import load_data_from_excel_with_defect
from signal_analyzer import SignalAnalyzer


file_path2 = r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\Пробная задача про корреляции\Data_2Channel.xlsx"
file_path3 = r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\Пробная задача про корреляции\Data_3channel.xlsx"

spectra2 = load_data_from_excel_with_defect(file_path2)
spectra3 = load_data_from_excel_with_defect(file_path3)

spectra1 = SignalAnalyzer.merge_spectra(spectra1=spectra2,spectra2=spectra3)
spectra1[7].plot_intensities()



"""
analyser = SignalAnalyzer(spectra1)

test = analyser.different_smoothing_parallel(31,4)
print(analyser.find_best_rms())
analyser.apply_best_smoothing()
aver = analyser.average()
aver.plot_intensities()
best = analyser.correlation(True)

for method in ['asls', 'airpls', 'mpls', 'imodpoly', 'combo']:
    analyser.reset_all_signals()
    analyser.remove_baseline(method=method)
    test = analyser.different_smoothing_parallel(31,4)
    print(analyser.find_best_rms())
    analyser.apply_best_smoothing()
    aver = analyser.average()
    aver.plot_intensities()
    best = analyser.correlation(True)
"""