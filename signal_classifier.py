import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from read_data import load_spectrum_raw
from pathlib import Path
from noise_class import Noise
from spectrum_class import Spectrum
from imblearn.over_sampling import SMOTE



def classify_signals(folder_path, spectrum_names):
    folder = Path(folder_path)
    spectra = []
    noises = []

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() == '.raw8':
            sign = load_spectrum_raw(file)

            if file.name in spectrum_names:
                spectrum = Spectrum(sign.filename, sign.wavelengths, sign.intensities, sign.int_time, sign.averages, sign.smoothing)
                spectra.append(spectrum)
            else:
                noise = Noise(sign.filename, sign.wavelengths, sign.intensities, sign.int_time, sign.averages, sign.smoothing)
                noises.append(noise)

    return spectra, noises




folder_path = r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\2023-0101 Spectroscopy\10"
spectrum_names = [
    "1712307U3_0031.Raw8", "1712307U3_0032.Raw8", "1712307U3_0033.Raw8", "1712307U3_0034.Raw8",
    "1712307U3_0035.Raw8", "1712307U3_0036.Raw8", "1712307U3_0037.Raw8", "1712307U3_0038.Raw8",
    "1712307U3_0039.Raw8", "1712307U3_0040.Raw8", "1712307U3_0041.Raw8", "1712307U3_0042.Raw8",
    "1712307U3_0043.Raw8", "1712307U3_0044.Raw8", "1712307U3_0045.Raw8", "1712307U3_0046.Raw8",
    "1712307U3_0047.Raw8", "1712307U3_0048.Raw8"
]

folder_path2 = r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\2023-0101 Spectroscopy\11"
spectrum_names2 = [
    "1712307U3_0031.Raw8", "1712307U3_0032.Raw8", "1712307U3_0033.Raw8", "1712307U3_0034.Raw8",
    "1712307U3_0035.Raw8", "1712307U3_0036.Raw8", "1712307U3_0037.Raw8", "1712307U3_0038.Raw8",
    "1712307U3_0039.Raw8", "1712307U3_0040.Raw8", "1712307U3_0041.Raw8", "1712307U3_0042.Raw8",
    "1712307U3_0043.Raw8", "1712307U3_0044.Raw8", "1712307U3_0045.Raw8", "1712307U3_0046.Raw8",
    "1712307U3_0047.Raw8", "1712307U3_0048.Raw8"
]




def preparate_dataset(spectra, noises):
    X = []
    y = []
    for spec in spectra:
        X.append(spec.get_features())
        y.append(1)
    for noise in noises:
        X.append(noise.get_features())
        y.append(0)
    return np.array(X), np.array(y)


spectra, noises = classify_signals(folder_path, spectrum_names)

# Первая часть — обучение
X, y = preparate_dataset(spectra, noises)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=10000, random_state=42)
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test_scaled)
print("Валидация на первой выборке:")
print(classification_report(y_test, y_pred))

# Вторая часть — проверка на внешней (новой) выборке
spectra, noises = classify_signals(folder_path2, spectrum_names2)
X_ext, y_ext = preparate_dataset(spectra, noises)

X_ext_scaled = scaler.transform(X_ext)
y_pred_ext = model.predict(X_ext_scaled)

print("Оценка на внешней выборке:")
print(classification_report(y_ext, y_pred_ext))