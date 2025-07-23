import numpy as np
import pandas as pd
import os
import struct
from signal_class import Signal


def load_spectrum_raw(file_path, header_size = 328, n_points = 2048):

    with open(file_path, 'rb') as f:
        header = f.read(header_size)

    integration_time_ms = struct.unpack_from('<f', header, 93)[0]
    averages = struct.unpack_from('<I', header, 101)[0]
    smoothing = struct.unpack_from('<I', header, 8)[0]
    filename = os.path.basename(file_path)

    data = np.fromfile(file_path, dtype=np.float32, offset=header_size)
    wavelengths = data[:n_points]
    intensities = data[n_points: 2*n_points]

    spec = Signal(filename, wavelengths, intensities, int_time=integration_time_ms, averages=averages, smoothing=smoothing)
    return spec

def load_spectrum_excel(file_path):
    spectra = []
    params_data = pd.read_excel(file_path, header=None)

    file_names = params_data.iloc[0, 1:].astype(str).values
    int_time = params_data.iloc[2, 1:].values
    averages = params_data.iloc[3, 1:].values
    smoothing = params_data.iloc[4, 1:].values

    wavelengths = params_data.iloc[6:,0].values
    intensities = params_data.iloc[6:,1:].values

    for i, filename in enumerate(file_names):
        spec = Signal(filename, wavelengths, intensities[:,i], int_time=int_time[i], averages=averages[i], smoothing=smoothing[i])
        spectra.append(spec)
    return spectra

