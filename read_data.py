import numpy as np
import pandas as pd
import os
import struct
from signal_class import Signal
from spectrum_class import Spectrum


def load_spectrum_raw(file_path, header_size=328, n_points=2048):
    """
    Load a single spectrum from a binary raw file.

    Parameters:
    - file_path: path to the .raw file
    - header_size: number of bytes in the header (default 328)
    - n_points: number of wavelength points (default 2048)

    Returns:
    - Signal object
    """
    # Read header
    with open(file_path, 'rb') as f:
        header = f.read(header_size)

    # Extract metadata from header
    integration_time_ms = struct.unpack_from('<f', header, 93)[0]  # integration time
    averages = struct.unpack_from('<I', header, 101)[0]            # number of averages
    smoothing = struct.unpack_from('<I', header, 8)[0]             # smoothing
    filename = os.path.basename(file_path)

    # Read data: first n_points = wavelengths, next n_points = intensities
    data = np.fromfile(file_path, dtype=np.float32, offset=header_size)
    wavelengths = data[:n_points]
    intensities = data[n_points: 2*n_points]

    # Create Signal object
    spec = Signal(filename, wavelengths, intensities,
                  int_time=integration_time_ms,
                  averages=averages,
                  smoothing=smoothing)
    return spec


def load_spectrum_excel(file_path):
    """
    Load multiple spectra from an Excel file where each column is one spectrum.

    Parameters:
    - file_path: path to the Excel file

    Returns:
    - list of Signal objects
    """
    spectra = []
    params_data = pd.read_excel(file_path, header=None)

    # Extract file names and acquisition parameters
    file_names = params_data.iloc[0, 1:].astype(str).values
    int_time = params_data.iloc[2, 1:].values
    averages = params_data.iloc[3, 1:].values
    smoothing = params_data.iloc[4, 1:].values

    # Extract wavelengths and intensities
    wavelengths = params_data.iloc[6:, 0].values
    intensities = params_data.iloc[6:, 1:].values

    # Create Signal objects for each column
    for i, filename in enumerate(file_names):
        spec = Signal(filename, wavelengths, intensities[:, i],
                      int_time=int_time[i],
                      averages=averages[i],
                      smoothing=smoothing[i])
        spectra.append(spec)
    return spectra


def load_data_from_excel_with_defect(file_path):
    """
    Load spectra and associated defect parameters from Excel.

    Assumes:
    - First 2048 columns = spectrum intensities
    - Next 7 columns = defect parameters: h_u, h_g, h_e, h_p, h_s, h_m, h_i

    Parameters:
    - file_path: path to Excel file

    Returns:
    - list of Spectrum objects
    """
    spectra = []
    params_data = pd.read_excel(file_path, header=None)

    # First row: wavelengths
    wavelengths = params_data.iloc[0, :2048].values
    # Remaining rows: intensities and defects
    intensities = params_data.iloc[1:, :2048].values
    defects = params_data.iloc[1:, 2048:].values

    # Create Spectrum objects
    for i in range(intensities.shape[0]):
        spec = Spectrum(
            filename=f"Spectrum_{i}",
            wavelengths=wavelengths,
            intensities=intensities[i],
            h_u=defects[i, 0],
            h_g=defects[i, 1],
            h_e=defects[i, 2],
            h_p=defects[i, 3],
            h_s=defects[i, 4],
            h_m=defects[i, 5],
            h_i=defects[i, 6]
        )
        spectra.append(spec)

    return spectra
