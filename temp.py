import numpy as np
import struct

def inspect_header(filepath, header_size=512):
    with open(filepath, 'rb') as f:
        header = f.read(header_size)

    # uint16, uint32, int32, float32 представления
    u16 = np.frombuffer(header, dtype='<u2')
    u32 = np.frombuffer(header, dtype='<u4')
    i32 = np.frombuffer(header, dtype='<i4')
    f32 = np.frombuffer(header, dtype='<f4')

    print("\n=== Смещения uint16 (offset, value) ===")
    for idx, v in enumerate(u16):
        if v in (0,1,2,10):
            print(f"  byte {idx*2:4d}: uint16 = {v}")

    print("\n=== Смещения uint32/int32 (offset, uint32, int32) ===")
    for idx, (uv, iv) in enumerate(zip(u32, i32)):
        if uv in (0,1,2,10,2048) or iv in (0,1,2,10,2048):
            print(f"  byte {idx*4:4d}: uint32 = {uv}, int32 = {iv}")

    print("\n=== Смещения float32 (offset, value) ===")
    for idx, v in enumerate(f32):
        if np.isnan(v):
            continue
        rv = round(v)
        if abs(v - rv) < 1e-6 and rv in (0,1,2,10,2048):
            print(f"  byte {idx*4:4d}: float32 = {v}")

    print("\n=== ASCII dump ===")
    # заменяем непечатаемые символы точкой
    ascii_repr = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in header)
    for i in range(0, len(ascii_repr), 64):
        print(f"{i:04d}: {ascii_repr[i:i+64]}")

#inspect_header(r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\2023-0101 Spectroscopy\10\1712307U3_0037.Raw8")


def dump_ascii_chunks(header_bytes: bytes):
    """
    Разбивает заголовок по нулевому байту и печатает
    все «содержательные» ASCII‑фрагменты.
    """
    # Разбиваем по 0x00
    parts = header_bytes.split(b'\x00')
    for p in parts:
        # снимаем пробелы и отбрасываем слишком короткие куски
        s = p.strip()
        if len(s) < 4:
            continue
        try:
            txt = s.decode('ascii')
        except UnicodeDecodeError:
            continue
        # печатаем, если в тексте есть буквы
        if any(ch.isalpha() for ch in txt):
            print(txt)

# Пример использования:
with open(r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\2023-0101 Spectroscopy\10\1712307U3_0037.Raw8", "rb") as f:
    header = f.read(512)
    dump_ascii_chunks(header)


import struct

def find_le_uint32_offsets(header: bytes, values: dict):
    """
    Для каждого ключа из values (val) ищет в header байтовую подпоследовательность
    little‑endian записи uint32=val и возвращает словарь {key: offset}.
    """
    offsets = {}
    for key, val in values.items():
        pat = struct.pack('<I', val)
        idx = header.find(pat)
        offsets[key] = idx if idx != -1 else None
    return offsets

# Пример использования:
with open(r"C:\Users\marin\OneDrive\Рабочий стол\Научка\Анализ спектра\2023-0101 Spectroscopy\10\1712307U3_0037.Raw8", "rb") as f:
    hdr = f.read(512)

want = {
    'integration_time_ms': 10,
    'averages': 2,
    'n_points': 2048,
    # сглаживание=0 найдём, но нулей много — позже уточним по расположению
    'smoothing': 0,
}

offs = find_le_uint32_offsets(hdr, want)
print("Найдено смещений uint32:", offs)
