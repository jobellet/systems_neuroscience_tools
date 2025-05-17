import numpy as np
from scipy.signal import medfilt


def movmedian(data, window_size):
    # Ensure an odd window size for medfilt.
    if window_size % 2 == 0:
        window_size += 1
    filtered = medfilt(data, kernel_size=window_size)
    median_start = np.median(data[:window_size//2])
    median_end = np.median(data[-window_size//2:])
    filtered[:window_size//2] = median_start
    filtered[-window_size//2:] = median_end
    return filtered