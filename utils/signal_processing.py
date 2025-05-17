import numpy as np
from scipy.signal import medfilt


def movmedian(data, window_size):
    """
    Apply a moving median filter to a 1D array with custom handling of edges.

    Parameters:
    - data: np.ndarray
        A 1D NumPy array of numerical data to filter.
    - window_size: int
        The size of the moving window used to compute the median.
        If an even number is provided, it will be increased by 1 to make it odd,
        as required by `scipy.signal.medfilt`.

    Returns:
    - filtered: np.ndarray
        The filtered array, with the central values smoothed by a moving median filter,
        and the edges filled with the median of their respective border windows.
    """
    # Ensure an odd window size for medfilt.
    if window_size % 2 == 0:
        window_size += 1

    # Apply median filter
    filtered = medfilt(data, kernel_size=window_size)

    # Handle the start and end edges separately to avoid edge artifacts
    median_start = np.median(data[:window_size // 2])
    median_end = np.median(data[-window_size // 2:])
    filtered[:window_size // 2] = median_start
    filtered[-window_size // 2:] = median_end

    return filtered


def downsample_array(array, factor, axis):
    """
    Downsample a NumPy array along a specified axis by a given factor.

    Parameters:
    - array: np.ndarray, input data of any shape.
    - factor: int, downsampling factor (keep every `factor`-th element). For instance factor = 2 halfs the sampling rate.
    - axis: int, axis along which to downsample.

    Returns:
    - downsampled_array: np.ndarray, array downsampled along the specified axis.
    """
    # Build a slicing object: slice(None) for all axes, except step-slice on the target axis
    slicer = [slice(None)] * array.ndim
    slicer[axis] = slice(None, None, factor)
    return array[tuple(slicer)]
