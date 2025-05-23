# utils/image_processing.py
"""
image_processing.py

This module contains utility functions for processing visual stimuli.
It includes:
  - A function to simulate the magnocellular pathway (Gaussian low-pass filtering)
  - A function to compute the radial average of an FFT magnitude spectrum
  - A function to process a folder of images: apply the filter and save outputs
  - Functions to compute and plot radial frequency spectra from original and filtered images

These utilities are useful in time-resolved representational geometry analyses
of ventral stream data.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def m_pathway_filter_gaussian(image,
                              cutoff_cpd     = 0.5,   # desired cut-off in cycles/degree
                              fov_deg        = 8.0,   # image spans this many degrees
                              atten_dB_at_cut = -20):  # attenuation (dB) at cutoff
    """
    Gaussian low-pass that is attenuated by `atten_dB_at_cut`
    """
    # 1) convert to cycles per image width
    cutoff_cycles = cutoff_cpd * fov_deg

    # 2) linear gain at cutoff
    gain = 10**(atten_dB_at_cut / 20)

    # 3) solve σ so that exp(–(cutoff_cycles)^2/(2σ^2)) == gain
    sigma = cutoff_cycles / np.sqrt(-2 * np.log(gain))

    # 4) forward FFT (no fftshift)
    F = np.fft.fft2(image)

    # 5) build frequency grid (in cycles per image)
    rows, cols = image.shape
    row_freqs = np.fft.fftfreq(rows) * rows
    col_freqs = np.fft.fftfreq(cols) * cols
    U, V = np.meshgrid(col_freqs, row_freqs)
    D = np.sqrt(U**2 + V**2)

    # 6) isotropic Gaussian low-pass
    H = np.exp(-(D**2) / (2 * sigma**2))

    # 7) apply filter and inverse FFT
    Filt = F * H
    image_filtered = np.fft.ifft2(Filt).real

    return image_filtered

def radial_average_vectorized(magnitude_spectrum, visual_angle=10):
    """
    Compute the radial average of a 2D magnitude spectrum.
    
    Parameters
    ----------
    magnitude_spectrum : ndarray
        The 2D FFT magnitude spectrum.
    visual_angle : float, optional
        Visual angle (degrees) to scale the spatial frequency axis.
        
    Returns
    -------
    radial_avg : ndarray
        The average magnitude as a function of radius.
    cpd : ndarray
        The spatial frequency axis (cycles per degree), linearly spaced.
    """
    rows, cols = magnitude_spectrum.shape[-2:]
    cy, cx = rows // 2, cols // 2
    y, x = np.indices((rows, cols))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    
    radial_sum = np.bincount(r.ravel(), weights=magnitude_spectrum.ravel())
    radial_count = np.bincount(r.ravel())
    radial_count[radial_count == 0] = 1  # avoid division by zero
    
    radial_avg = radial_sum / radial_count
    cpd = np.linspace(0, (rows // 2) / visual_angle, len(radial_avg))
    return radial_avg, cpd

def process_and_filter_images(stimulus_folder, output_folder, filter_func=m_pathway_filter_gaussian):
    """
    Process images from the stimulus folder by applying the specified filter
    and saving the filtered images to the output folder.
    
    Parameters
    ----------
    stimulus_folder : str
        Path to the folder with original images.
    output_folder : str
        Path where filtered images will be saved.
    filter_func : callable, optional
        The filtering function to apply. Defaults to m_pathway_filter_gaussian.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([f for f in os.listdir(stimulus_folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for img_name in image_files:
        img_path = os.path.join(stimulus_folder, img_name)
        output_path = os.path.join(output_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        filtered_img = filter_func(img)
        cv2.imwrite(output_path, filtered_img)
    print(f"Processed {len(image_files)} images from {stimulus_folder} and saved to {output_folder}.")

def compute_radial_spectra(stimulus_folder, output_folder, visual_angle=10, epsilon=1e-8):
    """
    Compute the radial spectra (log magnitude) for both original and filtered images.
    
    Parameters
    ----------
    stimulus_folder : str
        Path to the folder with original images.
    output_folder : str
        Path to the folder with filtered images.
    visual_angle : float, optional
        Visual angle in degrees for computing the frequency axis.
    epsilon : float, optional
        Small constant added before taking the logarithm to avoid log(0).
        
    Returns
    -------
    cpd : ndarray
        The spatial frequency axis (cycles per degree).
    mean_orig : ndarray
        Mean log magnitude spectrum for original images.
    sem_orig : ndarray
        Standard error of the mean for the original images.
    mean_filt : ndarray
        Mean log magnitude spectrum for filtered images.
    sem_filt : ndarray
        Standard error of the mean for the filtered images.
    """
    image_files = sorted([f for f in os.listdir(stimulus_folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    orig_log_radial_list = []
    filt_log_radial_list = []
    cpd = None

    for img_name in image_files:
        original_path = os.path.join(stimulus_folder, img_name)
        filtered_path = os.path.join(output_folder, img_name)
        original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        filtered_img = cv2.imread(filtered_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None or filtered_img is None:
            continue
        
        original_fft = np.fft.fftshift(np.fft.fft2(original_img))
        filtered_fft = np.fft.fftshift(np.fft.fft2(filtered_img))
        original_magnitude = np.abs(original_fft)
        filtered_magnitude = np.abs(filtered_fft)
        
        original_avg, cpd = radial_average_vectorized(original_magnitude, visual_angle)
        filtered_avg, _ = radial_average_vectorized(filtered_magnitude, visual_angle)
        
        orig_log_radial_list.append(np.log(original_avg + epsilon))
        filt_log_radial_list.append(np.log(filtered_avg + epsilon))
    
    orig_log_radial = np.array(orig_log_radial_list)
    filt_log_radial = np.array(filt_log_radial_list)
    
    mean_orig = np.mean(orig_log_radial, axis=0)
    mean_filt = np.mean(filt_log_radial, axis=0)
    sem_orig = np.std(orig_log_radial, axis=0, ddof=1) / np.sqrt(orig_log_radial.shape[0])
    sem_filt = np.std(filt_log_radial, axis=0, ddof=1) / np.sqrt(filt_log_radial.shape[0])
    
    return cpd, mean_orig, sem_orig, mean_filt, sem_filt

def plot_radial_spectra(cpd, mean_orig, sem_orig, mean_filt, sem_filt, title="Mean Frequency Spectrum ± SEM"):
    """
    Plot the radial frequency spectra with SEM shaded error bars.
    
    Parameters
    ----------
    cpd : ndarray
        The spatial frequency axis (cycles per degree).
    mean_orig : ndarray
        Mean log magnitude spectrum for original images.
    sem_orig : ndarray
        Standard error of the mean for the original images.
    mean_filt : ndarray
        Mean log magnitude spectrum for filtered images.
    sem_filt : ndarray
        Standard error of the mean for the filtered images.
    title : str, optional
        Plot title.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(cpd, mean_orig, label="Original Spectrum", linestyle="dashed", color="blue")
    plt.fill_between(cpd, mean_orig - sem_orig, mean_orig + sem_orig, color="blue", alpha=0.3)
    plt.plot(cpd, mean_filt, label="Filtered Spectrum", linestyle="solid", color="red")
    plt.fill_between(cpd, mean_filt - sem_filt, mean_filt + sem_filt, color="red", alpha=0.3)
    plt.xlabel("Spatial Frequency (cpd)")
    plt.ylabel("Log Magnitude")
    plt.title(title)
    plt.legend()
    plt.show()

# Optional: You can add an __main__ block here for basic testing.
if __name__ == "__main__":
    # Example usage with a small folder of images.
    stimulus_folder = 'path/to/your/stimulus_folder'
    output_folder = 'path/to/your/output_folder'
    
    # Process and filter images.
    process_and_filter_images(stimulus_folder, output_folder)
    
    # Compute and plot radial spectra.
    cpd, mean_orig, sem_orig, mean_filt, sem_filt = compute_radial_spectra(stimulus_folder, output_folder)
    plot_radial_spectra(cpd, mean_orig, sem_orig, mean_filt, sem_filt, title="Example Frequency Spectrum")
