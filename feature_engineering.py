import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
import time
from scipy.stats import skew, kurtosis

## New feature engineering (can incorporate into extract_all_features)
 
## Time-domain statistical features
def compute_rms(signal_data):
    signal_data = np.asarray(signal_data)
    return np.sqrt(np.mean(signal_data ** 2))

def compute_skewness(signal_data):
    return skew(signal_data)

def compute_kurtosis(signal_data):
    return kurtosis(signal_data)

def compute_crest_factor(signal_data):
    rms = compute_rms(signal_data)
    if rms == 0:
        return 0
    return np.max(np.abs(signal_data)) / rms

## Improved peak based features
def compute_peak_rate(num_peaks, duration_seconds):
    if duration_seconds == 0:
        return 0
    return num_peaks / duration_seconds

def peak_magnitude_stats(peak_magnitudes):
    if len(peak_magnitudes) == 0:
        return 0, 0, 0
    return (
        skew(peak_magnitudes),
        kurtosis(peak_magnitudes),
        np.percentile(peak_magnitudes, 75) - np.percentile(peak_magnitudes, 25)
    )

def compute_burstiness(inter_peak_times):
    if len(inter_peak_times) < 2:
        return 0
    return np.std(inter_peak_times) / np.mean(inter_peak_times)

def compute_band_powers(signal_data, fs=1000, bands=None):
    if bands is None:
        bands = {
            "low": (0, 200),
            "mid": (200, 800),
            "high": (800, fs // 2)
        }

    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
    total_power = np.sum(psd)

    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        band_power = np.sum(psd[mask])
        band_powers[f"{band}_band_power"] = band_power / total_power if total_power > 0 else 0

    return band_powers


def compute_band_ratios(band_powers):
    low = band_powers.get("low_band_power", 0)
    high = band_powers.get("high_band_power", 0)
    return {
        "high_to_low_ratio": high / low if low > 0 else 0
    }

def compute_spectrogram_features(signal_data, fs=1000):
    freqs, times, Sxx = signal.spectrogram(signal_data, fs=fs)
    Sxx_norm = Sxx / (np.sum(Sxx, axis=0, keepdims=True) + 1e-12)

    entropy_t = -np.sum(Sxx_norm * np.log2(Sxx_norm + 1e-12), axis=0)
    centroid_t = np.sum(freqs[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-12)

    return {
        "mean_spectral_entropy_time": np.mean(entropy_t),
        "var_spectral_centroid_time": np.var(centroid_t)
    }


def compute_spectral_flux(signal_data, fs=1000):
    freqs, times, Sxx = signal.spectrogram(signal_data, fs=fs)
    flux = np.sum(np.diff(Sxx, axis=1) ** 2, axis=0)
    return np.mean(flux)


import pywt

def compute_wavelet_energy(signal_data, wavelet="morl", scales=None):
    if scales is None:
        scales = np.arange(1, 64)

    coeffs, _ = pywt.cwt(signal_data, scales, wavelet)
    energy = np.sum(coeffs ** 2, axis=1)

    return {
        "wavelet_energy_low_scale": np.mean(energy[:len(energy)//3]),
        "wavelet_energy_mid_scale": np.mean(energy[len(energy)//3:2*len(energy)//3]),
        "wavelet_energy_high_scale": np.mean(energy[2*len(energy)//3:])
    }


## Regime transition/change point features
def compute_rms_change_points(signal_data, window_size=1000, threshold=2.0):
    rms_values = []
    for i in range(0, len(signal_data) - window_size, window_size):
        rms_values.append(compute_rms(signal_data[i:i+window_size]))

    rms_values = np.array(rms_values)
    diffs = np.abs(np.diff(rms_values))
    return np.sum(diffs > threshold * np.std(rms_values))

def compute_regime_dominance(candidates):
    if not candidates.candidate_lst:
        return 0
    hit_counts = [len(c.hit_indices) for c in candidates.candidate_lst]
    return max(hit_counts) / sum(hit_counts)
