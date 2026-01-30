"""
src/METEOR_core/star_resolution.py

Branch of 15 microns
Distance between two lines of a same branch: 0.5 microns
Measure the resolution by FFT of the star pattern

metadata is a dictionnary that required the following entries:
- Physical pixel size X (µm)
- Image Size Z (number of Z planes)

"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage import filters, morphology

from . import logger
from .functions import dimension_slicer, norm_img


def _res_key(r):
    val = r["extra_measurements"]["resolution_length_um"]
    return val if val is not None else float("inf")


def crop(img) -> tuple[int, int, int, int]:
    """Return (top, bottom, left, right) indices for cropping the star."""
    if img.ndim != 2:
        raise ValueError(f"_crop_star_2d expects a 2D array, got shape {img.shape}")

    thr = filters.threshold_otsu(img)
    binary = img > thr
    binary = morphology.binary_closing(binary)
    binary = morphology.binary_opening(binary)

    coords = np.column_stack(np.where(binary))
    if coords.size == 0:
        raise RuntimeError("No foreground pixels found for cropping.")

    top_pixel = coords[np.argmin(coords[:, 0])]  # (row, col)
    bottom_pixel = coords[np.argmax(coords[:, 0])]
    left_pixel = coords[np.argmin(coords[:, 1])]
    right_pixel = coords[np.argmax(coords[:, 1])]

    return top_pixel[0], bottom_pixel[0], left_pixel[1], right_pixel[1]


def oneZ_resolution_analysis(
    img_array: np.ndarray,
    metadata: dict[str, Any],
    visu: bool = True,
    debug: bool = False,
) -> dict[str, Any]:
    if img_array.ndim != 2:
        raise ValueError(
            f"oneZ_resolution_analysis expects a 2D plane, got shape {img_array.shape}"
        )

    pxl_size_xy = float(metadata["Physical pixel size X"])  # assumed µm

    # normalize image
    img = norm_img(img_array)

    # crop and center the image
    top, bottom, left, right = crop(img)
    star = img[top:bottom, left:right]
    Ny, Nx = star.shape

    # Debug: basic geometry info
    if debug:
        print("=== Star resolution debug ===")
        print(f"Original img shape: {img.shape}")
        print(f"Cropped star shape: {star.shape}")
        print(
            f"Crop indices (top, bottom, left, right): {top}, {bottom}, {left}, {right}"
        )
        print(f"Pixel size XY (µm): {pxl_size_xy}")

    # 2D FFT
    F = np.fft.fft2(star)
    F_shift = np.fft.fftshift(F)
    amp = np.abs(F_shift)

    # Spatial frequency coordinates (cycles per µm)
    pixel_size = pxl_size_xy
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=pixel_size))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=pixel_size))
    FX, FY = np.meshgrid(fx, fy)
    FR = np.sqrt(FX**2 + FY**2)

    # Radial profile of FFT amplitude
    nbins = 100
    freq_bins = np.linspace(0, FR.max(), nbins + 1)
    radial_mean = np.zeros(nbins, dtype=float)

    for i in range(nbins):
        mask = (freq_bins[i] <= FR) & (freq_bins[i + 1] > FR)
        if np.any(mask):
            radial_mean[i] = amp[mask].mean()
        else:
            radial_mean[i] = 0.0

    # Smooth & normalize
    radial_mean = gaussian_filter1d(radial_mean, sigma=2)
    if radial_mean[0] == 0:
        raise RuntimeError(
            "Radial profile at zero frequency is zero; cannot normalize."
        )
    radial_mean /= radial_mean[0]

    center_freqs = (freq_bins[:-1] + freq_bins[1:]) / 2

    # Noise floor from high-frequency tail (last 20%)
    noise_start = int(0.8 * len(radial_mean))
    tail = radial_mean[noise_start:]
    noise_floor = float(np.mean(tail))
    noise_std = float(np.std(tail))

    # Threshold = noise_floor + 3·σ
    snr_threshold = noise_floor + 3 * noise_std
    above_snr = np.where(radial_mean > snr_threshold)[0]

    if above_snr.size > 0:
        last_snr_idx = int(above_snr[-1])
        resolution_cyc_per_um = float(center_freqs[last_snr_idx])
        resolution_length_um = float(1.0 / (2.0 * resolution_cyc_per_um))
        snr_at_cutoff = float(radial_mean[last_snr_idx] / noise_floor)
    else:
        resolution_cyc_per_um = None
        resolution_length_um = None
        snr_at_cutoff = None

    if debug:
        print(
            f"Radial profile: min={radial_mean.min():.4g}, max={radial_mean.max():.4g}"
        )
        print(f"Max radial frequency (cycles/µm): {FR.max():.3f}")
        print(f"Noise floor (normalized): {noise_floor:.4g}")
        print(f"Noise std (normalized): {noise_std:.4g}")
        print(f"SNR threshold (noise + 3σ): {snr_threshold:.4g}")
        if resolution_length_um is not None:
            print(f"Cutoff index: {last_snr_idx}")
            print(f"Cutoff freq: {resolution_cyc_per_um:.3f} cycles/µm")
            print(f"Half-pitch resolution: {resolution_length_um:.3f} µm")
            print(f"SNR at cutoff (amp / noise_floor): {snr_at_cutoff:.2f}")
        else:
            print("No frequency bin above SNR threshold – resolution not determined.")

    # Plotting
    fig = None
    if visu:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(star, cmap="gray")
        ax1.set_title("Star Pattern Image")
        ax1.axis("off")

        ax2.imshow(np.log1p(amp), cmap="inferno")
        ax2.set_title("Log FFT Amplitude")
        ax2.axis("off")

        ax3.plot(center_freqs, radial_mean)
        ax3.axhline(y=snr_threshold, color="red", linestyle="--", label="Noise + 3σ")
        if resolution_cyc_per_um is not None:
            ax3.axvline(
                x=resolution_cyc_per_um, color="black", linestyle="--", label="Cutoff"
            )
        ax3.set_xlabel("Spatial frequency (cycles/µm)")
        ax3.set_ylabel("Normalized FFT amplitude")
        ax3.set_title("Radial frequency response of star pattern FFT")
        ax3.set_yscale("log")
        ax3.legend()

        if resolution_length_um is not None:
            stats_text = (
                f"Noise floor: {noise_floor:.4f}    "
                f"SNR at cutoff: {snr_at_cutoff:.2f}    "
                f"Threshold (noise + 3σ): {snr_threshold:.4f}\n"
                f"Cutoff: {resolution_cyc_per_um:.3f} cycle/µm    "
                f"Half-pitch: {resolution_length_um:.3f} µm\n"
            )
        else:
            stats_text = (
                f"Noise floor: {noise_floor:.4f}\n"
                f"Threshold (noise + 3σ): {snr_threshold:.4f}\n"
                "Cutoff: not found"
            )

        fig.subplots_adjust(bottom=0.1, left=0.05, right=0.95)
        fig.text(
            0.5,
            0.08,
            stats_text,
            ha="center",
            va="top",
            fontsize=10,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.5},
        )

        plt.tight_layout()
        if __name__ == "__main__":
            plt.show()

    return {
        "figure": fig if visu else None,
        "figure name": "star_resolution",
        "status": "success" if resolution_length_um is not None else "error",
        "measurements": [
            {
                "category": "Star resolution",
                "name": "Half-pitch resolution",
                "value": resolution_length_um,
                "unit": "µm",
            },
        ],
        "extra_measurements": {
            "resolution_length_um": resolution_length_um,
            "resolution_cyc_per_um": resolution_cyc_per_um,
            "noise_floor": noise_floor,
            "noise_std": noise_std,
            "snr_threshold": snr_threshold,
            "snr_at_cutoff": snr_at_cutoff,
            "radial_freqs": center_freqs,
            "radial_profile": radial_mean,
        },
    }


def star_resolution_analysis(
    img_array, metadata, visu: bool = True, debug: bool = False, dim: str = "ZYX"
) -> dict[str, Any]:
    """
    Compute resolution from a star target via FFT radial profile.
    If a Z-stack is present, runs per-Z and returns the best (smallest) half-pitch.

    Parameters:
    - img_array should be a numpy array with shape (Z, Y, X) or (Y, X) for single Z.
    - metadata is a dictionnary that required the following entries:
        - Physical pixel size X (µm)
        - Image Size Z (number of Z planes)
    """
    if (
        int(metadata.get("Image Size T", 0)) > 1
    ):  # in case of multi time points, take the first one
        img_array = img_array[:1, ...]
        if debug:
            logger.debug(
                f"[calculate_field_uniformity]: taking first time point for analysis: shape now {img_array.shape}"
            )
    if (
        int(metadata.get("Image Size C", 0)) > 1
    ):  # in case of multi channels, take the first one
        img_array = img_array[:, :1, ...]
        if debug:
            logger.debug(
                f"[calculate_field_uniformity]: taking first channel for analysis: shape now {img_array.shape}"
            )

    img_array = dimension_slicer(img_array, metadata, dim)

    zplane = int(metadata.get("Image Size Z", 1))

    if zplane == 1:
        img_array = img_array[0]  # extract the 2D plane
        return oneZ_resolution_analysis(img_array, metadata, visu=visu, debug=debug)
    else:
        results_list = []
        for z in range(zplane):
            plane = img_array[z]
            try:
                # No figure for screening: visu=False
                res_z = oneZ_resolution_analysis(
                    plane, metadata, visu=False, debug=debug
                )
                res_z["extra_measurements"]["z_index"] = z
            except (
                ValueError,
                RuntimeError,
            ):  # failed, still record it even if not strctly needed
                res_z = {
                    "figure": None,
                    "extra_measurements": {
                        "resolution_length_um": None,
                        "z_index": z,
                    },
                }
            results_list.append(res_z)

        results_list.sort(key=_res_key)
        best = results_list[0]
        best_z = best["extra_measurements"].get("z_index", 0)

        # Recompute best Z with figure
        if visu:
            plane = img_array[best_z]
            best = oneZ_resolution_analysis(plane, metadata, visu=True, debug=debug)
            best["extra_measurements"]["z_index"] = best_z

        return best  # return best resolution
