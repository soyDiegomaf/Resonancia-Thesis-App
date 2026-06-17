import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def normalize_for_display(slice_2d: np.ndarray) -> np.ndarray:
    arr = slice_2d.astype(np.float32)

    low, high = np.percentile(arr, [1, 99])

    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)

    arr = (arr - low) / (high - low)
    arr = np.clip(arr, 0, 1)

    arr[arr < 0.005] = 0

    return arr


def fig_to_base64(fig) -> str:
    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=140,
        facecolor="black",
    )

    plt.close(fig)

    buffer.seek(0)

    encoded = base64.b64encode(buffer.read()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def make_mri_png(mri_slice: np.ndarray) -> str:
    img = normalize_for_display(mri_slice)

    fig, ax = plt.subplots(
        figsize=(4, 4),
        facecolor="black"
    )

    ax.set_facecolor("black")

    ax.imshow(
        np.rot90(img),
        cmap="bone",
        interpolation="bilinear"
    )

    ax.axis("off")

    return fig_to_base64(fig)


def make_mask_png(mask_slice: np.ndarray) -> str:
    fig, ax = plt.subplots(
        figsize=(4, 4),
        facecolor="black"
    )

    ax.set_facecolor("black")

    ax.imshow(
        np.rot90(mask_slice),
        cmap="magma",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax.axis("off")

    return fig_to_base64(fig)


def make_probability_png(prob_slice: np.ndarray) -> str:
    fig, ax = plt.subplots(
        figsize=(4, 4),
        facecolor="black"
    )

    ax.set_facecolor("black")

    ax.imshow(
        np.rot90(prob_slice),
        cmap="magma",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax.axis("off")

    return fig_to_base64(fig)


def make_overlay_png(mri_slice: np.ndarray, mask_slice: np.ndarray) -> str:
    img = normalize_for_display(mri_slice)

    overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)

    overlay[..., 0] = 0.62
    overlay[..., 1] = 0.30
    overlay[..., 2] = 0.87
    overlay[..., 3] = (mask_slice > 0).astype(np.float32) * 0.55

    fig, ax = plt.subplots(
        figsize=(4, 4),
        facecolor="black"
    )

    ax.set_facecolor("black")

    ax.imshow(
        np.rot90(img),
        cmap="bone",
        interpolation="bilinear"
    )

    ax.imshow(
        np.rot90(overlay),
        interpolation="bilinear"
    )

    ax.axis("off")

    return fig_to_base64(fig)