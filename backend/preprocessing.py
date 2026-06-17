from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import nibabel as nib

from config import MODALITIES


def load_nifti(path: Path):
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32), img.affine


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    mask = volume > 0

    if mask.sum() == 0:
        return volume.astype(np.float32)

    mean = volume[mask].mean()
    std = volume[mask].std()

    if std < 1e-6:
        std = 1.0

    out = (volume - mean) / std
    out[~mask] = 0

    return out.astype(np.float32)


def build_input_volume(
    files: Dict[str, Optional[Path]]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, bool]]:
    loaded = {}
    reference_affine = None
    reference_shape = None

    for modality in MODALITIES:
        path = files.get(modality)

        if path is not None:
            vol, affine = load_nifti(path)
            loaded[modality] = normalize_volume(vol)

            if reference_shape is None:
                reference_shape = vol.shape

            if reference_affine is None:
                reference_affine = affine

    if reference_shape is None:
        raise ValueError("At least one MRI modality must be uploaded.")

    channels = []
    available = []
    availability_map = {}

    for modality in MODALITIES:
        if modality in loaded:
            vol = loaded[modality]

            if vol.shape != reference_shape:
                raise ValueError(
                    f"{modality.upper()} shape does not match the reference shape."
                )

            channels.append(vol)
            available.append(1.0)
            availability_map[modality] = True

        else:
            channels.append(np.zeros(reference_shape, dtype=np.float32))
            available.append(0.0)
            availability_map[modality] = False

    x = np.stack(channels, axis=0).astype(np.float32)
    available = np.array(available, dtype=np.float32)

    return x, available, availability_map, reference_affine