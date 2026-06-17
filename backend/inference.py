from typing import Dict, Any, Tuple
import numpy as np
import torch

from model import build_model, load_checkpoint
from config import CHECKPOINT_DIR, DEVICE, AVAILABLE_MODELS

_MODEL_CACHE = {}


def get_model(model_name: str):
    model_info = AVAILABLE_MODELS[model_name]
    checkpoint_path = CHECKPOINT_DIR / model_info["checkpoint"]

    cache_key = (model_name, str(checkpoint_path))

    if cache_key not in _MODEL_CACHE:
        model = build_model(model_name).to(DEVICE)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = load_checkpoint(model, checkpoint_path, DEVICE)
        model.eval()

        _MODEL_CACHE[cache_key] = model

    return _MODEL_CACHE[cache_key]


def pad_to_multiple(
    x: np.ndarray,
    multiple: int = 16,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    _, h, w, d = x.shape

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_d = (multiple - d % multiple) % multiple

    x_padded = np.pad(
        x,
        pad_width=((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)),
        mode="constant",
        constant_values=0,
    )

    return x_padded.astype(np.float32), (h, w, d)


def crop_to_original(
    probs: np.ndarray,
    original_shape: Tuple[int, int, int],
) -> np.ndarray:
    h, w, d = original_shape
    return probs[:, :h, :w, :d]


def run_full_volume_inference(
    x: np.ndarray,
    available: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    model = get_model(model_name)

    x_padded, original_shape = pad_to_multiple(x, multiple=16)

    xt = torch.from_numpy(x_padded).unsqueeze(0).to(DEVICE)
    av = torch.from_numpy(available).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        try:
            logits = model(xt, av)
        except TypeError:
            logits = model(xt)

        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    probs = crop_to_original(probs, original_shape)

    class_masks = (probs >= threshold).astype(np.uint8)

    whole_tumor_mask = (class_masks.sum(axis=0) > 0).astype(np.uint8)
    whole_tumor_prob = probs.max(axis=0)

    return {
        "probability": whole_tumor_prob.astype(np.float32),
        "mask": whole_tumor_mask.astype(np.uint8),
        "class_probabilities": probs.astype(np.float32),
        "class_masks": class_masks.astype(np.uint8),
    }