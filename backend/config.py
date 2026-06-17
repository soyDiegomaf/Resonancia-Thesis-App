from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

MODALITIES = ["t1", "t1ce", "t2", "flair"]

DEVICE = "cpu"

BASE_CHANNELS = 16
DROPOUT = 0.20

AVAILABLE_MODELS = {
    "segresnet": {
        "display_name": "SegResNet",
        "architecture": "MONAI SegResNet",
        "checkpoint": "segresnet_seed42.pt",
        "input_channels": 4,
        "output_channels": 3,
        "missing_modality_strategy": "Zero-filled missing channels",
    },
    "early_fusion_unet": {
        "display_name": "Early Fusion U-Net",
        "architecture": "Custom 3D U-Net baseline",
        "checkpoint": "early_fusion_unet_seed42.pt",
        "input_channels": 4,
        "output_channels": 3,
        "missing_modality_strategy": "Zero-filled missing channels",
    },
    "gated_fusion_unet": {
        "display_name": "Gated Fusion U-Net",
        "architecture": "Custom modality-gated 3D U-Net",
        "checkpoint": "gated_fusion_unet_seed42.pt",
        "input_channels": 4,
        "output_channels": 3,
        "missing_modality_strategy": "Learned modality gating using availability vector",
    },
    "hemis_fusion_unet": {
        "display_name": "HeMIS Fusion U-Net",
        "architecture": "HeMIS-inspired 3D fusion U-Net",
        "checkpoint": "hemis_fusion_unet_seed42.pt",
        "input_channels": 4,
        "output_channels": 3,
        "missing_modality_strategy": "Mean/variance fusion using available modalities",
    },
    "attention_fusion_unet": {
        "display_name": "Attention Fusion U-Net",
        "architecture": "Custom attention-based 3D fusion U-Net",
        "checkpoint": "attention_fusion_unet_seed42.pt",
        "input_channels": 4,
        "output_channels": 3,
        "missing_modality_strategy": "Cross-modality attention using availability vector",
    },
}

APP_DISCLAIMER = (
    "This application is a research prototype for brain tumor segmentation "
    "and is not intended for clinical diagnosis or treatment planning."
)