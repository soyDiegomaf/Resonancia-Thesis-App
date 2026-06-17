from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import Optional
from uuid import uuid4
import shutil
import os

import nibabel as nib
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from preprocessing import build_input_volume, load_nifti
from inference import run_full_volume_inference
from metrics import dice_score
from visualization import make_mri_png, make_mask_png, make_probability_png, make_overlay_png
from config import APP_DISCLAIMER, AVAILABLE_MODELS

app = FastAPI(title="Resonancia API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS = {}


def save_upload(upload: Optional[UploadFile], folder: Path, name: str) -> Optional[Path]:
    if upload is None:
        return None

    suffix = ".nii.gz" if upload.filename.endswith(".nii.gz") else Path(upload.filename).suffix
    path = folder / f"{name}{suffix}"

    with path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)

    return path


def render_slice(session, slice_index: int):
    x = session["x"]
    mask = session["mask"]
    probs = session["probability"]
    availability = session["availability"]

    depth = mask.shape[2]
    slice_index = int(max(0, min(slice_index, depth - 1)))

    display_channel = 3 if availability["flair"] else int(np.argmax(session["available"]))

    mri_slice = x[display_channel, :, :, slice_index]
    mask_slice = mask[:, :, slice_index]
    prob_slice = probs[:, :, slice_index]

    return {
        "slice_index": slice_index,
        "images": {
            "mri": make_mri_png(mri_slice),
            "mask": make_mask_png(mask_slice),
            "overlay": make_overlay_png(mri_slice, mask_slice),
            "probability": make_probability_png(prob_slice),
        },
    }


@app.get("/")
def health_check():
    return {
        "app": "Resonancia",
        "status": "running",
        "disclaimer": APP_DISCLAIMER,
    }


@app.get("/models")
def list_models():
    return AVAILABLE_MODELS


@app.post("/segment")
def segment(
    t1: Optional[UploadFile] = File(None),
    t1ce: Optional[UploadFile] = File(None),
    t2: Optional[UploadFile] = File(None),
    flair: Optional[UploadFile] = File(None),
    ground_truth: Optional[UploadFile] = File(None),
    slice_index: Optional[int] = Form(None),
    model_name: str = Form("segresnet"),
):
    try:
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")

        with TemporaryDirectory() as tmp:
            folder = Path(tmp)

            files = {
                "t1": save_upload(t1, folder, "t1"),
                "t1ce": save_upload(t1ce, folder, "t1ce"),
                "t2": save_upload(t2, folder, "t2"),
                "flair": save_upload(flair, folder, "flair"),
            }

            gt_path = save_upload(ground_truth, folder, "ground_truth")

            x, available, availability_map, affine = build_input_volume(files)

            result = run_full_volume_inference(x, available, model_name=model_name)

            mask = result["mask"]
            probs = result["probability"]

            depth = mask.shape[2]

            if slice_index is None:
                slice_index = depth // 2

            dice = None

            if gt_path is not None:
                gt, _ = load_nifti(gt_path)

                if gt.shape != mask.shape:
                    raise ValueError("Ground truth shape does not match prediction shape.")

                dice = dice_score(mask, gt > 0)

            tumor_voxels = int(mask.sum())
            tumor_volume_cm3 = round(tumor_voxels / 1000.0, 2)

            mean_tumor_probability = None
            if tumor_voxels > 0:
                mean_tumor_probability = float(probs[mask > 0].mean())

            max_tumor_probability = float(probs.max())

            session_id = str(uuid4())

            SESSIONS[session_id] = {
                "x": x,
                "available": available,
                "availability": availability_map,
                "mask": mask,
                "probability": probs,
                "affine": affine,
                "model_name": model_name,
            }

            slice_payload = render_slice(SESSIONS[session_id], slice_index)

            return {
                "session_id": session_id,
                "model": AVAILABLE_MODELS[model_name],
                "availability": availability_map,
                "available_vector": available.tolist(),
                "slice_index": slice_payload["slice_index"],
                "depth": depth,
                "tumor_voxels": tumor_voxels,
                "tumor_volume_cm3": tumor_volume_cm3,
                "mean_tumor_probability": mean_tumor_probability,
                "max_tumor_probability": max_tumor_probability,
                "dice": dice,
                "images": slice_payload["images"],
                "disclaimer": APP_DISCLAIMER,
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/slice/{session_id}/{slice_index}")
def get_slice(session_id: str, slice_index: int):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found. Please run segmentation again.")

    return render_slice(SESSIONS[session_id], slice_index)


@app.get("/download/{session_id}")
def download_mask(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found. Please run segmentation again.")

    session = SESSIONS[session_id]

    mask = session["mask"].astype(np.uint8)
    affine = session["affine"]

    img = nib.Nifti1Image(mask, affine)

    tmp = NamedTemporaryFile(delete=False, suffix=".nii.gz")
    tmp.close()

    nib.save(img, tmp.name)

    return FileResponse(
        tmp.name,
        media_type="application/gzip",
        filename="resonancia_predicted_mask.nii.gz",
        background=BackgroundTask(lambda: os.remove(tmp.name)),
    )