"""Evaluate a trained Synapse checkpoint and retain per-case, per-organ results.

Uses the existing test_single_volume preprocessing, inference, and metrics.
Distances and saved NIfTI spacing follow the original unit-grid protocol;
the HDF5 inputs do not establish physical spacing in millimetres.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import h5py
import numpy as np
import torch

from config import _C, _update_config_from_file
from utils import test_single_volume


ORGANS = ["Aorta", "Gallbladder", "Kidney (L)", "Kidney (R)",
          "Liver", "Pancreas", "Spleen", "Stomach"]
FIELDS = ["case", "class_id", "organ", "dice", "dsc_percent", "hd95_grid_units"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Synapse directory or its test_vol_h5 subdirectory")
    parser.add_argument("--cfg", default="configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--case-list", type=Path, default=Path("lists/lists_Synapse/test_vol.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--head", choices=["resize", "convtr"], default="resize")
    parser.add_argument("--use-sp", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be positive")
    data_dir = args.data_root
    if (data_dir / "test_vol_h5").is_dir():
        data_dir = data_dir / "test_vol_h5"
    cases = [line.strip() for line in args.case_list.read_text().splitlines() if line.strip()]
    if not cases or len(cases) != len(set(cases)):
        parser.error("The case list must be nonempty and contain unique case IDs")
    missing = [str(data_dir / (case + ".npy.h5")) for case in cases
               if not (data_dir / (case + ".npy.h5")).is_file()]
    if missing:
        parser.error("Missing test volumes: " + ", ".join(missing))
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    architecture = dict(HEAD=args.head, USE_SP=str(args.use_sp), SP_MODE="add",
                        USE_BIM="0", USE_AG="0", BIM_DECODER="1", LCC="0")
    os.environ.update(architecture)
    # Architecture switches are read when the network module is imported.
    from networks.vision_transformer import SwinUnet

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config = _C.clone()
    _update_config_from_file(config, args.cfg)
    model = SwinUnet(config, img_size=224, num_classes=9).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    print(model.load_state_dict(state, strict=True), flush=True)
    del state
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions = args.output_dir / "predictions"
    predictions.mkdir()
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest.hexdigest(),
        "data_directory": str(data_dir.resolve()), "cases": cases,
        "architecture": architecture, "image_size": 224, "seed": 1234,
        "device": device, "threads": args.threads,
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "hd95_units": "unit-grid distances, matching the original evaluation (voxelspacing=None)",
        "nifti_spacing": "(1,1,1) placeholder matching the original export; physical spacing is not verified",
        "prediction_files": "predictions/<case>_{img,gt,pred}.nii.gz",
        "baseline_comparison": "not performed; this run evaluates one checkpoint",
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Evaluating {len(cases)} cases on {device}; results: {args.output_dir}", flush=True)

    metrics = []
    started = time.perf_counter()
    with (args.output_dir / "per_case_per_organ.csv").open("w", newline="") as all_file, \
            (args.output_dir / "aorta_per_case.csv").open("w", newline="") as aorta_file:
        all_writer = csv.DictWriter(all_file, fieldnames=FIELDS)
        aorta_writer = csv.DictWriter(aorta_file, fieldnames=FIELDS)
        all_writer.writeheader()
        aorta_writer.writeheader()
        for index, case in enumerate(cases):
            case_started = time.perf_counter()
            with h5py.File(data_dir / f"{case}.npy.h5", "r") as source:
                image = source["image"][:]
                label = source["label"][:]
            print(f"[{index + 1}/{len(cases)}] {case}: {image.shape[0]} slices; starting", flush=True)
            if image.shape != label.shape or image.ndim != 3:
                raise ValueError(f"Invalid image/label shapes in {case}")
            values = np.asarray(test_single_volume(
                torch.from_numpy(image).unsqueeze(0), torch.from_numpy(label).unsqueeze(0),
                model, classes=9, patch_size=[224, 224],
                test_save_path=str(predictions), case=case, z_spacing=1,
            ), dtype=float)
            metrics.append(values)
            for class_id, (organ, (dice, hd95)) in enumerate(zip(ORGANS, values), start=1):
                row = dict(case=case, class_id=class_id, organ=organ,
                           dice=float(dice), dsc_percent=float(dice * 100), hd95_grid_units=float(hd95))
                all_writer.writerow(row)
                if class_id == 1:
                    aorta_writer.writerow(row)
            all_file.flush()
            aorta_file.flush()
            print(f"[{index + 1}/{len(cases)}] {case}: aorta DSC={values[0, 0] * 100:.4f}% "
                  f"HD95={values[0, 1]:.6f}; mean DSC={values[:, 0].mean() * 100:.4f}% "
                  f"HD95={values[:, 1].mean():.6f}; {time.perf_counter() - case_started:.1f}s", flush=True)

    means = np.asarray(metrics).mean(axis=0)
    summary = {
        "case_count": len(cases), "mean_dsc_percent": float(means[:, 0].mean() * 100),
        "mean_hd95_grid_units": float(means[:, 1].mean()),
        "per_organ": {organ: {"dsc_percent": float(dice * 100), "hd95_grid_units": float(hd95)}
                      for organ, (dice, hd95) in zip(ORGANS, means)},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
