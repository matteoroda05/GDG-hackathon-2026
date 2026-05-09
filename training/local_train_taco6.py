#!/usr/bin/env python3
"""Local end-to-end runner for TACO6 YOLOv6 training.

This intentionally avoids Kaggle. It can use an existing local TACO dataset or
download the official TACO data from GitHub/Flickr.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = REPO_ROOT / "training" / "artifacts" / "local"
YOLOV6_REPO = "https://github.com/meituan/YOLOv6.git"
TACO_REPO = "https://github.com/pedropro/TACO.git"
YOLOV6N_WEIGHTS_URL = (
    "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt"
)


def run(cmd: list[str | Path], cwd: Path | None = None) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"$ {printable}", flush=True)
    subprocess.run([str(part) for part in cmd], cwd=cwd, check=True)


def ensure_venv(venv_dir: Path, python: str) -> Path:
    python_bin = venv_dir / "bin" / "python"
    if not python_bin.exists():
        run([python, "-m", "venv", venv_dir])
    run([python_bin, "-m", "pip", "install", "--upgrade", "pip", "wheel"])
    return python_bin


def ensure_yolov6(yolov6_dir: Path) -> None:
    if not yolov6_dir.exists():
        run(["git", "clone", "--depth", "1", YOLOV6_REPO, yolov6_dir])


def patch_yolov6_seed(yolov6_dir: Path, seed: int) -> None:
    train_py = yolov6_dir / "tools" / "train.py"
    text = train_py.read_text(encoding="utf-8")
    old = "set_random_seed(1+args.rank, deterministic=(args.rank == -1))"
    new = f"set_random_seed({seed} + max(args.rank, 0), deterministic=(args.rank == -1))"
    if old in text:
        train_py.write_text(text.replace(old, new), encoding="utf-8")
    elif new not in text:
        raise RuntimeError("YOLOv6 train.py seed line changed; patch it manually.")


def patch_yolov6_torch_load(yolov6_dir: Path) -> None:
    checkpoint_py = yolov6_dir / "yolov6" / "utils" / "checkpoint.py"
    text = checkpoint_py.read_text(encoding="utf-8")
    replacements = {
        "torch.load(weights, map_location=map_location)": (
            "torch.load(weights, map_location=map_location, weights_only=False)"
        ),
        "torch.load(ckpt_path, map_location=map_location)": (
            "torch.load(ckpt_path, map_location=map_location, weights_only=False)"
        ),
        "torch.load(ckpt_path, map_location=torch.device('cpu'))": (
            "torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)"
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    checkpoint_py.write_text(text, encoding="utf-8")


def ensure_weights(yolov6_dir: Path) -> Path:
    weights_dir = yolov6_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "yolov6n.pt"
    if not weights_path.exists():
        print(f"Downloading {YOLOV6N_WEIGHTS_URL}", flush=True)
        urllib.request.urlretrieve(YOLOV6N_WEIGHTS_URL, weights_path)
    return weights_path


def ensure_official_taco(taco_dir: Path, python_bin: Path) -> Path:
    if not taco_dir.exists():
        run(["git", "clone", "--depth", "1", TACO_REPO, taco_dir])
    annotations = taco_dir / "data" / "annotations.json"
    if not annotations.exists():
        raise FileNotFoundError(f"Missing official TACO annotations: {annotations}")
    run([python_bin, "-m", "pip", "install", "Pillow", "requests"])
    run([python_bin, taco_dir / "download.py", "--dataset_path", annotations], cwd=taco_dir)
    return taco_dir / "data"


def install_training_dependencies(python_bin: Path, yolov6_dir: Path) -> None:
    run([python_bin, "-m", "pip", "install", "-r", yolov6_dir / "requirements.txt"])
    run([python_bin, "-m", "pip", "install", "psutil"])
    run([python_bin, "-m", "pip", "install", "onnx", "onnxsim"])


def prepare_taco6(
    python_bin: Path,
    source_root: Path,
    output_dir: Path,
    seed: int,
    limit_images: int | None,
) -> None:
    cmd: list[str | Path] = [
        python_bin,
        REPO_ROOT / "training" / "prepare_taco_yolov6.py",
        "--source-root",
        source_root,
        "--output-dir",
        output_dir,
        "--seed",
        str(seed),
        "--overwrite",
    ]
    if limit_images is not None:
        cmd.extend(["--limit-images", str(limit_images)])
    run(cmd)


def train(
    python_bin: Path,
    yolov6_dir: Path,
    dataset_dir: Path,
    runs_dir: Path,
    name: str,
    epochs: int,
    batch_size: int,
    device: str,
    eval_final_only: bool,
) -> None:
    cmd: list[str | Path] = [
        python_bin,
        "tools/train.py",
        "--data-path",
        dataset_dir / "dataset.yaml",
        "--conf-file",
        "configs/yolov6n_finetune.py",
        "--output-dir",
        runs_dir,
        "--name",
        name,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--specific-shape",
        "--height",
        "288",
        "--width",
        "512",
        "--fuse_ab",
        "--device",
        device,
    ]
    if eval_final_only:
        cmd.append("--eval-final-only")
    else:
        cmd.extend(["--eval-interval", "10"])
    run(cmd, cwd=yolov6_dir)


def newest_run(runs_dir: Path, prefix: str) -> Path:
    runs = sorted(runs_dir.glob(f"{prefix}*"), key=lambda path: path.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No run found under {runs_dir} matching {prefix}*")
    return runs[-1]


def export_and_sample(
    python_bin: Path,
    yolov6_dir: Path,
    dataset_dir: Path,
    runs_dir: Path,
    run_prefix: str,
    device: str,
) -> None:
    run_dir = newest_run(runs_dir, run_prefix)
    best_weights = run_dir / "weights" / "best_ckpt.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {best_weights}")

    run(
        [
            python_bin,
            "tools/infer.py",
            "--weights",
            best_weights,
            "--source",
            dataset_dir / "images" / "val",
            "--yaml",
            dataset_dir / "dataset.yaml",
            "--img-size",
            "288",
            "512",
            "--conf-thres",
            "0.25",
            "--project",
            runs_dir / "inference",
            "--name",
            "taco6_samples",
            "--device",
            device,
        ],
        cwd=yolov6_dir,
    )
    run(
        [
            python_bin,
            "deploy/ONNX/export_onnx.py",
            "--weights",
            best_weights,
            "--img-size",
            "288",
            "512",
            "--batch-size",
            "1",
            "--simplify",
            "--device",
            device,
        ],
        cwd=yolov6_dir,
    )

    print(f"Best weights: {best_weights}")
    print(f"ONNX: {best_weights.with_suffix('.onnx')}")
    print(f"Dataset summary: {dataset_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TACO6 YOLOv6 training locally.")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--source-root", default=None, help="Existing local TACO data root.")
    parser.add_argument(
        "--download-official-taco",
        action="store_true",
        help="Clone official TACO and download images from Flickr instead of using Kaggle.",
    )
    parser.add_argument("--python", default="python3.12")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu", help="YOLOv6 device, e.g. cpu or 0.")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--full-epochs", type=int, default=150)
    parser.add_argument("--full-batch-size", type=int, default=32)
    parser.add_argument("--smoke-limit-images", type=int, default=120)
    parser.add_argument("--smoke-batch-size", type=int, default=16)
    parser.add_argument("--clean", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work_dir = Path(args.work_dir).expanduser().resolve()
    if args.clean and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    venv_dir = work_dir / ".venv"
    yolov6_dir = work_dir / "YOLOv6"
    runs_dir = work_dir / "yolov6_runs"
    smoke_dataset = work_dir / "taco6_yolov6_smoke"
    full_dataset = work_dir / "taco6_yolov6"

    python_bin = ensure_venv(venv_dir, args.python)
    ensure_yolov6(yolov6_dir)
    install_training_dependencies(python_bin, yolov6_dir)
    patch_yolov6_seed(yolov6_dir, args.seed)
    patch_yolov6_torch_load(yolov6_dir)
    ensure_weights(yolov6_dir)

    if args.source_root:
        source_root = Path(args.source_root).expanduser().resolve()
    elif args.download_official_taco:
        source_root = ensure_official_taco(work_dir / "TACO", python_bin)
    else:
        raise SystemExit(
            "Pass --source-root /path/to/taco/data or --download-official-taco."
        )

    if not args.skip_smoke:
        prepare_taco6(
            python_bin,
            source_root,
            smoke_dataset,
            seed=args.seed,
            limit_images=args.smoke_limit_images,
        )
        train(
            python_bin,
            yolov6_dir,
            smoke_dataset,
            runs_dir,
            "taco6_smoke_yolov6n_512x288",
            epochs=1,
            batch_size=args.smoke_batch_size,
            device=args.device,
            eval_final_only=True,
        )

    if args.smoke_only:
        return 0

    prepare_taco6(
        python_bin,
        source_root,
        full_dataset,
        seed=args.seed,
        limit_images=None,
    )
    train(
        python_bin,
        yolov6_dir,
        full_dataset,
        runs_dir,
        "taco6_yolov6n_512x288",
        epochs=args.full_epochs,
        batch_size=args.full_batch_size,
        device=args.device,
        eval_final_only=False,
    )
    export_and_sample(
        python_bin,
        yolov6_dir,
        full_dataset,
        runs_dir,
        "taco6_yolov6n_512x288",
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
