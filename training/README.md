# TACO6 YOLOv6 Training

This folder prepares and trains a YOLOv6 nano detector for the six eco-island
bins:

```text
0 plastic
1 metal
2 paper
3 glass
4 organic
5 generic
```

The pipeline is intentionally split into small pieces so the dataset conversion
can be tested locally and the full training run can happen on Kaggle GPU.

## Files

- `taco6_map.yaml` maps the 60 TACO categories into the six bins. It is
  JSON-compatible YAML so the converter needs only the Python standard library.
- `prepare_taco_yolov6.py` converts TACO COCO annotations into YOLOv6
  `images/{train,val}` and `labels/{train,val}` folders.
- `kaggle_train_taco6.ipynb` is the Kaggle notebook for smoke training, full
  training, sample inference, and ONNX export.

Generated datasets, training runs, checkpoints, and ONNX files should stay out
of git.

## Local Conversion Smoke Test

Run the unit tests:

```bash
python3 -m unittest discover -s tests
```

If you have a local copy of the Kaggle dataset, prepare a small debug split:

```bash
python3 training/prepare_taco_yolov6.py \
  --source-root /path/to/tacotrashdataset \
  --output-dir training/artifacts/taco6_yolov6_smoke \
  --limit-images 80 \
  --overwrite
```

The converter writes:

- `dataset.yaml`, suitable for `tools/train.py --data-path`
- `summary.json`, with split counts, class counts, skipped boxes, and
  oversampling details
- YOLO label files with normalized `class_id center_x center_y width height`

## Kaggle Full Run

1. Create a Kaggle notebook with GPU enabled.
2. Add the dataset `kneroma/tacotrashdataset` as notebook input.
3. Make this repository available at `/kaggle/working/GDG-hackathon-2026`
   or set the `PROJECT_ROOT` environment variable in the notebook.
4. Run `training/kaggle_train_taco6.ipynb`.

The notebook uses:

- YOLOv6 nano COCO weights from Meituan YOLOv6.
- Input shape `512x288` via `--specific-shape --height 288 --width 512`.
- `150` epochs, batch size `32`, `--fuse_ab`, validation every `10` epochs.
- Seed `2026`, patched into the Kaggle clone of YOLOv6 because upstream
  training currently hardcodes the seed.
- Moderate rare-bin oversampling: each rare class is raised toward 50% of the
  majority bin image count, with at most 3 appearances per source image.

## Artifacts To Keep

Download these from Kaggle output storage rather than committing them:

- `taco6_yolov6/summary.json`
- `yolov6_runs/taco6_yolov6n_512x288*/args.yaml`
- `yolov6_runs/taco6_yolov6n_512x288*/weights/best_ckpt.pt`
- `yolov6_runs/taco6_yolov6n_512x288*/weights/best_ckpt.onnx`
- `yolov6_runs/inference/taco6_samples*`

OAK4/RVC4 conversion is a later step; this stage stops at trained/evaluated
PyTorch and ONNX artifacts.

## Local Full Run Without Kaggle

The local runner can use either a local TACO directory or the official TACO
GitHub/Flickr source. Full CPU training on a Mac can be very slow; run the smoke
step first.

Using an existing local TACO dataset:

```bash
python3 training/local_train_taco6.py \
  --source-root /path/to/taco/data \
  --device cpu \
  --smoke-only
```

Downloading official TACO locally instead of using Kaggle:

```bash
python3 training/local_train_taco6.py \
  --download-official-taco \
  --device cpu \
  --smoke-only
```

After the smoke run succeeds, remove `--smoke-only` to launch the full 150-epoch
run:

```bash
python3 training/local_train_taco6.py \
  --download-official-taco \
  --device cpu
```

Local outputs are written under `training/artifacts/local/` by default.
