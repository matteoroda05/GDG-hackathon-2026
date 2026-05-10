#!/usr/bin/env python3
import json
import os
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List
import pprint

import depthai as dai

from model_config import COCO_LABELS, COCO_TO_TRASH
from utils.arguments import initialize_argparser
from webapp import DashboardServer, DashboardStore


CUSTOM_LABELS: List[str] | None = None


def _load_json_path(config_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing config.json at {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}") from exc


def _select_config_name(names: List[str], archive_path: Path) -> str:
    candidates = [name for name in names if name.endswith("config.json")]
    if not candidates:
        raise FileNotFoundError(f"config.json not found in {archive_path}")
    return min(candidates, key=len)


def _load_config_json(archive_path: Path) -> Dict[str, Any]:
    if archive_path.is_dir():
        return _load_json_path(archive_path / "config.json")

    suffix = "".join(archive_path.suffixes).lower()
    if suffix.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            config_name = _select_config_name(archive.namelist(), archive_path)
            data = archive.read(config_name)
        return json.loads(data.decode("utf-8"))

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            names = [member.name for member in archive.getmembers() if member.isfile()]
            config_name = _select_config_name(names, archive_path)
            member = archive.getmember(config_name)
            fileobj = archive.extractfile(member)
            if fileobj is None:
                raise FileNotFoundError(f"config.json not found in {archive_path}")
            data = fileobj.read()
        return json.loads(data.decode("utf-8"))

    sidecar_candidates = [archive_path.with_suffix(".json"), archive_path.parent / "config.json"]
    for candidate in sidecar_candidates:
        if candidate.exists():
            return _load_json_path(candidate)

    raise FileNotFoundError(
        "config.json not found. Expected it inside the archive or alongside the model file."
    )


def _load_custom_labels(archive_path: Path) -> List[str]:
    config = _load_config_json(archive_path)
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("config.json model not found")
    
    heads = model.get("heads")
    if isinstance(heads, list):
        if not heads:
            raise ValueError("config.json model.heads is empty")
        heads = heads[0]
    
    metadata = heads.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("config.json model.heads.metadata not found")

    classes = metadata.get("classes")
    if not isinstance(classes, list) or not all(isinstance(name, str) for name in classes):
        raise ValueError("config.json model.heads.metadata.classes must be a list of strings")

    return classes


def _detection_label(detection: Any) -> str:
    label_id = int(getattr(detection, "label", -1))
    if CUSTOM_LABELS and 0 <= label_id < len(CUSTOM_LABELS):
        return CUSTOM_LABELS[label_id]
    if 0 <= label_id < len(COCO_LABELS):
        return COCO_LABELS[label_id]
    return f"class {label_id}"


def _trash_label(label: str) -> str | None:
    if CUSTOM_LABELS is not None:
        return label
    return COCO_TO_TRASH.get(label)


def _detection_dict(detection: Any) -> Dict[str, Any]:
    xmin = float(getattr(detection, "xmin", 0.0))
    ymin = float(getattr(detection, "ymin", 0.0))
    xmax = float(getattr(detection, "xmax", xmin))
    ymax = float(getattr(detection, "ymax", ymin))
    label = _detection_label(detection)
    yolo_label = None if CUSTOM_LABELS is not None else label
    return {
        "label": label,
        "yolo_label": yolo_label,
        "trash_label": _trash_label(label),
        "label_id": int(getattr(detection, "label", -1)),
        "confidence": float(getattr(detection, "confidence", 0.0)),
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "width": xmax - xmin,
        "height": ymax - ymin,
    }


def _depth_value_from_message(message: Any) -> float | None:
    locations = getattr(message, "getSpatialLocations", lambda: [])()
    if not locations:
        return None
    return float(locations[0].spatialCoordinates.z)


def _consume_frames(frame_queue, store: DashboardStore):
    while True:
        message = frame_queue.get()
        if message is None:
            continue

        frame = message.getCvFrame()
        store.update_frame(frame)


def _consume_detections(detection_queue, store: DashboardStore):
    while True:
        message = detection_queue.get()
        if message is None:
            continue

        detections = [_detection_dict(detection) for detection in getattr(message, "detections", [])]
        store.update_detections(detections)


def _consume_depth(depth_queue, store: DashboardStore):
    while True:
        message = depth_queue.get()
        if message is None:
            continue

        store.update_depth(_depth_value_from_message(message))


def run_device(args):
    global CUSTOM_LABELS
    CUSTOM_LABELS = None

    device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    web_host = args.web_host or "0.0.0.0"
    web_port = int(args.web_port or os.environ.get("PORT", "8080"))

    preview_dimensions = (1280, 720)
    nn_dimensions = (512, 288)

    if not device.setIrLaserDotProjectorIntensity(1):
        print(
            "Failed to set IR laser projector intensity. Maybe your device does not support this feature."
        )

    pipeline = dai.Pipeline(device)
    print("Creating pipeline...")
    platform = device.getPlatform()

    if args.model_archive_path:
        archive_path = Path(args.model_archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"NN archive not found: {archive_path}")
        print(f"Loading NN archive from disk: {archive_path}")
        nn_archive = dai.NNArchive(str(archive_path))
        CUSTOM_LABELS = _load_custom_labels(archive_path)
    elif args.model_zoo_id:
        print(f"Fetching NN archive from Luxonis Model Zoo: {args.model_zoo_id}")
        model_description = dai.NNModelDescription(modelSlug=args.model_zoo_id)
        nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))
    else:
        model_description = dai.NNModelDescription.fromYamlFile(
            f"yolov6_nano_r2_coco.{platform.name}.yaml"
        )
        nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))
    camera_node = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    if platform == dai.Platform.RVC2:
        detection_network = pipeline.create(dai.node.DetectionNetwork)
        camera_node.requestOutput(nn_dimensions, dai.ImgFrame.Type.BGR888p).link(
            detection_network.input
        )
        detection_network.setNNArchive(nn_archive, numShaves=4)
    else:
        detection_network = pipeline.create(dai.node.DetectionNetwork).build(
            camera_node.requestOutput(nn_dimensions, dai.ImgFrame.Type.BGR888i),
            nn_archive,
        )

    outputToEncode = camera_node.requestOutput((1440, 1080), type=dai.ImgFrame.Type.NV12)
    preview_output = camera_node.requestOutput(
        preview_dimensions, type=dai.ImgFrame.Type.BGR888i
    )

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    left_output = left.requestOutput(nn_dimensions, type=dai.ImgFrame.Type.NV12)
    right_output = right.requestOutput(nn_dimensions, type=dai.ImgFrame.Type.NV12)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_output,
        right=right_output,
        presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
    )

    align = pipeline.create(dai.node.ImageAlign)
    stereo.depth.link(align.input)
    outputToEncode.link(align.inputAlignTo)

    spatial_calc = pipeline.create(dai.node.SpatialLocationCalculator)
    input_depth = align.outputAligned if platform == dai.Platform.RVC4 else stereo.depth
    input_depth.link(spatial_calc.inputDepth)

    config_depth = dai.SpatialLocationCalculatorConfigData()
    config_depth.roi = dai.Rect(0.2, 0.2, 0.8, 0.8)
    config_depth.depthThresholds.lowerThreshold = 100
    config_depth.depthThresholds.upperThreshold = 10000
    config_depth.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
    spatial_calc.initialConfig.addROI(config_depth)

    store = DashboardStore()
    dashboard = DashboardServer(web_host, web_port, store)

    preview_queue = preview_output.createOutputQueue()
    detection_queue = detection_network.out.createOutputQueue()
    depth_queue = spatial_calc.out.createOutputQueue()

    print(f"Web dashboard available at http://{web_host}:{web_port}")
    dashboard.start()

    pipeline.start()

    threading.Thread(target=_consume_frames, args=(preview_queue, store), daemon=True).start()
    threading.Thread(target=_consume_detections, args=(detection_queue, store), daemon=True).start()
    threading.Thread(target=_consume_depth, args=(depth_queue, store), daemon=True).start()

    print("Pipeline created.")

    try:
        while pipeline.isRunning():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopping pipeline on keyboard interrupt.")


if __name__ == "__main__":
    _, ags = initialize_argparser()
    run_device(ags)
