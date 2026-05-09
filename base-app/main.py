#!/usr/bin/env python3
import os
import threading
import time
from typing import Any, Dict, List

import cv2
import depthai as dai

from utils.arguments import initialize_argparser
from webapp import DashboardServer, DashboardStore


COCO_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def _detection_label(detection: Any) -> str:
    label_id = int(getattr(detection, "label", -1))
    if 0 <= label_id < len(COCO_LABELS):
        return COCO_LABELS[label_id]
    return f"class {label_id}"


def _detection_dict(detection: Any) -> Dict[str, Any]:
    xmin = float(getattr(detection, "xmin", 0.0))
    ymin = float(getattr(detection, "ymin", 0.0))
    xmax = float(getattr(detection, "xmax", xmin))
    ymax = float(getattr(detection, "ymax", ymin))
    return {
        "label": _detection_label(detection),
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


def _annotate_frame(frame, detections: List[Dict[str, Any]]):
    height, width = frame.shape[:2]
    for detection in detections:
        x1 = max(0, min(width - 1, int(detection["xmin"] * width)))
        y1 = max(0, min(height - 1, int(detection["ymin"] * height)))
        x2 = max(0, min(width - 1, int(detection["xmax"] * width)))
        y2 = max(0, min(height - 1, int(detection["ymax"] * height)))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (102, 227, 180), 2)
        label = f"{detection['label']} {int(detection['confidence'] * 100)}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        top = max(0, y1 - text_height - baseline - 8)
        cv2.rectangle(
            frame,
            (x1, top),
            (x1 + text_width + 10, top + text_height + baseline + 8),
            (17, 24, 39),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 5, top + text_height + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (235, 243, 255),
            2,
        )

    return frame


def _consume_frames(frame_queue, store: DashboardStore):
    while True:
        message = frame_queue.get()
        if message is None:
            continue

        frame = message.getCvFrame()
        detections = store.snapshot().detections
        annotated = _annotate_frame(frame, detections)
        success, encoded = cv2.imencode(
            ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
        )
        if success:
            store.update_frame(encoded.tobytes())


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

    spatial_calc = pipeline.create(dai.node.SpatialLocationCalculator)
    input_depth = align.outputAligned if platform == dai.Platform.RVC4 else stereo.depth
    input_depth.link(spatial_calc.inputDepth)

    config_depth = dai.SpatialLocationCalculatorConfigData()
    config_depth.roi = dai.Rect(0.4, 0.4, 0.6, 0.6)
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
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping pipeline on keyboard interrupt.")


if __name__ == "__main__":
    _, ags = initialize_argparser()
    run_device(ags)