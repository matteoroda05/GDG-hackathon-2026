#!/usr/bin/env python3
import cv2
import depthai as dai
from depthai_nodes.node import ApplyDepthColormap
from typing import Optional
from utils.arguments import initialize_argparser

import numpy as np


class DepthDashboardNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        # We no longer need an inputFrame, just the spatial data
        self.inputSpatial = self.createInput()
        self.output = self.createOutput()

        # Dimensions for our standalone info box
        self.width = 400
        self.height = 200

    def run(self):
        while self.isRunning():
            spatialMsg = self.inputSpatial.get()

            if spatialMsg is not None:
                # Create a black background canvas
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

                # Get the average depth from the spatial locations
                locations = spatialMsg.getSpatialLocations()
                if locations:
                    avg_z = locations[0].spatialCoordinates.z

                    # Style the dashboard
                    cv2.putText(frame, "AVERAGE DEPTH", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

                    color = (0, 255, 0) if avg_z > 0 else (0, 0, 255)
                    text = f"{avg_z:.1f} mm"
                    cv2.putText(frame, text, (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

                # Create and send the frame to the visualizer
                outMsg = dai.ImgFrame()
                outMsg.setData(frame)
                outMsg.setWidth(self.width)
                outMsg.setHeight(self.height)
                outMsg.setType(dai.ImgFrame.Type.BGR888i)
                self.output.send(outMsg)

def run_device(args):
    visualizer = dai.RemoteConnection(httpPort=8082)
    device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

    NN_DIMENSIONS = (512, 288)

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
    cameraNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    if platform == dai.Platform.RVC2:
        detectionNetwork = pipeline.create(dai.node.DetectionNetwork)
        cameraNode.requestOutput(NN_DIMENSIONS, dai.ImgFrame.Type.BGR888p).link(
            detectionNetwork.input
        )
        detectionNetwork.setNNArchive(nn_archive, numShaves=4)
    else:
        detectionNetwork = pipeline.create(dai.node.DetectionNetwork).build(
            cameraNode.requestOutput(NN_DIMENSIONS, dai.ImgFrame.Type.BGR888i),
            nn_archive,
        )

    outputToEncode = cameraNode.requestOutput((1440, 1080), type=dai.ImgFrame.Type.NV12)
    h264Encoder = pipeline.create(dai.node.VideoEncoder)
    h264Encoder.setDefaultProfilePreset(
        30, dai.VideoEncoderProperties.Profile.H264_MAIN
    )
    outputToEncode.link(h264Encoder.input)

    # Add the remote connector topics
    visualizer.addTopic("Raw video", outputToEncode)
    visualizer.addTopic("Detections", detectionNetwork.out)

    # Stereo depth - only for stereo devices
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    left_out = left.requestOutput(NN_DIMENSIONS, type=dai.ImgFrame.Type.NV12)
    right_out = right.requestOutput(NN_DIMENSIONS, type=dai.ImgFrame.Type.NV12)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_out,
        right=right_out,
        presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
    )

    align = pipeline.create(dai.node.ImageAlign)
    stereo.depth.link(align.input)
    outputToEncode.link(align.inputAlignTo)

    spatialCalc = pipeline.create(dai.node.SpatialLocationCalculator)
    input_depth = align.outputAligned if platform == dai.Platform.RVC4 else stereo.depth
    input_depth.link(spatialCalc.inputDepth)

    # Set ROI to full image (0,0 to 1,1)
    config_depth = dai.SpatialLocationCalculatorConfigData()
    config_depth.roi = dai.Rect(0.4, 0.4, 0.6, 0.6)
    config_depth.depthThresholds.lowerThreshold = 100  # mm
    config_depth.depthThresholds.upperThreshold = 10000  # mm
    config_depth.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
    spatialCalc.initialConfig.addROI(config_depth)

    dashboard = pipeline.create(DepthDashboardNode)
    spatialCalc.out.link(dashboard.inputSpatial)

    visualizer.addTopic("Depth Map", align.outputAligned if platform == dai.Platform.RVC4 else stereo.depth)
    visualizer.addTopic("Depth Stats", dashboard.output)

    spatialDataQueue = spatialCalc.out.createOutputQueue()


    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break


if __name__ == "__main__":
    _, ags = initialize_argparser()
    run_device(ags)