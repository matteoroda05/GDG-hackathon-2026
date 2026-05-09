import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.description = "Real-time object detection using a YOLOv6-nano model and stereo depth estimation (if the device has stereo cameras). It streams raw video, H.264/MJPEG encoded video, object detection results, and a colorized depth map to a remote visualizer for monitoring and analysis."

    parser.add_argument(
        "-d",
        "--device",
        help="Optional name, DeviceID or IP of the camera to connect to.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "--web-host",
        help="Host interface for the browser dashboard.",
        required=False,
        default="0.0.0.0",
        type=str,
    )

    parser.add_argument(
        "--web-port",
        help="Port for the browser dashboard.",
        required=False,
        default=8080,
        type=int,
    )

    args = parser.parse_args()

    return parser, args
