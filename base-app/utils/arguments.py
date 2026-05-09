import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.description = "Real-time rubbish detection using a YOLOv6-nano model and stereo depth estimation. It serves a browser bin monitor with an annotated stream, wrong-bin status, depth, and fullness estimation."

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
