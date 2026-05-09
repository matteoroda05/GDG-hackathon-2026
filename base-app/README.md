# Default Application

This application performs real-time object detection using a [YOLOv6 Nano](https://models.luxonis.com/luxonis/yolov6-nano/face58c4-45ab-42a0-bafc-19f9fee8a034) model and stereo depth estimation (if the device has stereo cameras). It now serves a browser dashboard that shows the live camera stream, YOLO detections, and the average depth reading.

## Demo

![Demo](./media/demo.gif)

## Usage

Running this example requires a **Luxonis device** connected to your computer. Refer to the [documentation](https://docs.luxonis.com/software-v3/) to setup your device if you haven't done it already.

You can run the example fully on device ([`STANDALONE` mode](#standalone-mode-rvc4-only)) or using your computer as host ([`PERIPHERAL` mode](#peripheral-mode)).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
--web-host WEB_HOST   Host interface for the browser dashboard. (default: 0.0.0.0)
--web-port WEB_PORT   Port for the browser dashboard. (default: 8080)
```

## Peripheral Mode

### Installation

You need to first prepare a **Python 3.10+** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

The browser dashboard uses Python's built-in HTTP server, so no additional web framework packages are needed.

You can simply install them by running:

```bash
pip install -r requirements.txt
```

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

### Examples

```bash
python3 main.py
```

This will run the example with default arguments. Open the dashboard in your browser at http://localhost:8080.

If you want to change the dashboard host or port, pass the new values explicitly:

```bash
python3 main.py --web-host 0.0.0.0 --web-port 8080
```

## Standalone Mode (RVC4 only)

Running the example in the standalone mode, app runs entirely on the device.
To run the example in this mode, first install the `oakctl` tool using the installation instructions [here](https://docs.luxonis.com/software-v3/oak-apps/oakctl).

The app can then be run with:

```bash
oakctl connect <DEVICE_IP>
oakctl app run .
```

This will run the example with default argument values. If you want to change these values you need to edit the `oakapp.toml` file (refer [here](https://docs.luxonis.com/software-v3/oak-apps/configuration/) for more information about this configuration file).
