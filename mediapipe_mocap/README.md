# hand_landmarks_mediapipe

ROS 2 node that detects hand landmarks from RGB images using Google MediaPipe and publishes them as a PointCloud message.

## Overview

This package subscribes to camera images and runs MediaPipe's HandLandmarker to detect hand positions in real-time. It outputs 21 normalized 3D landmarks per detected hand.

## Features

- **Real-time hand tracking** using MediaPipe Tasks API
- **Configurable detection thresholds** for detection, presence, and tracking confidence
- **FPS measurement** to monitor processing performance
- **Low-latency mode** using sensor data QoS profile
- **Built-in viewer** to overlay landmarks with MediaPipe drawing styles

## Published Topics

- `/hand_landmarks` (`sensor_msgs/PointCloud`) - 21 normalized hand landmarks
  - `point.x`: normalized x coordinate [0, 1]
  - `point.y`: normalized y coordinate [0, 1]
  - `point.z`: normalized depth-like value (wrist ≈ 0)

## Subscribed Topics

- `/camera/color/image_raw` (`sensor_msgs/Image`) - Input RGB images

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_topic` | string | `/camera/color/image_raw` | Input image topic |
| `landmarks_topic` | string | `/hand_landmarks` | Output landmarks topic |
| `model_path` | string | `<auto-resolved>` | Path to MediaPipe model file (auto-resolved to `<package_share>/models/hand_landmarker.task`) |
| `num_hands` | int | 1 | Maximum number of hands to detect |
| `min_hand_detection_confidence` | double | 0.5 | Minimum confidence for hand detection |
| `min_hand_presence_confidence` | double | 0.5 | Minimum confidence for hand presence |
| `min_tracking_confidence` | double | 0.5 | Minimum tracking confidence |
| `running_mode` | string | `VIDEO` | MediaPipe running mode: `VIDEO` (sync) or `LIVE_STREAM` (async callback) |
| `enable_one_euro_filter` | bool | false | Enable One Euro smoothing on each landmark coordinate |
| `one_euro_frequency` | double | 30.0 | Expected landmark update frequency in Hz |
| `one_euro_mincutoff` | double | 1.0 | Minimum cutoff frequency (lower = smoother) |
| `one_euro_beta` | double | 0.1 | Speed coefficient (higher = more responsive) |
| `visualize` | bool | false | Show a local OpenCV window with landmarks overlay |
| `window_name` | string | `Hand Landmarks (Node)` | Window title when `visualize` is enabled |
 

## Usage

### Prerequisites

1. Install MediaPipe in a virtual environment:

   ```bash
   # Install venv support (Ubuntu 24.04)
   sudo apt install python3.12-venv

   # Create venv in workspace root (inherits ROS system packages)
   cd ~/dev/extender_workspace
   python3 -m venv .venv_mediapipe --system-site-packages
   source .venv_mediapipe/bin/activate

   # Upgrade pip only (do NOT upgrade setuptools/wheel to avoid colcon conflicts)
   python -m pip install --upgrade pip

   # Install mediapipe (mediapipe will pull numpy 2.x — this is expected)
   python -m pip install mediapipe

   # Verify
   python -c "import mediapipe as mp; print(mp.__version__)"
   ```

   > **Note on dependency warnings:** mediapipe installs numpy 2.x, which is incompatible
   > with the system `scipy`. This is safe as long as `scipy` is not used in this package.
   > If you need scipy compatibility, pin numpy: `pip install "numpy<2"` before installing mediapipe
   > (mediapipe 0.10.x accepts numpy 1.x too).

2. Rebuild the package **with the venv active**:

   ```bash
   source .venv_mediapipe/bin/activate
   colcon build --packages-select mediapipe_mocap --symlink-install
   ```

2. The MediaPipe hand landmarker model is **provided by default** in the `models/` folder of this package and automatically resolved at runtime.

### Running the Node

Use the launch file (recommended):

```bash
ros2 launch mediapipe_mocap hand_landmarks_launch.py
```

Or run directly (model path is auto-resolved):

```bash
ros2 run mediapipe_mocap hand_landmarks_node
```

### Running the Viewer

Use the bundled viewer to overlay landmarks on the input image:

```bash
ros2 launch mediapipe_mocap viewer_launch.py
```

Or run directly:

```bash
ros2 run mediapipe_mocap viewer_node
```

Topics can be overridden via parameters (`image_topic`, `landmarks_topic`, `window_name`).

### Configuration

Parameters are loaded from `config/hand_landmarks_node.yaml`. 

**Running mode (`running_mode`):**
- `VIDEO`: synchronous processing with `detect_for_video`, good default for deterministic frame-by-frame handling.
- `LIVE_STREAM`: asynchronous processing with `detect_async` + callback, useful for stream-oriented pipelines.

**Model path handling:**
- By default, `model_path` is left empty in the YAML file, which triggers automatic resolution to `<package_share>/models/hand_landmarker.task`
- To use a custom model, edit `config/hand_landmarks_node.yaml` and set `model_path` to an absolute path:
  ```yaml
  hand_landmarks_node:
    ros__parameters:
      model_path: "/path/to/custom/hand_landmarker.task"
  ```

Override parameters at runtime:

```bash
ros2 run mediapipe_mocap hand_landmarks_node \
  --ros-args \
  -p image_topic:=/my/custom/image/topic \
  -p model_path:=/path/to/custom/model.task \
  -p running_mode:=VIDEO
```

Run with asynchronous mode:

```bash
ros2 run mediapipe_mocap hand_landmarks_node \
  --ros-args \
  -p running_mode:=LIVE_STREAM
```

Enable built-in visualization directly in the node:

```bash
ros2 run mediapipe_mocap hand_landmarks_node \
  --ros-args \
  -p visualize:=true \
  -p window_name:="Hand Landmarks (Node)"
```

## Launch Files

### hand_landmarks_launch.py
Starts only the hand landmarks detection node:
```bash
ros2 launch mediapipe_mocap hand_landmarks_launch.py
```

Select running mode at launch time:

```bash
ros2 launch mediapipe_mocap hand_landmarks_launch.py running_mode:=LIVE_STREAM
```

With built-in visualization enabled:
```bash
ros2 launch mediapipe_mocap hand_landmarks_launch.py visualize:=true window_name:="Hand Landmarks (Node)"
```

### viewer_launch.py
Starts only the viewer node for visualization:
```bash
ros2 launch mediapipe_mocap viewer_launch.py
```

### webcam_hand_landmarks_launch.py
Complete pipeline: captures video from a USB/integrated webcam, detects hand landmarks, and displays results.

```bash
ros2 launch mediapipe_mocap webcam_hand_landmarks_launch.py
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_id` | 0 | Camera device ID (0 = primary webcam) |
| `fps` | 0 | Publishing rate in Hz (0 = use native camera FPS) |
| `frame_width` | 0 | Frame width in pixels (0 = use native camera width) |
| `frame_height` | 0 | Frame height in pixels (0 = use native camera height) |

**Example with custom parameters:**
```bash
ros2 launch mediapipe_mocap webcam_hand_landmarks_launch.py camera_id:=1 fps:=60 frame_width:=1280 frame_height:=720
```

### test_offline_video_hand_landmarks_launch.py
Processes video files from a directory. Requires the `offline_media_publisher` package:
```bash
ros2 launch mediapipe_mocap test_offline_video_hand_landmarks_launch.py folder_path:=/path/to/videos fps:=30
```

## Webcam Publisher Node

The `webcam_publisher` node captures frames from a USB or integrated webcam and publishes them as ROS 2 Image messages. It runs standalone or can be used with any image processing pipeline.

**Executable:** `webcam_publisher`

**Published Topics:**
- `/camera/color/image_raw` (`sensor_msgs/Image`) - RGB image frames

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_id` | int | 0 | Camera device ID |
| `fps` | int | 0 | Capture rate in Hz (0 = use native camera FPS) |
| `frame_width` | int | 0 | Frame width in pixels (0 = use native camera width) |
| `frame_height` | int | 0 | Frame height in pixels (0 = use native camera height) |

**Configuration file:** `config/webcam_publisher.yaml`

**Finding available camera IDs:**

```bash
sudo apt install v4l-utils
v4l2-ctl --list-devices
```

Example output:
```
Integrated Camera (usb-0000:00:1a.0-1.6):
        /dev/video0
        /dev/video1

USB Webcam (usb-0000:00:1d.0-1.2):
        /dev/video2
        /dev/video3
```

The integer after `/dev/video` is the `camera_id` to use (e.g., `camera_id:=2`).

**Run standalone:**
```bash
ros2 run mediapipe_mocap webcam_publisher
```

**With custom parameters:**
```bash
ros2 run mediapipe_mocap webcam_publisher --ros-args -p camera_id:=1 -p fps:=60
```



## Building

```bash
cd ~/dev/extender_workspace
source .venv_mediapipe/bin/activate          # activate venv with mediapipe
colcon build --packages-select mediapipe_mocap --symlink-install
source install/setup.bash
```

## Dependencies

- ROS 2 (Humble or later)
- OpenCV
- cv_bridge
- MediaPipe (Python)
- NumPy

## License

BSD-3-Clause
