# mediapipe_mocap

ROS 2 package for MediaPipe hand tracking from RGB images and OAK-D S2 RGBD
streams.

## Overview

This package provides two hand tracking workflows:

- `hand_landmarks_node` subscribes to RGB images, runs MediaPipe
  HandLandmarker, and publishes 2D reference-relative hand-control points.
- `3d_hand_landmarks_oak_node` captures OAK-D S2 RGB and stereo depth
  directly with DepthAI v3, back-projects MediaPipe landmarks to metric 3D,
  and publishes either normalized control points or metric relative points.

## Features

- **Real-time hand tracking** using MediaPipe Tasks API
- **OAK-D S2 RGBD 3D hand tracking** using DepthAI v3 aligned stereo depth
- **Configurable detection thresholds** for detection, presence, and tracking confidence
- **FPS measurement** to monitor processing performance
- **Low-latency mode** using sensor data QoS profile
- **Built-in viewer** to overlay landmarks with MediaPipe drawing styles

## Published Topics

- `/hand_landmarks` (`sensor_msgs/PointCloud`)
  - 2D node: 21 reference-relative control points. `point.x` and `point.y`
    are normalized by image scale, and `point.z` is `0`.
  - OAK node: 21 saturated normalized control points in `[-1, 1]` by default.
    Set `publish_normalized_landmarks=false` to publish reference-relative
    metric coordinates in meters.
- Optional OAK raw topic from `raw_landmarks_topic` (`sensor_msgs/PointCloud`)
  - Reference-relative metric coordinates before normalization.

## Subscribed Topics

- `/camera/color/image_raw` (`sensor_msgs/Image`) - Input RGB images for the
  2D node
- `/reset_reference` (`std_msgs/Bool`) - Rising edge recenters the 2D hand
  reference. The OAK node uses its configured `reset_reference_topic`.

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
| `reset_reference_topic` | string | `/reset_reference` | Topic used to recenter the hand reference |
| `reset_reference_cooldown_sec` | double | 0.25 | Minimum time between accepted reference resets |
| `initial_reference` | double array | `[0.5, 0.5, 0.5]` | Initial reference point in normalized image coordinates |
| `show_control_zones` | bool | true | Draw dead-zone and saturation-zone overlays when visualizing |
| `dead_zone` | double | 0.05 | Reference-relative radius treated as no motion |
| `saturation_zone` | double | 0.3 | Reference-relative distance shown as the saturation boundary |
| `tracked_landmark_index` | int | 0 | Landmark used for reference reset and control-zone feedback |

## Usage

### Prerequisites

1. Install MediaPipe and DepthAI in a virtual environment. These Python
   packages are intentionally installed with pip in `.venv_mediapipe`, not
   through rosdep package dependencies.

   ```bash
   # Install venv support (Ubuntu 24.04)
   sudo apt install python3.12-venv

   # Create venv in workspace root (inherits ROS system packages)
   cd ~/dev/extender_workspace
   python3 -m venv .venv_mediapipe --system-site-packages
   source .venv_mediapipe/bin/activate

   # Upgrade pip only (do NOT upgrade setuptools/wheel to avoid colcon conflicts)
   python -m pip install --upgrade pip

   # Install MediaPipe (MediaPipe will pull numpy 2.x — this is expected)
   python -m pip install mediapipe

   # Install DepthAI v3 for the OAK-D S2 RGBD node
   python -m pip install depthai

   # Verify
   python -c "import mediapipe as mp; print(mp.__version__)"
   python -c "import depthai as dai; print(dai.__version__)"

   # Check whether your OAK camera is detected with oak-viewer:
   # https://docs.luxonis.com/software-v3/depthai/tools/oak-viewer/
   # Prefer the USB port that reports the highest USB speed.
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

3. The MediaPipe hand landmarker model is **provided by default** in the
   `models/` folder of this package and automatically resolved at runtime.

### Running the Node

Use the launch file (recommended):

```bash
ros2 launch mediapipe_mocap hand_landmarks_launch.py
```

Or run directly (model path is auto-resolved):

```bash
ros2 run mediapipe_mocap hand_landmarks_node
```

### Running the OAK-D S2 RGBD 3D Node

The OAK node captures RGB and stereo depth directly with DepthAI v3, aligns depth
to the RGB frame, runs MediaPipe HandLandmarker on RGB, back-projects each
landmark with the RGB intrinsics, and publishes 21 reference-relative 3D points.
By default those points are saturated normalized control inputs in `[-1, 1]`.

Use the standalone launch file (recommended):

```bash
ros2 launch mediapipe_mocap oak_hand_landmarks_launch.py
```

Or run the executable directly with the OAK config:

```bash
ros2 run mediapipe_mocap 3d_hand_landmarks_oak_node \
  --ros-args \
  --params-file $(ros2 pkg prefix mediapipe_mocap)/share/mediapipe_mocap/config/3d_hand_landmarks_oak_node.yaml
```

For the complete OAK hand joystick pipeline, launch it from `hand_joystick_interfaces`:

```bash
ros2 launch hand_joystick_interfaces oak_hand_joystick_launch.py
```

Useful overrides:

```bash
ros2 launch mediapipe_mocap oak_hand_landmarks_launch.py \
  fps:=50.0 \
  rgb_width:=640 \
  rgb_height:=400 \
  visualize:=true \
  publish_normalized_landmarks:=false \
  raw_landmarks_topic:=/hand_landmarks_raw \
  saturation_zone:=0.4 \
  landmark_index:=0
```

Key OAK parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rgb_width` | `640` | RGB/depth output width used for MediaPipe and depth sampling |
| `rgb_height` | `400` | RGB/depth output height used for MediaPipe and depth sampling |
| `fps` | `50.0` | OAK camera FPS |
| `stereo_preset` | `FAST_DENSITY` | DepthAI StereoDepth preset |
| `depth_sample_radius_px` | `2` | Median/percentile depth sampling window radius around each landmark |
| `publish_normalized_landmarks` | `true` | Publish saturated normalized inputs instead of metric relative coordinates |
| `saturation_zone` | `0.4` | Metric displacement that maps to `+/-1` |
| `normalization_mode` | `axis` | `axis` clips each component with `saturation_zone`; `vector` clips by vector norm |
| `auto_reference_on_first_detection` | `true` | Use the first valid tracked landmark as the 3D reference |

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

### oak_hand_landmarks_launch.py
Starts only the OAK-D S2 RGBD 3D hand landmarks node:
```bash
ros2 launch mediapipe_mocap oak_hand_landmarks_launch.py
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `fps` | 50.0 | OAK camera FPS |
| `rgb_width` | 640 | OAK RGB/depth output width in pixels |
| `rgb_height` | 400 | OAK RGB/depth output height in pixels |
| `visualize` | true | Show local OpenCV visualization window |
| `window_name` | `3D Hand Landmarks OAK` | Window title when `visualize` is enabled |
| `publish_normalized_landmarks` | true | Publish normalized control landmarks instead of metric 3D landmarks |
| `raw_landmarks_topic` | empty | Optional topic for metric camera-frame landmarks before normalization |
| `dead_zone` | 0.05 | Dead zone radius used by the OAK feedback overlay |
| `saturation_zone` | 0.4 | XYZ saturation distance used by normalization and feedback overlay |
| `landmark_index` | 0 | Tracked landmark index (0-20) for OAK feedback overlay |

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
- MediaPipe (Python, pip-installed in `.venv_mediapipe`)
- DepthAI (Python, pip-installed in `.venv_mediapipe`, for OAK-D S2 RGBD tracking)
- NumPy

## License

BSD-3-Clause
