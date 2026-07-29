# mediapipe_mocap

ROS 2 package for MediaPipe hand tracking from RGB images and OAK-D S2 RGBD
streams.

## Overview

This package provides two hand tracking workflows:

- `hand_landmarks_node` subscribes to RGB images, runs MediaPipe
  HandLandmarker, and publishes 2D reference-relative hand-control points.
- `oak_hand_landmarks_node` captures OAK-D S2 RGB and stereo depth
  directly with DepthAI v3, back-projects MediaPipe landmarks to metric 3D,
  and publishes reference-relative metric points.

## Features

- **Real-time hand tracking** using MediaPipe Tasks API
- **OAK-D S2 RGBD 3D hand tracking** using DepthAI v3 aligned stereo depth
- **Configurable detection thresholds** for detection, presence, and tracking confidence
- **FPS measurement** to monitor processing performance
- **Bounded image latency** using reliable, keep-last depth-1 subscription QoS
- **Built-in viewer** to overlay landmarks with MediaPipe drawing styles

## Published Topics

- `/hand_landmarks` (`sensor_msgs/PointCloud`)
  - 2D node: 21 reference-relative points. `point.x` and `point.y` are
    aspect-corrected normalized image offsets, and `point.z` is `0`. The
    producer does not apply dead-zone removal or saturation.
  - OAK node: 21 reference-relative RGB-camera points in meters. The producer
    does not apply dead-zone removal or saturation.

Landmark publishers and reset subscriptions use ROS queue depth 10 with the
default reliable/volatile QoS policies.

## Subscribed Topics

- `/camera/color/image_raw` (`sensor_msgs/Image`) - Input RGB images for the
  2D node, subscribed with reliable, volatile, keep-last depth-1 QoS.
- `/reset_reference` (`std_msgs/Bool`) - The shared reset topic for both
  producers. A false-to-true transition queues recentering on the next valid
  detected hand, subject to the cooldown.

## 2D node parameters

The table shows the built-in defaults used when the executable is run without
a parameter file.

| Parameter | Type | Built-in default | Description |
|-----------|------|---------|-------------|
| `image_topic` | string | `/camera/color/image_raw` | Input image topic |
| `landmarks_topic` | string | `/hand_landmarks` | Output landmarks topic |
| `model_path` | string | `<auto-resolved>` | Path to MediaPipe model file (auto-resolved to `<package_share>/models/hand_landmarker.task`) |
| `num_hands` | int | 1 | Maximum number of hands to detect |
| `min_hand_detection_confidence` | double | 0.5 | Minimum confidence for hand detection |
| `min_hand_presence_confidence` | double | 0.5 | Minimum confidence for hand presence |
| `min_tracking_confidence` | double | 0.5 | Minimum tracking confidence |
| `delegate` | string | `AUTO` | Execution policy: `AUTO`, `CPU`, or `GPU`; `AUTO` prefers GPU on native Linux and retries CPU if GPU initialization fails |
| `enable_one_euro_filter` | bool | true | Enable One Euro smoothing on each landmark coordinate |
| `one_euro_frequency` | double | 30.0 | Expected landmark update frequency in Hz |
| `one_euro_mincutoff` | double | 1.0 | Minimum cutoff frequency (lower = smoother) |
| `one_euro_beta` | double | 0.1 | Speed coefficient (higher = more responsive) |
| `one_euro_derivative_cutoff` | double | 1.0 | Derivative low-pass cutoff frequency in Hz |
| `visualize` | bool | false | Show a local OpenCV window with landmarks overlay |
| `window_name` | string | `Hand Landmarks (Node)` | Window title when `visualize` is enabled |
| `reset_reference_topic` | string | `/reset_reference` | Topic used to recenter the hand reference |
| `reset_reference_cooldown_sec` | double | 0.25 | Minimum time between accepted reference resets |
| `initial_reference` | double array | `[0.5, 0.5, 0.5]` | Initial reference point in normalized image coordinates |
| `show_control_zones` | bool | true | Draw dead-zone and saturation-zone overlays when visualizing |
| `dead_zone` | double | 0.05 | Display-only dead-zone radius used by the viewer |
| `saturation_zone` | double | 0.3 | Display-only saturation boundary used by the viewer |
| `tracked_landmark_index` | int | 0 | Landmark used for reference reset and control-zone feedback |

The built-in defaults and shipped `config/hand_landmarks_node.yaml` both enable
One Euro filtering with `frequency=30.0`, `mincutoff=1.0`, `beta=0.1`, and
`derivative_cutoff=1.0`. The shipped YAML enables visualization, while the
standalone launch overrides it to `false`. The USB-camera launch uses the YAML
default and overrides `selfie_mode=true`.

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

   # Install the validated Jazzy/Python 3.12 versions.
   python -m pip install \
     --constraint src/tools/mediapipe_mocap/constraints-jazzy.txt \
     numpy opencv-contrib-python mediapipe depthai

   # Verify
   python -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
   python -c "import mediapipe, depthai; print(mediapipe.__version__, depthai.__version__)"

   # Check whether your OAK camera is detected with oak-viewer:
   # https://docs.luxonis.com/software-v3/depthai/tools/oak-viewer/
   # Prefer the USB port that reports the highest USB speed.
   ```

   Do not upgrade this environment to NumPy 2.x. ROS Jazzy's system
   Matplotlib/OpenCV extensions are built against NumPy 1.x, and mixing them
   with NumPy 2.x causes an ABI import failure. The constraints file pins the
   validated NumPy 1.26.4 baseline.

   Activating this pinned environment remains recommended. At startup the
   nodes prefer the active virtual environment and otherwise search up to four
   parent directories for `.venv_mediapipe`, adding its versioned
   `site-packages` directory to `sys.path`.

2. Install the ROS 2 USB camera driver:

   ```bash
   sudo apt update
   sudo apt install ros-$ROS_DISTRO-usb-cam
   ```

   Verify that the package is available:

   ```bash
   ros2 pkg executables usb_cam
   ```

   The output should include `usb_cam usb_cam_node_exe`.

3. Rebuild the package **with the venv active**:

   ```bash
   source .venv_mediapipe/bin/activate
   colcon build --packages-up-to mediapipe_mocap --symlink-install
   ```

4. The MediaPipe hand landmarker model is **provided by default** in the
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
The producer does not apply the `dead_zone`; that parameter affects only the
viewer overlay.

Use the standalone launch file (recommended):

```bash
ros2 launch mediapipe_mocap oak_hand_landmarks_launch.py
```

Or run the executable directly with the OAK config:

```bash
ros2 run mediapipe_mocap oak_hand_landmarks_node \
  --ros-args \
  --params-file $(ros2 pkg prefix mediapipe_mocap)/share/mediapipe_mocap/config/oak_hand_landmarks_node.yaml
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
  saturation_zone:=0.4 \
  landmark_index:=0
```

Key OAK parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rgb_width` | `640` | RGB/depth output width used for MediaPipe and depth sampling |
| `rgb_height` | `400` | RGB/depth output height used for MediaPipe and depth sampling |
| `fps` | `50.0` | OAK camera FPS |
| `delegate` | `AUTO` | MediaPipe execution policy; `AUTO` provides GPU-to-CPU fallback on native Linux |
| `stereo_preset` | `FAST_DENSITY` | DepthAI StereoDepth preset |
| `depth_sample_radius_px` | `2` | Median/percentile depth sampling window radius around each landmark |
| `saturation_zone` | `0.4` | Display-only metric saturation boundary used by the viewer |
| `dead_zone` | `0.05` | Display-only radius; it does not modify published points |
| `auto_reference_on_first_detection` | `true` | Use the first valid tracked landmark as the 3D reference |
| `reset_reference_topic` | `/reset_reference` | Reset topic shared by both producers |
| `enable_one_euro_filter` | `true` | Enable shared One Euro filtering |
| `one_euro_mincutoff` | `1.0` | Minimum cutoff frequency in Hz |
| `one_euro_beta` | `0.1` | Speed coefficient |
| `one_euro_derivative_cutoff` | `1.0` | Derivative low-pass cutoff frequency in Hz |
| `visualize` | `true` | Show a local OpenCV window when loading the shipped OAK YAML; the standalone OAK launch overrides this to `false` |

The OAK filter derives its nominal fallback frequency from `fps`; filtering
normally uses the timestamp interval between consecutive frames.

### Visualization

Hand-landmark visualization is built into the landmark producer nodes. Enable it
with the `visualize` parameter and select the OpenCV window title with
`window_name`. The shared, ROS-independent implementation is available to Python
callers as `mediapipe_mocap.viewer.HandLandmarksViewer`. The performance overlay
refreshes half-second window averages for FPS and MediaPipe processing time;
current-frame status such as missing OAK depth remains live.

### Configuration

Parameters are loaded from `config/hand_landmarks_node.yaml`.

Both landmark producers use MediaPipe's synchronous `VIDEO` mode. Timestamps
are made strictly increasing before each `detect_for_video` call.

**Execution delegate (`delegate`):**

- `AUTO`: prefer GPU on native Linux and retry with CPU if GPU initialization fails.
- `CPU`: require CPU execution.
- `GPU`: require GPU execution and surface initialization errors.

Before importing the computer-vision stack, each node applies the virtual
environment discovery described above. On Linux systems where an NVIDIA
driver is detected, it also sets the PRIME render-offload and GLX vendor
environment variables.

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
  -p model_path:=/path/to/custom/model.task
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
| `visualize` | false | Show local OpenCV visualization window |
| `window_name` | `3D Hand Landmarks OAK` | Window title when `visualize` is enabled |
| `dead_zone` | 0.05 | Display-only dead-zone radius used by the OAK feedback overlay |
| `saturation_zone` | 0.4 | Display-only XYZ saturation distance used by the feedback overlay |
| `landmark_index` | 0 | Tracked landmark index (0-20) for OAK feedback overlay |
| `reset_reference_topic` | `/reset_reference` | Reset topic loaded from the shipped OAK YAML |

### usb_cam_hand_landmarks_launch.py
Complete pipeline using the maintained ROS 2 `usb_cam` driver. The raw camera
topic remains unmirrored; selfie mirroring is applied only inside the hand
landmarks detector.

```bash
ros2 launch mediapipe_mocap usb_cam_hand_landmarks_launch.py
```

Camera defaults are loaded from `config/usb_cam.yaml`. Leave an override empty
to use the YAML value.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `usb_cam_params_file` | package `config/usb_cam.yaml` | usb_cam YAML parameter file |
| `hand_landmarks_params_file` | package `config/hand_landmarks_node.yaml` | Detector YAML parameter file |
| `video_device` | empty | Override the YAML video device, such as `/dev/video1` |
| `framerate` | empty | Override the YAML camera frame rate |
| `image_width` | empty | Override the YAML image width |
| `image_height` | empty | Override the YAML image height |
| `pixel_format` | empty | Override the YAML usb_cam pixel format |
| `frame_id` | empty | Override the YAML camera frame ID |
| `image_topic` | `/camera/color/image_raw` | Raw image topic published by usb_cam and consumed by the detector |
| `selfie_mode` | true | Mirror frames inside the detector |

**Example with custom parameters:**
```bash
ros2 launch mediapipe_mocap usb_cam_hand_landmarks_launch.py \
  video_device:=/dev/video1 \
  framerate:=60.0 \
  image_width:=1280 \
  image_height:=720 \
  pixel_format:=mjpeg2rgb
```

### Testing Hand Landmarks with Offline Video

Run the offline video publisher in one terminal:

```bash
ros2 run offline_media_publisher video_publisher --ros-args \
  --params-file "$(ros2 pkg prefix offline_media_publisher)/share/offline_media_publisher/config/video_publisher.yaml" \
  -p folder_path:=/path/to/videos \
  -p fps:=30
```

In a second terminal, run hand-landmark detection with its standard
configuration:

```bash
ros2 run mediapipe_mocap hand_landmarks_node --ros-args \
  --params-file "$(ros2 pkg prefix mediapipe_mocap)/share/mediapipe_mocap/config/hand_landmarks_node.yaml"
```

## Building

```bash
cd ~/dev/extender_workspace
source .venv_mediapipe/bin/activate          # activate venv with mediapipe
colcon build --packages-up-to mediapipe_mocap --symlink-install
source install/setup.bash
```

## Testing

Activate the pinned virtual environment, build the package and its
`signal_processing` dependency, then run both test suites:

```bash
cd ~/dev/extender_workspace
source .venv_mediapipe/bin/activate
colcon build --packages-up-to mediapipe_mocap --symlink-install
colcon test --packages-select signal_processing mediapipe_mocap
colcon test-result --verbose
```

The current unit and lint tests do not require a camera.

## Dependencies

- Ubuntu 24.04, ROS 2 Jazzy, and Python 3.12
- OpenCV
- cv_bridge
- usb_cam
- MediaPipe (Python, pip-installed in `.venv_mediapipe`)
- DepthAI (Python, pip-installed in `.venv_mediapipe`, for OAK-D S2 RGBD tracking)
- NumPy
- signal_processing

Other ROS distributions and operating-system versions are currently
unverified.

## License

Package code is [BSD-3-Clause](LICENSE). The bundled MediaPipe model remains
under its upstream Apache-2.0 terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
