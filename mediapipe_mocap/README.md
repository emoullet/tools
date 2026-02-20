# mediapipe_mocap

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
- `/camera/aligned_depth_to_color/image_raw` (`sensor_msgs/Image`) - Optional aligned depth image (when `use_depth=true`)
- `/camera/color/camera_info` (`sensor_msgs/CameraInfo`) - Optional camera intrinsics for depth projection (when `use_depth=true`)

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_topic` | string | `/camera/color/image_raw` | Input image topic |
| `landmarks_topic` | string | `/hand_landmarks` | Output landmarks topic |
| `model_path` | string | `<auto-resolved>` | Path to MediaPipe model file (auto-resolved to `<package_share>/models/hand_landmarker.task`) |
| `num_hands` | int | 1 | Maximum number of hands to detect |
| `use_depth` | bool | false | Enable RGB+depth fusion to publish metric XYZ landmarks |
| `depth_topic` | string | `/camera/aligned_depth_to_color/image_raw` | Depth image topic aligned with `image_topic` |
| `camera_info_topic` | string | `/camera/color/camera_info` | Camera intrinsics topic used for projection |
| `depth_time_tolerance_ms` | double | 10.0 | Maximum accepted RGB/depth timestamp mismatch |
| `depth_min_m` | double | 0.05 | Minimum valid depth (meters) |
| `depth_max_m` | double | 2.0 | Maximum valid depth (meters) |
| `min_hand_detection_confidence` | double | 0.5 | Minimum confidence for hand detection |
| `min_hand_presence_confidence` | double | 0.5 | Minimum confidence for hand presence |
| `min_tracking_confidence` | double | 0.5 | Minimum tracking confidence |
 

## Usage

### Prerequisites

1. Install MediaPipe:
   ```bash
   pip install mediapipe
   ```

2. The MediaPipe hand landmarker model is **provided by default** in the `models/` folder of this package and automatically resolved at runtime.

### Running the Node

Use the launch file (recommended):

```bash
ros2 launch mediapipe_mocap hand_landmarks_launch.py
```

Enable depth fusion from launch:

```bash
ros2 launch mediapipe_mocap hand_landmarks_launch.py \
  use_depth:=true \
  depth_topic:=/camera/aligned_depth_to_color/image_raw \
  camera_info_topic:=/camera/color/camera_info
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

### Testing with Offline Video

**Prerequisites:**

The offline video test requires the `offline_media_publisher` package. Install it with:

```bash
sudo apt install ros-humble-offline-media-publisher
```

To test hand landmark detection on pre-recorded video files without requiring a live camera feed, use the offline video test launch file:

```bash
ros2 launch mediapipe_mocap test_offline_video_hand_landmarks_launch.py folder_path:=/path/to/video/folder fps:=50
```

**Parameters:**

- `folder_path` (required) - Absolute path to a folder containing video files (e.g., `.mp4`, `.avi`)
- `fps` (optional, default: 50) - Publishing rate in Hz. Values lower than the native video FPS will slow down playback, higher values will speed it up

**What it does:**

This launch file starts three nodes together:
1. **offline_media_publisher** - Reads video files from the specified folder and publishes frames to `/camera/color/image_raw`
2. **hand_landmarks_node** - Processes frames and detects hand landmarks, publishing to `/hand_landmarks`
3. **viewer_node** - Displays the input frames with overlaid detected hand landmarks in real-time

**Example:**

```bash
ros2 launch mediapipe_mocap test_offline_video_hand_landmarks_launch.py folder_path:=$HOME/my_videos fps:=30
```

### Configuration

Parameters are loaded from `config/hand_landmarks_node.yaml`. 

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

 

## Building

```bash
cd /path/to/workspace
colcon build --packages-select mediapipe_mocap
source install/setup.bash
```

## Dependencies

- ROS 2 (Humble or later)
- OpenCV
- cv_bridge
- MediaPipe (Python)
- NumPy
- `offline_media_publisher` (optional, required only for offline video testing)

## License

BSD-3-Clause
