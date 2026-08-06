# mediapipe_mocap

ROS 2 producers for MediaPipe hand tracking from RGB images and OAK-D S2
RGBD streams.

## Producers

The package installs two executables:

| Executable | Input | Published coordinates |
|------------|-------|-----------------------|
| `hand_landmarks_node` | `sensor_msgs/Image` | Reference-relative, aspect-corrected normalized image coordinates |
| `oak_hand_landmarks_node` | OAK RGB and aligned stereo depth captured with DepthAI v3 | Reference-relative RGB optical-camera coordinates in meters |

Both producers run MediaPipe Hand Landmarker synchronously in `VIDEO` mode.
`running_mode` is intentionally not a ROS parameter. Source timestamps are
made strictly increasing before each `detect_for_video` call.

## Quick starts

Build and source the workspace first. See [Environment](#environment) for the
validated Python setup.

### RGB with a USB camera

The complete USB-camera pipeline starts `usb_cam` and
`hand_landmarks_node`:

```bash
ros2 launch mediapipe_mocap usb_cam_hand_landmarks_launch.py \
  video_device:=/dev/video0
```

The camera defaults to 640×480 at 30 FPS. The detector consumes
`/camera/color/image_raw`, and `selfie_mode=true` mirrors the image inside the
detector without changing the camera publisher.

To use an existing RGB image topic instead:

```bash
ros2 launch mediapipe_mocap hand_landmarks_launch.py
```

Override the topic in a custom parameter file, or run the executable directly:

```bash
ros2 run mediapipe_mocap hand_landmarks_node --ros-args \
  --params-file "$(ros2 pkg prefix mediapipe_mocap)/share/mediapipe_mocap/config/hand_landmarks_node.yaml" \
  -p image_topic:=/my/camera/image_raw
```

The shipped RGB YAML enables the OpenCV viewer and its display-only control
overlay. Disable either without changing publication:

```bash
ros2 launch mediapipe_mocap hand_landmarks_launch.py \
  visualize:=false \
  show_control_overlay:=false
```

### OAK-D S2

The OAK producer owns capture, synchronization, aligned depth, MediaPipe
detection, and back-projection:

```bash
ros2 launch mediapipe_mocap oak_hand_landmarks_launch.py
```

The shipped OAK configuration requests 640×400 RGB and aligned depth at
50 FPS. Its viewer is enabled and the optional control overlay is disabled.

Common launch overrides are:

```bash
ros2 launch mediapipe_mocap oak_hand_landmarks_launch.py \
  fps:=50.0 \
  rgb_width:=640 \
  rgb_height:=400 \
  visualize:=true \
  show_control_overlay:=true \
  overlay_normalization_mode:=vector \
  landmark_index:=0
```

For the producer and joystick consumer together:

```bash
ros2 launch hand_joystick_interfaces oak_hand_joystick_launch.py
```

## Published data contract

`/hand_landmarks` remains a `sensor_msgs/msg/PointCloud`.

For every published cloud:

- `points` contains exactly 21 points in MediaPipe landmark order, indices
  0–20, from the first detected hand.
- The points come from the post-filter pipeline state: One Euro-filtered when
  enabled, or converted without smoothing when disabled.
- No dead-zone, saturation, control normalization, axis mapping, or sign
  change is applied.
- No cloud is published when a complete valid hand is unavailable. Consumers
  are responsible for watchdog behavior when input stops.

### RGB coordinates

The RGB producer publishes planar offsets:

```text
x = (landmark_x - reference_x) * width  / min(width, height)
y = (landmark_y - reference_y) * height / min(width, height)
z = 0
```

This preserves equal geometric scale on both axes for non-square images.
MediaPipe's relative landmark depth is not published by the RGB producer.
The cloud copies the complete header of the input `Image`, including its stamp
and `frame_id`.

### OAK coordinates

The OAK producer samples aligned depth near each MediaPipe landmark and
back-projects it through the RGB intrinsics. A valid hand is a complete
21-point metric result after applying the configured missing-depth policy.
The published values are:

```text
point = filtered RGB-camera metric point - metric reference
```

All three coordinates are in meters. The cloud is stamped when the synchronized
RGBD frame is processed and uses `camera_frame_id`, which defaults to
`oak_rgb_camera_optical_frame`.

If the hand cannot produce a complete metric result—for example because too
many depth samples are invalid—no cloud is published, the filters are reset,
and the next valid hand starts a new filter sequence.

## Processing and reference contract

The canonical per-frame state is:

```text
MediaPipe landmarks
        ↓
coordinate conversion / OAK back-projection
        ↓
One Euro filtering (when enabled)
        ↓
reference update ──→ relative PointCloud
        └──────────→ viewer and optional control overlay
```

Publication and visualization therefore use the same post-filter landmark
state. Raw MediaPipe coordinates are never substituted in the viewer when
filtering is enabled. For OAK visualization, the filtered metric points are
reprojected through the RGB intrinsics; an invalid reprojection skips the hand
instead of falling back to raw image coordinates. Rendering does not modify
the published values.

The RGB reference starts at `initial_reference`, `[0.5, 0.5, 0.5]` by
default. The OAK producer defaults to
`auto_reference_on_first_detection=true`, so its reference is initialized
from `tracked_landmark_index` on the first complete hand with valid metric
depth. Until that happens, it publishes no cloud.

Both producers subscribe to `/reset_reference`. A `false`-to-`true`
`std_msgs/Bool` transition requests recentering on the next valid hand. Further
requests inside `reset_reference_cooldown_sec` are ignored. The current
reference is retained when a hand is temporarily lost.

For a manual reset:

```bash
ros2 topic pub --once /reset_reference std_msgs/msg/Bool "{data: false}"
ros2 topic pub --once /reset_reference std_msgs/msg/Bool "{data: true}"
```

## Topics and QoS

| Direction | Topic default | Type | QoS |
|-----------|---------------|------|-----|
| RGB input | `/camera/color/image_raw` | `sensor_msgs/msg/Image` | Reliable, volatile, keep-last depth 1 |
| Output | `/hand_landmarks` | `sensor_msgs/msg/PointCloud` | Reliable, volatile, keep-last depth 10 |
| Reset input | `/reset_reference` | `std_msgs/msg/Bool` | Reliable, volatile, keep-last depth 10 |

The depth-1 RGB subscription bounds queued image latency. OAK images and depth
are exchanged internally through the DepthAI pipeline rather than ROS topics.

## Filtering and visualization

One Euro filtering is enabled by default. Filtering is applied independently
to every axis of all 21 points and occurs exactly once. If filtering is
disabled, the converted but unfiltered points become the canonical state used
by reference, publisher, and viewer.

The RGB filter has a nominal `one_euro_frequency` of 30 Hz. The OAK filter uses
the configured camera `fps`; it has no separate frequency parameter. Both use
frame timestamps for actual sample intervals.

The viewer shows the filtered hand, reference marker, MediaPipe processing
time, and FPS. Timing and FPS are averaged over 0.5-second windows to keep the
display readable. OAK depth validity remains a current-frame status.

`show_control_overlay` adds a display-only preview based on
`tracked_landmark_index`:

- `vector` uses circular/spherical dead and saturation boundaries and
  preserves direction during normalization.
- `axis` uses rectangular/cuboid boundaries and normalizes each component
  independently.

The overlay reports filtered displacement, normalized preview, mode, and
`DEAD`, `ACTIVE`, or `SATURATED` state. It deliberately excludes consumer axis
mapping and sign changes. Its `overlay_*` parameters are ignored when
`show_control_overlay=false`, and they never affect `/hand_landmarks`.

## Parameters

The tables below list executable built-in defaults. The shipped YAML files
override selected viewer defaults as summarized in
[Configuration defaults](#configuration-defaults).

### Common MediaPipe parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `landmarks_topic` | `/hand_landmarks` | Output `PointCloud` topic |
| `model_path` | bundled model | Hand Landmarker task path; an empty string selects the bundled model |
| `num_hands` | `1` | Maximum hands detected; only the first is published |
| `min_hand_detection_confidence` | `0.5` | Palm-detection confidence threshold |
| `min_hand_presence_confidence` | `0.5` | Hand-presence confidence threshold |
| `min_tracking_confidence` | `0.5` | Landmark-tracking confidence threshold |
| `delegate` | `AUTO` | `AUTO`, `CPU`, or `GPU` |
| `enable_one_euro_filter` | `true` | Filter before reference, publication, and display |
| `one_euro_mincutoff` | `1.0` Hz | One Euro minimum cutoff |
| `one_euro_beta` | `0.1` | One Euro speed coefficient |
| `one_euro_derivative_cutoff` | `1.0` Hz | One Euro derivative cutoff |
| `visualize` | `false` | Open the producer diagnostics window |
| `window_name` | node-specific | OpenCV window title |
| `show_control_overlay` | `false` | Add the display-only control preview |
| `overlay_dead_zone` | `0.05` | Display-only dead-zone boundary |
| `overlay_saturation_zone` | `0.3` | Display-only saturation boundary |
| `overlay_normalization_mode` | `vector` | Display-only `axis` or `vector` preview |
| `tracked_landmark_index` | `0` | Landmark used for reference reset and overlay |
| `reset_reference_topic` | `/reset_reference` | Rising-edge reset topic |
| `reset_reference_cooldown_sec` | `0.25` s | Minimum interval between accepted resets |

For RGB, overlay distance is measured in aspect-corrected normalized image
coordinates. For OAK, it is measured in meters.

### RGB-only parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_topic` | `/camera/color/image_raw` | Input RGB image topic |
| `selfie_mode` | `false` | Mirror input images before detection |
| `one_euro_frequency` | `30.0` Hz | Nominal filter sampling frequency |
| `initial_reference` | `[0.5, 0.5, 0.5]` | Initial normalized-image reference; output remains planar |
| `window_name` | `Hand Landmarks (Node)` | RGB viewer title |

### OAK-only parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_frame_id` | `oak_rgb_camera_optical_frame` | Published cloud frame |
| `rgb_width` | `640` px | Aligned RGB/depth width |
| `rgb_height` | `400` px | Aligned RGB/depth height |
| `fps` | `50.0` Hz | OAK capture rate and nominal filter frequency |
| `rgb_socket` | `CAM_A` | RGB camera-board socket |
| `left_socket` | `CAM_B` | Left mono camera-board socket |
| `right_socket` | `CAM_C` | Right mono camera-board socket |
| `stereo_preset` | `FAST_DENSITY` | DepthAI StereoDepth preset |
| `stereo_left_right_check` | `true` | Enable stereo consistency checking |
| `stereo_subpixel` | `false` | Enable subpixel disparity |
| `stereo_extended_disparity` | `false` | Enable extended disparity |
| `stereo_rectify_edge_fill_color` | `0` | Rectification edge-fill value, 0–255 |
| `sync_threshold_ms` | `15.0` ms | Maximum RGB/depth synchronization offset |
| `sync_attempts` | `-1` | DepthAI synchronization attempt policy |
| `sync_run_on_host` | `true` | Run RGB/depth synchronization on the host |
| `depth_sample_radius_px` | `2` px | Sampling radius around each landmark |
| `min_depth_m` | `0.12` m | Minimum accepted depth |
| `max_depth_m` | `3.0` m | Maximum accepted depth |
| `depth_percentile` | `50.0` | Percentile selected from valid nearby samples |
| `missing_depth_strategy` | `reuse_last` | `skip_frame`, `reuse_last`, or `hand_median` |
| `max_missing_depth_landmarks` | `8` | Maximum missing direct samples accepted before rejecting the hand |
| `initial_reference` | `[0.0, 0.0, 0.6]` m | Used when automatic first-hand reference is disabled |
| `auto_reference_on_first_detection` | `true` | Initialize from the first complete metric hand |
| `window_name` | `3D Hand Landmarks OAK` | OAK viewer title |

With `reuse_last`, a missing landmark uses its last valid depth when available,
then the current hand median. `hand_median` uses the current hand median.
`skip_frame` rejects any hand with a missing direct sample. Every policy still
requires a complete 21-point result before publication.

Parameter descriptors include units, ranges, and enum constraints. Inspect
them at runtime with:

```bash
ros2 param describe /hand_landmarks_node overlay_dead_zone
ros2 param describe /oak_hand_landmarks_node missing_depth_strategy
```

## Configuration defaults

The standalone launches load the package YAML, then apply only explicit launch
overrides. Effective precedence is:

```text
explicit terminal launch argument → YAML parameter → executable built-in
```

An omitted standalone launch argument uses the YAML value. The internal
`__use_yaml__` sentinel is never passed to a node.

| Setting | Built-in | RGB YAML | OAK YAML |
|---------|----------|----------|----------|
| `visualize` | `false` | `true` | `true` |
| `show_control_overlay` | `false` | `true` | `false` |
| Filter enabled | `true` | `true` | `true` |
| Nominal filter frequency | RGB 30 Hz / OAK `fps` | 30 Hz | 50 Hz |

`usb_cam_hand_landmarks_launch.py` additionally sets the producer
`image_topic` from its launch argument and defaults `selfie_mode` to `true`.
Camera settings come from `config/usb_cam.yaml`; a non-empty terminal camera
argument overrides the corresponding YAML value.

To use a custom model:

```yaml
hand_landmarks_node:
  ros__parameters:
    model_path: "/path/to/custom/hand_landmarker.task"
```

`AUTO` delegate selection prefers GPU on native Linux and retries with CPU if
GPU initialization fails. `CPU` and `GPU` require the requested delegate and
surface initialization errors. Before importing the vision stack, each node
prefers an active virtual environment or discovers `.venv_mediapipe` in up to
four parent directories. On Linux with an NVIDIA driver, it also requests
PRIME render offload through the NVIDIA environment variables.

## Environment

The supported baseline is Ubuntu 24.04, ROS 2 Jazzy, and Python 3.12. Other
ROS distributions and operating-system versions are unverified.

MediaPipe and DepthAI are installed with pip in `.venv_mediapipe`, not through
rosdep:

```bash
sudo apt install python3.12-venv

cd ~/dev/extender_workspace
python3 -m venv .venv_mediapipe --system-site-packages
source .venv_mediapipe/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --constraint src/tools/mediapipe_mocap/constraints-jazzy.txt \
  numpy opencv-contrib-python mediapipe depthai

python -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
python -c "import mediapipe, depthai; print(mediapipe.__version__, depthai.__version__)"
```

Do not upgrade this environment to NumPy 2.x. ROS Jazzy's system extensions
use the NumPy 1.x ABI. The constraints file pins the validated baseline:

- `numpy==1.26.4`
- `opencv-contrib-python==4.11.0.86`
- `mediapipe==0.10.35`
- `depthai==3.7.1`

Install the ROS USB camera driver when using the RGB USB launch:

```bash
sudo apt update
sudo apt install ros-$ROS_DISTRO-usb-cam
ros2 pkg executables usb_cam
```

Build with the virtual environment active:

```bash
cd ~/dev/extender_workspace
source .venv_mediapipe/bin/activate
colcon build --packages-up-to mediapipe_mocap --symlink-install
source install/setup.bash
```

The Hand Landmarker model is bundled and resolved automatically.

## Offline video

Run the offline publisher:

```bash
ros2 run offline_media_publisher video_publisher --ros-args \
  --params-file "$(ros2 pkg prefix offline_media_publisher)/share/offline_media_publisher/config/video_publisher.yaml" \
  -p folder_path:=/path/to/videos \
  -p fps:=30
```

Then start the RGB producer:

```bash
ros2 launch mediapipe_mocap hand_landmarks_launch.py
```

## Testing

The unit, lint, and launch-contract tests require no camera:

```bash
cd ~/dev/extender_workspace
source .venv_mediapipe/bin/activate
colcon build --packages-up-to mediapipe_mocap --symlink-install
colcon test --packages-select signal_processing mediapipe_mocap
colcon test-result --verbose
```

## License

Package code is [BSD-3-Clause](LICENSE). The bundled MediaPipe model remains
under its upstream Apache-2.0 terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
