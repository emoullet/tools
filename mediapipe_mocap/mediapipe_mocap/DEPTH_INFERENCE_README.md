# Hand Landmarks Depth Inference Node

## Overview

This node extends the basic hand landmarks detection by computing the hand's depth relative to a registered reference position. It uses:
- **2D hand landmarks** (normalized coordinates from MediaPipe)
- **3D world landmarks** (metric coordinates in hand-centric frame)
- **Camera intrinsics** (approximated from image dimensions)
- **Reference position** (registered via service call)

## How It Works

### Depth Estimation Algorithm

The node estimates the absolute depth of the hand using perspective projection:

1. **Camera Intrinsics Estimation**: On the first image, the node estimates camera intrinsics (focal length, principal point) assuming a typical webcam with ~60° horizontal field of view.

2. **Metric Scale from 3D Landmarks**: MediaPipe provides 3D world landmarks in metric units (meters) in a hand-centric coordinate frame. The node computes the hand span (wrist to middle finger tip) in both:
   - 3D metric space (from world landmarks)
   - 2D pixel space (from normalized image coordinates)

3. **Depth Calculation**: Using perspective projection:
   ```
   depth = (focal_length × real_hand_size) / pixel_hand_size
   ```

4. **Relative Depth**: Once a reference position is registered, all subsequent depth measurements are reported relative to that reference.

## Usage

### Launch the Node

```bash
ros2 launch mediapipe_mocap hand_landmarks_depth_inference_launch.py
```

Or run directly:
```bash
ros2 run mediapipe_mocap hand_landmarks_node_depth_inference
```

### Register Reference Position

Once the node is running and detecting your hand, register the current position as the reference:

```bash
ros2 service call /hand_landmarks_depth_inference_node/register_reference std_srvs/srv/Trigger
```

After registration, the node will publish depth values relative to this reference position.

### Topics

**Subscribed:**
- `/camera/color/image_raw` (sensor_msgs/Image) - Input RGB image

**Published:**
- `/hand_landmarks_depth` (sensor_msgs/PointCloud) - 21 hand landmarks with:
  - `x`, `y`: normalized 2D coordinates [0, 1]
  - `z`: estimated absolute depth (meters)
- `/hand_depth` (std_msgs/Float32) - Relative depth from reference position (meters)
  - Positive values: hand is further from camera than reference
  - Negative values: hand is closer to camera than reference

### Services

- `~/register_reference` (std_srvs/Trigger) - Register current hand position as reference

### Parameters

Configure in [hand_landmarks_node_depth_inference.yaml](../config/hand_landmarks_node_depth_inference.yaml):

- `image_topic`: Input image topic (default: `/camera/color/image_raw`)
- `landmarks_topic`: Output landmarks topic (default: `/hand_landmarks_depth`)
- `depth_topic`: Output depth topic (default: `/hand_depth`)
- `model_path`: Path to MediaPipe model file (default: auto-detected)
- `num_hands`: Maximum number of hands to detect (default: 1)
- `min_hand_detection_confidence`: Minimum confidence for hand detection (default: 0.5)
- `min_hand_presence_confidence`: Minimum confidence for hand presence (default: 0.5)
- `min_tracking_confidence`: Minimum confidence for hand tracking (default: 0.5)
- `focal_length_factor`: Adjustment factor for focal length estimation (default: 1.0)
  - Increase if depth is underestimated
  - Decrease if depth is overestimated

## Example Use Cases

1. **Hand-based depth control**: Control robot or application parameters by moving your hand forward/backward
2. **Gesture recognition with depth**: Combine hand gestures with depth information for richer interaction
3. **Virtual painting**: Use depth as a third dimension for 3D drawing applications
4. **Volume control**: Map hand depth to audio volume or other continuous parameters

## Calibration Tips

If depth estimates seem inaccurate:

1. **Adjust focal_length_factor**: The default assumes a 60° FOV. If your camera has different optics, adjust this parameter.
2. **Camera calibration**: For better accuracy, use proper camera calibration and modify the `_estimate_camera_intrinsics` method to load calibration data.
3. **Lighting conditions**: Ensure good lighting for reliable MediaPipe detection.
4. **Hand visibility**: Keep your hand fully visible and avoid occlusions.

## Limitations

- **Single hand tracking**: Currently tracks only the first detected hand
- **Approximate intrinsics**: Camera parameters are estimated, not measured
- **Scale ambiguity**: Depth accuracy depends on hand size consistency
- **Reference drift**: Re-register reference if tracking is lost or hand detection quality degrades

## Technical Details

The depth inference uses the fact that MediaPipe's world landmarks provide metric scale information. By comparing the metric size of the hand (from 3D world landmarks) to its apparent size in the image (from 2D landmarks), we can solve for the hand's distance from the camera using the pinhole camera model.

This approach doesn't require depth sensors or stereo cameras, making it useful for monocular RGB camera setups.
