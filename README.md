# Tools

A collection of utility packages for the extender robot control framework, providing message definitions, motion capture, media publishing, and data replay capabilities.

## Packages

### dynamic_parameters_identification
**Package**: `dynamic_parameters_identification`

Dynamic parameters identification pipeline (trajectories generation and parameters identification). Works only for manipulators.

**Features**:
- Generates non synchronized sinusoids trajectories for each joints.
- Record and replay data in order to identify parameters via Weighted Least Square estimation.
- Saves the parameters in a *.npy file in order to re-use them.

### apriltag_detector
**Package**: `apriltag_detector`

Real-time AprilTag detection and 3D pose estimation from camera feeds. Uses the apriltag library to detect Tag36h11 family tags and compute their position and orientation.

**Features**:
- AprilTag detection from image streams
- 3D pose estimation using camera intrinsics
- Configurable tag sizes and detection parameters
- Integration with USB camera for easy deployment
- Publishes detected tag poses as `AprilTagPoseArray` messages

**Topics**:
- Subscribes: `/image_raw`, `/camera_info`
- Publishes: on a topic defined by a parameter.

### extender_msgs
**Package**: `extender_msgs`

ROS2 message definitions for the extender framework. Defines custom message types for teleoperation commands, joint control, and vision-based applications.

**Messages**:
- `TeleopCommand`: Teleoperation commands with velocity and mode selection
- `JointPositionCommand`: Joint position commands with named joint targeting
- `AprilTagPose`: Single AprilTag detection with pose information
- `AprilTagPoseArray`: Collection of detected AprilTag poses

### mediapipe_mocap
**Package**: `mediapipe_mocap`

Motion capture using Google MediaPipe. Detects and tracks landmarks from RGB camera streams with real-time visualization capabilities.


**Components**:
- `hand_landmarks_node`: Detects hand landmarks from an image using mediapipe
- `3d_hand_landmarks_oak_node`: Detects metric hand landmarks with an OAK camera
- `HandLandmarksViewer`: Shared Python visualization library used by landmark nodes

### offline_media_publisher
**Package**: `offline_media_publisher`

Simulated camera stream publisher for testing and development. Publishes images or video files as camera topics without requiring physical cameras.

**Components**:
- `video_publisher`: Publishes video files as camera topics with configurable playback

### replay_joint_positions
**Package**: `replay_joint_positions`

Utility for replaying recorded joint position trajectories. Useful for testing controllers and reproducing robot motions from recorded data.

## Dependencies

### Common Dependencies
- ROS2 (Humble or later)
- Python 3

### Package-Specific Dependencies

**apriltag_detector**:
- OpenCV
- apriltag
- Eigen3
- cv_bridge
- sensor_msgs
- usb_cam (optional, for camera input)

**mediapipe_mocap**:
- MediaPipe
- OpenCV
- NumPy

**offline_media_publisher**:
- OpenCV

**extender_msgs**:
- ROS2 message generation tools

## Contributing

When adding new tools:
1. Follow ROS2 package structure conventions
2. Include comprehensive documentation
3. Add appropriate tests and examples
4. Update this README with package descriptions
5. Ensure compatibility with the extender framework

## License

TODO: Add license information

