# Extender Messages

A ROS2 message package that defines custom message types for the extender robot control framework. This package provides standardized interfaces for teleoperation and joint control commands.

## Overview

The `extender_msgs` package contains ROS2 message definitions used throughout the extender project for:

- **Teleoperation Commands**: Structured messages for robot teleoperation with mode selection
- **Joint Position Commands**: Messages for direct joint-space position control
- **Shared Control Goals**: Messages for representing detected goals (e.g., from AprilTags) in shared control scenarios
- **Standardized Interfaces**: Consistent message formats across different controllers and interfaces

## Messages

### CartesianVelocityCommand

An atomic Cartesian velocity command carrying both the physical reference frame and the frame used
to interpret its angular components.

**Fields**:

- **`header`** (`std_msgs/Header`): Timestamp and physical frame for the command.
- **`twist`** (`geometry_msgs/Twist`): Linear and angular velocity components.
- **`orientation_frame_id`** (`string`): Angular-input frame selector. Supported constants are
  `BASE_FRAME` (`base_frame`), `EFFECTOR_FRAME` (`effector_frame`), and `HYBRID_FRAME`
  (`hybrid_frame`).

### TeleopCommand

A comprehensive teleoperation command message that combines velocity commands with operational modes.

**Fields**:

- **`twist`** (`geometry_msgs/Twist`): Desired Cartesian velocity command
  - `linear`: Linear velocity components (x, y, z) in m/s
  - `angular`: Angular velocity components (x, y, z) in rad/s

- **`mode`** (`uint8`): Teleoperation mode selector
  - `TRANSLATION_ROTATION` (0): Combined translation and rotation control
  - `ROTATION` (1): Rotation-only control (linear velocity ignored)
  - `TRANSLATION` (2): Translation-only control (angular velocity ignored)
  - `BOTH` (3): Full 6DOF control (legacy mode)

### JointPositionCommand

A message for commanding specific joint positions by name.

**Fields**:

- **`joint_names`** (`string[]`): Array of joint names to command
- **`desired_position`** (`float64[]`): Corresponding target positions in radians (or meters for prismatic joints)

**Requirements**:
- `joint_names` and `desired_position` arrays must have the same length
- Joint names must match those defined in the robot's URDF
- Positions must be within joint limits

### SharedControlGoal

A message representing a single shared control goal, typically corresponding to a detected object or target pose.

**Fields**:

- **`id`** (`int32`): Unique identifier of the goal (e.g., AprilTag ID)
- **`goal_pose`** (`geometry_msgs/Pose`): 3D pose of the goal
  - `position`: 3D position (x, y, z) in meters
  - `orientation`: Quaternion (x, y, z, w) representing goal orientation

### SharedControlGoalArray

A collection of shared control goals with timestamp information.

**Fields**:

- **`header`** (`std_msgs/Header`): Standard ROS header with timestamp and frame information
- **`goal_array`** (`extender_msgs/SharedControlGoal[]`): Array of shared control goals

**Usage**: Used for publishing arrays of detected goals from perception pipelines, such as AprilTag detection for shared control applications

## Installation

1. Ensure the package is in your ROS2 workspace:
   ```bash
   cd ~/ros2_humble_ws/src/extender_workspace/tools
   # extender_msgs should be present here
   ```

2. Build the workspace:
   ```bash
   cd ~/ros2_humble_ws
   colcon build --packages-select extender_msgs
   ```

The messages will be automatically generated and made available to other ROS2 packages.
