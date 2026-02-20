from setuptools import find_packages, setup

package_name = 'mediapipe_mocap'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/hand_landmarks_launch.py',
            'launch/hand_landmarks_depth_inference_launch.py',
            'launch/webcam_hand_landmarks.launch.py',
            'launch/realsense_d435_hand_landmarks.launch.py',
            'launch/viewer_launch.py',
            'launch/test_offline_video_hand_landmarks_launch.py'
        ]),
        ('share/' + package_name + '/config', [
            'config/hand_landmarks_node.yaml',
            'config/hand_landmarks_node_depth_inference.yaml'
        ]),
        ('share/' + package_name + '/models', ['models/hand_landmarker.task']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='emoullet',
    maintainer_email='etienne.moullet@gmail.com',
    description='ROS2 node extracting hand landmarks from an RGB image using MediaPipe.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_landmarks_node = mediapipe_mocap.hand_landmarks_node:main',
            'hand_landmarks_node_depth_inference = mediapipe_mocap.hand_landmarks_node_depth_inference:main',
            'viewer_node = mediapipe_mocap.viewer_node:main',
        ],
    },
)
