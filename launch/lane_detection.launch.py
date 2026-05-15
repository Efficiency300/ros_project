#!/usr/bin/env python3
#
# Copyright 2026 - TurtleBot3 Yellow Lane Follower
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# CALIBRATION LAUNCH
# ------------------
# Use this launch file to calibrate perspective_src and yellow HSV values
# BEFORE running full autonomous driving.
#
# Usage:
#   ros2 launch turtlebot3_lane_follower lane_detection.launch.py
#
# Then in rqt → Image View, inspect:
#   /lane/image_projected   → bird's-eye view (tune perspective_src)
#   /lane/image_yellow_mask → yellow HSV mask (tune HSV range)
#   /lane/image_lane_debug  → sliding windows, fitted lanes, midpoint marker
#
# Edit param/lane_detection.yaml to save your calibrated values,
# then rebuild: colcon build --packages-select turtlebot3_lane_follower

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('turtlebot3_lane_follower')

    detection_params = os.path.join(pkg_share, 'param', 'lane_detection.yaml')

    # Lane detector with debug images forced ON for calibration
    # Camera runs on the robot (2jz) and is already available on the ROS network
    lane_detector = Node(
        package='turtlebot3_lane_follower',
        executable='lane_detector_node',
        name='lane_detector_node',
        output='screen',
        parameters=[
            detection_params,
            {'debug_mode': True},   # override YAML to force debug images on
        ],
    )

    return LaunchDescription([
        lane_detector,
    ])
