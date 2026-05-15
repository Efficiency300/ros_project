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
# FULL AUTONOMOUS DRIVING LAUNCH
# --------------------------------
# Starts camera + lane detector + lane controller.
# Calibrate first with lane_detection.launch.py before using this.
#
# Usage (normal run — no debug overhead):
#   ros2 launch turtlebot3_lane_follower lane_following.launch.py
#
# Usage (with live debug images for monitoring):
#   ros2 launch turtlebot3_lane_follower lane_following.launch.py debug_mode:=true
#
# The robot will start driving as soon as the lane detector publishes
# a valid /lane/center_error.  Place the robot centred on the track
# before launching.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('turtlebot3_lane_follower')

    detection_params = os.path.join(pkg_share, 'param', 'lane_detection.yaml')
    controller_params = os.path.join(pkg_share, 'param', 'lane_controller.yaml')

    # Optional debug_mode argument (default off for production)
    debug_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description=(
            'Publish /lane/image_projected, /lane/image_yellow_mask, '
            '/lane/image_lane_debug for live monitoring. '
            'Adds ~40 ms latency on RPi — keep false for best performance.'
        ),
    )

    # Camera runs on the robot (2jz) — already available on the ROS network.
    # Start it separately: ros2 launch turtlebot3_bringup camera.launch.py width:=320 height:=240 framerate:=10

    # Lane detector (debug_mode controlled by launch argument)
    lane_detector = Node(
        package='turtlebot3_lane_follower',
        executable='lane_detector_node',
        name='lane_detector_node',
        output='screen',
        parameters=[
            detection_params,
            {'debug_mode': LaunchConfiguration('debug_mode')},
        ],
    )

    # Lane controller
    lane_controller = Node(
        package='turtlebot3_lane_follower',
        executable='lane_controller_node',
        name='lane_controller_node',
        output='screen',
        parameters=[controller_params],
    )

    return LaunchDescription([
        debug_arg,
        lane_detector,
        lane_controller,
    ])
