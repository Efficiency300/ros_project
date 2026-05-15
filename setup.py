import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'turtlebot3_lane_follower'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'param'),
            glob('param/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.com',
    description='Yellow dual-lane follower for TurtleBot3 using OpenCV and ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detector_node = turtlebot3_lane_follower.lane_detector_node:main',
            'lane_controller_node = turtlebot3_lane_follower.lane_controller_node:main',
            'perspective_calibrator = turtlebot3_lane_follower.perspective_calibrator:main',
        ],
    },
)
