import os
import glob
from setuptools import find_packages, setup

package_name = 'ur5_dual_robot_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='raphael',
    maintainer_email='r.ullrich.1@campus.tu-berlin.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dual_arm_teleop_node = ur5_dual_robot_teleop.dual_arm_teleop_node:main',
            'hand_tracker_node = ur5_dual_robot_teleop.hand_tracker_node:main',
            'workspace_visualizer = ur5_dual_robot_teleop.workspace_visualizer_node:main',
            'gripper_node = ur5_dual_robot_teleop.gripper_node:main',
        ],
    },
)
