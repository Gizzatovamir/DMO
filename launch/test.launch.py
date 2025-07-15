from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
import launch


def generate_launch_description():
    pkg_path = get_package_share_directory('dmo')
    # print(pkg_path)
    rviz_config = os.path.join(pkg_path, 'dmo.rviz')
    bag_file = os.path.join(pkg_path, 'bag', "single_lidar_0.db3")
    rviz = Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=[f'-d {rviz_config}']
        )
    dmo = Node(
            package='dmo',
            namespace='',
            executable='lidar_detection_node',
            name='lidar_detection_node',
        )
    
    bag = launch.actions.ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag_file],
            output='screen'
        )
    return LaunchDescription(
        [
            rviz,
            dmo,
            bag
        ]
    )