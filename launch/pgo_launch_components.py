from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import os
from ament_index_python.packages import get_package_share_directory

PACKAGE_NAME = "pgo_with_visual"

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory(PACKAGE_NAME),
        "config",
        "settings.yaml",
    )

    container = ComposableNodeContainer(
        name="pogolm_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        composable_node_descriptions=[
            ComposableNode(
                package=PACKAGE_NAME,
                plugin="pose_graph_np::PoseGraphComponent",
                name="pose_graph",
                parameters=[config],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            ComposableNode(
                package=PACKAGE_NAME,
                plugin="loop_detector_np::LoopDetectorComponent",
                name="loop_detector",
                parameters=[config],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            ComposableNode(
                package=PACKAGE_NAME,
                plugin="map_module_np::MapModuleComponent",
                name="map_module",
                parameters=[config],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
    )

    return LaunchDescription([container])