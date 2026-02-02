import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from pathlib import Path

def generate_launch_description():
    robot_desc_dir = get_package_share_directory('robot_description')
    map_pub_dir = get_package_share_directory('map_publisher')

    world = os.path.join(robot_desc_dir, 'world', 'depot.sdf')
    urdf_file = os.path.join(robot_desc_dir, 'src', 'description', 'robot.urdf')
    rviz_config_file = os.path.join(robot_desc_dir, 'rviz', 'config.rviz')
    gz_bridge_config = os.path.join(robot_desc_dir, 'config', 'gz_bridge.yaml')

    map_yaml_file = os.path.join(map_pub_dir, 'maps', 'my_map.yaml')
    amcl_params_file = os.path.join(map_pub_dir, 'config', 'amcl_params.yaml')

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=':'.join([
            os.path.join(robot_desc_dir, 'world'),
            str(Path(robot_desc_dir).parent.resolve())
        ])
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py',
            )
        ),
        launch_arguments={'gz_args': ['-r -v 4 ', world]}.items(),
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': gz_bridge_config,
            'qos_overrides./tf_static.publisher.durability': 'transient_local',
        }],
        output='screen'
    )

    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': robot_desc}
        ]
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'robot',
            '-topic', '/robot_description',
            '-x', '0',
            '-y', '0',
            '-z', '0.9',
        ],
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    lidar_frame_alias_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_frame_alias_tf',
        output='screen',
        arguments=[
            '--frame-id', 'base_link',
            '--child-frame-id', 'robot/base_link/rplidar_c1_sensor',
            '--x', '0', '--y', '0', '--z', '0.4',
            '--roll', '0', '--pitch', '0', '--yaw', '0'
        ],
        parameters=[{'use_sim_time': True}]
    )

    frame_id_converter_node = Node(
        package='map_publisher',
        executable='frame_id_converter',
        name='frame_id_converter_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    odom_tf_publisher = Node(
        package='map_publisher',
        executable='odom_tf_publisher',
        name='odom_tf_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    motor_command_node = Node(
        package='map_publisher',
        executable='motor_command_node',
        name='motor_command_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'yaml_filename': map_yaml_file},
            {'frame_id': 'map'},
            {'topic_name': '/map'}
        ]
    )
    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[
            amcl_params_file,
            {'use_sim_time': True}
        ],
        remappings=[
            ('/odom', '/ekf/odom')
        ]
    )

    particle_cloud_converter_node = Node(
        package='map_publisher',
        executable='particle_cloud_converter',
        name='particle_cloud_converter',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map_server',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'autostart': True},
            {'node_names': ['map_server', 'amcl']}
        ]
    )

    astar_planner_node = Node(
        package='map_publisher',
        executable='astar_planner_node',
        name='astar_planner_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='True'),

        gz_resource_path,

        gz_sim,
        bridge,

        start_robot_state_publisher_cmd,
        spawn_entity,

        lidar_frame_alias_tf,

        rviz_node,

        frame_id_converter_node,
        odom_tf_publisher,
        motor_command_node,

        map_server_node,
        amcl_node,
        particle_cloud_converter_node,
        lifecycle_manager_node,
        astar_planner_node,
    ])