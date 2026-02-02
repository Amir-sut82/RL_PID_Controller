import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
import yaml
from PIL import Image
import numpy as np
import os

class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher_node')

        self.declare_parameter('map_file', '')
        self.declare_parameter('yaml_file', '')

        self.map_publisher = self.create_publisher(
            OccupancyGrid,
            '/map',
            10
        )

        self.timer = self.create_timer(1.0, self.publish_map)

        self.occupancy_grid = self.load_map()

        self.get_logger().info('Map Publisher Node Started!')

    def load_map(self):

        yaml_file = self.get_parameter('yaml_file').value

        if not yaml_file or not os.path.exists(yaml_file):
            self.get_logger().error(f'YAML file not found: {yaml_file}')
            return None

        with open(yaml_file, 'r') as f:
            map_metadata = yaml.safe_load(f)

        yaml_dir = os.path.dirname(yaml_file)
        pgm_file = os.path.join(yaml_dir, map_metadata['image'])

        if not os.path.exists(pgm_file):
            self.get_logger().error(f'PGM file not found: {pgm_file}')
            return None

        image = Image.open(pgm_file)
        image_array = np.array(image)

        occupancy_grid = OccupancyGrid()

        occupancy_grid.header.frame_id = 'map'

        occupancy_grid.info.resolution = float(map_metadata['resolution'])
        occupancy_grid.info.width = image_array.shape[1]
        occupancy_grid.info.height = image_array.shape[0]

        origin = map_metadata['origin']
        occupancy_grid.info.origin.position.x = float(origin[0])
        occupancy_grid.info.origin.position.y = float(origin[1])
        occupancy_grid.info.origin.position.z = 0.0

        occupancy_grid.info.origin.orientation.x = 0.0
        occupancy_grid.info.origin.orientation.y = 0.0
        occupancy_grid.info.origin.orientation.z = np.sin(origin[2] / 2.0)
        occupancy_grid.info.origin.orientation.w = np.cos(origin[2] / 2.0)

        occupied_thresh = map_metadata.get('occupied_thresh', 0.65)
        free_thresh = map_metadata.get('free_thresh', 0.25)

        occupancy_data = []

        image_array = np.flipud(image_array)

        for pixel in image_array.flatten():
            normalized = pixel / 255.0

            if normalized > free_thresh:
                occupancy_data.append(0)
            elif normalized < occupied_thresh:
                occupancy_data.append(100)
            else:
                occupancy_data.append(-1)

        occupancy_grid.data = occupancy_data

        self.get_logger().info(f'Map loaded successfully!')
        self.get_logger().info(f'  Resolution: {occupancy_grid.info.resolution}')
        self.get_logger().info(f'  Width: {occupancy_grid.info.width}')
        self.get_logger().info(f'  Height: {occupancy_grid.info.height}')
        self.get_logger().info(f'  Origin: ({occupancy_grid.info.origin.position.x}, '
                              f'{occupancy_grid.info.origin.position.y})')

        return occupancy_grid

    def publish_map(self):
        if self.occupancy_grid is not None:
            self.occupancy_grid.header.stamp = self.get_clock().now().to_msg()

            self.map_publisher.publish(self.occupancy_grid)
            self.get_logger().debug('Map published')

def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()