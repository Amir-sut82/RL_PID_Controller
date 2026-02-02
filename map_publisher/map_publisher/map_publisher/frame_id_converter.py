import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSPresetProfiles
import copy

class FrameIdConverter(Node):
    def __init__(self):
        super().__init__('frame_id_converter_node')

        self.pub_ = self.create_publisher(
            LaserScan,
            '/scan',
            QoSPresetProfiles.SENSOR_DATA.value
        )

        self.sub_ = self.create_subscription(
            LaserScan,
            '/scan_raw',
            self.scan_callback,
            QoSPresetProfiles.SENSOR_DATA.value
        )

        self.get_logger().info(
            'FrameIdConverter started. Listening to /scan_raw, publishing to /scan with frame_id=rplidar_c1'
        )

    def scan_callback(self, msg: LaserScan):
        new_msg = copy.deepcopy(msg)
        new_msg.header.frame_id = 'rplidar_c1'
        self.pub_.publish(new_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FrameIdConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
