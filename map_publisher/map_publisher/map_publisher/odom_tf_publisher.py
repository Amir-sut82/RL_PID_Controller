import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class OdomTfPublisher(Node):
    def __init__(self):
        super().__init__('odom_tf_publisher')


        self.declare_parameter('odom_topic', '/wheel_encoder/odom')
        odom_topic = self.get_parameter('odom_topic').value

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            sensor_qos
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(f'Odom TF Publisher started')
        self.get_logger().info(f'Listening to: {odom_topic}')
        self.get_logger().info(f'Publishing TF: odom → base_link')

    def odom_callback(self, msg):
        tf_msg = TransformStamped()

        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'base_link'

        tf_msg.transform.translation.x = msg.pose.pose.position.x
        tf_msg.transform.translation.y = msg.pose.pose.position.y
        tf_msg.transform.translation.z = msg.pose.pose.position.z

        tf_msg.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tf_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OdomTfPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
