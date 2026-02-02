import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav2_msgs.msg import ParticleCloud
from geometry_msgs.msg import PoseArray, Pose

class ParticleCloudConverter(Node):
    def __init__(self):
        super().__init__('particle_cloud_converter')

        amcl_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        rviz_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            ParticleCloud,
            '/particle_cloud',
            self.callback,
            amcl_qos
        )

        self.publisher = self.create_publisher(
            PoseArray,
            '/particle_cloud_poses',
            rviz_qos
        )

        self.get_logger().info('Particle Cloud Converter Started - Listening on /particle_cloud, Publishing to /particle_cloud_poses')

    def callback(self, msg):
        pose_array = PoseArray()
        pose_array.header = msg.header

        if not pose_array.header.frame_id:
            pose_array.header.frame_id = 'map'

        for particle in msg.particles:
            pose = Pose()
            pose.position.x = particle.pose.position.x
            pose.position.y = particle.pose.position.y
            pose.position.z = particle.pose.position.z
            pose.orientation = particle.pose.orientation
            pose_array.poses.append(pose)

        self.publisher.publish(pose_array)
        self.get_logger().debug(f'Published {len(pose_array.poses)} particle poses')

def main(args=None):
    rclpy.init(args=args)
    node = ParticleCloudConverter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()