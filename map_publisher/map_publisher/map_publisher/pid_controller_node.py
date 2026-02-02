import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan
from std_srvs.srv import Empty

def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))

class PID:
    def __init__(self, kp, ki, kd, i_limit=1.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.i_limit = abs(float(i_limit))
        self.integral = 0.0
        self.prev_err = 0.0
        self.init = False

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0
        self.init = False

    def step(self, err, dt):
        if dt <= 0.0:
            return 0.0
        if not self.init:
            self.prev_err = err
            self.init = True

        self.integral += err * dt
        self.integral = max(-self.i_limit, min(self.integral, self.i_limit))

        derr = (err - self.prev_err) / dt
        self.prev_err = err

        return self.kp * err + self.ki * self.integral + self.kd * derr

class PIDPathFollowerNode(Node):

    def __init__(self):
        super().__init__('pid_path_follower_node')


        self.declare_parameter('dt', 0.05)
        self.declare_parameter('waypoint_threshold', 0.3)

        self.declare_parameter('kp_lin', 0.8)
        self.declare_parameter('ki_lin', 0.0)
        self.declare_parameter('kd_lin', 0.1)

        self.declare_parameter('kp_ang', 2.5)
        self.declare_parameter('ki_ang', 0.0)
        self.declare_parameter('kd_ang', 0.2)

        self.declare_parameter('v_max', 0.5)
        self.declare_parameter('w_max', 1.0)
        self.declare_parameter('heading_slowdown_rad', 0.6)

        self.dt = float(self.get_parameter('dt').value)
        self.wp_thresh = float(self.get_parameter('waypoint_threshold').value)
        self.v_max = float(self.get_parameter('v_max').value)
        self.w_max = float(self.get_parameter('w_max').value)
        self.heading_slow = float(self.get_parameter('heading_slowdown_rad').value)

        self.pid_lin = PID(
            self.get_parameter('kp_lin').value,
            self.get_parameter('ki_lin').value,
            self.get_parameter('kd_lin').value,
            i_limit=0.7
        )
        self.pid_ang = PID(
            self.get_parameter('kp_ang').value,
            self.get_parameter('ki_ang').value,
            self.get_parameter('kd_ang').value,
            i_limit=1.0
        )

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.have_pose = False

        self.path = []
        self.wp_idx = 0
        self.is_following = False

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )

        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.plan_client = self.create_client(GetPlan, '/plan_path')

        self.start_service = self.create_service(
            Empty, '/start_pid_following', self.start_following_callback
        )
        self.stop_service = self.create_service(
            Empty, '/stop_pid_following', self.stop_following_callback
        )

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f'PID Path Follower started | dt={self.dt} | '
            f'v_max={self.v_max} | w_max={self.w_max}'
        )
        self.get_logger().info('Waiting for path on /plan topic...')

    def pose_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        self.have_pose = True

    def path_callback(self, msg):
        self.path = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            self.path.append((x, y))

        self.wp_idx = 0
        self.pid_lin.reset()
        self.pid_ang.reset()

        self.get_logger().info(f'Received path with {len(self.path)} waypoints')

        self.is_following = True

        cmd = Twist()
        cmd.linear.x = 0.1
        self.cmd_pub.publish(cmd)
        self.get_logger().info('PID path following started!')

    def goal_callback(self, msg):
        self.get_logger().info(
            f'Goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})'
        )

        if not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Plan service not available!')
            return

        request = GetPlan.Request()
        request.goal = msg

        future = self.plan_client.call_async(request)
        future.add_done_callback(self.plan_response_callback)

    def plan_response_callback(self, future):
        try:
            response = future.result()
            if len(response.plan.poses) > 0:
                self.get_logger().info('Path received from A* planner')
            else:
                self.get_logger().error('No path returned from planner')
        except Exception as e:
            self.get_logger().error(f'Planning failed: {e}')

    def start_following_callback(self, request, response):
        self.is_following = True
        self.get_logger().info('Started PID path following')
        return response

    def stop_following_callback(self, request, response):
        self.is_following = False
        self.stop_robot()
        self.get_logger().info('Stopped PID path following')
        return response

    def control_loop(self):
        if not self.is_following:
            return

        if not self.have_pose:
            self.get_logger().warn('No pose yet!', throttle_duration_sec=2.0)
            return

        if not self.path:
            return

        if self.wp_idx >= len(self.path):
            self.get_logger().info('Goal reached!')
            self.is_following = False
            self.stop_robot()
            return

        wx, wy = self.path[self.wp_idx]
        dx = wx - self.x
        dy = wy - self.y
        dist = math.hypot(dx, dy)

        if dist < self.wp_thresh:
            self.wp_idx += 1
            self.pid_lin.reset()
            self.pid_ang.reset()
            self.get_logger().info(f'Waypoint {self.wp_idx}/{len(self.path)} reached')
            return

        target_yaw = math.atan2(dy, dx)
        heading_err = wrap_to_pi(target_yaw - self.yaw)

        w_cmd = self.pid_ang.step(heading_err, self.dt)
        v_cmd = self.pid_lin.step(dist, self.dt)

        if abs(heading_err) > self.heading_slow:
            v_cmd *= 0.2

        v_cmd = max(-self.v_max, min(self.v_max, v_cmd))
        w_cmd = max(-self.w_max, min(self.w_max, w_cmd))

        cmd = Twist()
        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(w_cmd)
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = PIDPathFollowerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
