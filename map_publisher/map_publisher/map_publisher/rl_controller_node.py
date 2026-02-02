import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan
from std_srvs.srv import Empty

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt, atan2, pi
import os

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc_action = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.leaky_relu(self.fc3(x), 0.01)
        x = F.leaky_relu(self.fc4(x), 0.01)
        return torch.tanh(self.fc_action(x))

class RLControllerNode(Node):
    def __init__(self):
        super().__init__('rl_controller_node')

        default_model_path = '/home/amirhossein/ros2_ws/src/robotic_course/map_publisher/map_publisher/map_publisher/models/ddpg_final.pth'

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('max_linear_vel', 0.5)
        self.declare_parameter('max_angular_vel', 1.0)
        self.declare_parameter('goal_threshold', 0.4)
        self.declare_parameter('control_rate', 10.0)

        self.model_path = self.get_parameter('model_path').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.goal_threshold = self.get_parameter('goal_threshold').value
        control_rate = self.get_parameter('control_rate').value

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.path = []
        self.current_waypoint_idx = 0
        self.is_following = False
        self.have_pose = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = 5
        self.action_dim = 2
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self._load_model()

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

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
            Empty, '/start_following', self.start_following_callback
        )
        self.stop_service = self.create_service(
            Empty, '/stop_following', self.stop_following_callback
        )

        self.control_timer = self.create_timer(1.0 / control_rate, self.control_loop)

        self.get_logger().info('='*50)
        self.get_logger().info('RL Controller Node Started!')
        self.get_logger().info(f'Model: {self.model_path}')
        self.get_logger().info('Waiting for path on /plan topic...')
        self.get_logger().info('='*50)

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.actor.load_state_dict(checkpoint['actor'])
                self.actor.eval()
                self.get_logger().info('Model loaded successfully!')
            except Exception as e:
                self.get_logger().error(f'Failed to load model: {e}')
                self.get_logger().warn('Using untrained model!')
        else:
            self.get_logger().error(f'Model NOT FOUND at: {self.model_path}')
            self.get_logger().warn('Please train first: python3 rl_trainer.py --episodes 1000')

    def pose_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = atan2(siny_cosp, cosy_cosp)
        self.have_pose = True

    def path_callback(self, msg):
        self.path = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            self.path.append((x, y))

        self.current_waypoint_idx = 0
        self.get_logger().info(f'Received path with {len(self.path)} waypoints')

        self.is_following = True

        cmd = Twist()
        cmd.linear.x = 0.1
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('RL path following started!')

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
                self.get_logger().error('No path returned!')
        except Exception as e:
            self.get_logger().error(f'Planning failed: {e}')

    def start_following_callback(self, request, response):
        self.is_following = True
        self.get_logger().info('Started path following')
        return response

    def stop_following_callback(self, request, response):
        self.is_following = False
        self._stop_robot()
        self.get_logger().info('Stopped path following')
        return response

    def control_loop(self):
        if not self.is_following or len(self.path) == 0:
            return

        if not self.have_pose:
            return

        if self.current_waypoint_idx >= len(self.path):
            self.get_logger().info('Goal reached!')
            self.is_following = False
            self._stop_robot()
            return

        target = self.path[self.current_waypoint_idx]
        distance = sqrt(
            (target[0] - self.robot_x)**2 +
            (target[1] - self.robot_y)**2
        )

        if distance < self.goal_threshold:
            self.current_waypoint_idx += 1
            self.get_logger().info(
                f'Waypoint {self.current_waypoint_idx}/{len(self.path)} reached'
            )
            return

        state = self._get_state()
        action = self._get_action(state)
        self._apply_action(action)

    def _get_state(self):
        target = self.path[self.current_waypoint_idx]

        cross_track_error = self._calculate_cross_track_error()

        desired_yaw = atan2(
            target[1] - self.robot_y,
            target[0] - self.robot_x
        )
        heading_error = self._normalize_angle(desired_yaw - self.robot_yaw)

        distance = sqrt(
            (target[0] - self.robot_x)**2 +
            (target[1] - self.robot_y)**2
        )

        state = np.array([
            np.clip(cross_track_error / 1.5, -1, 1),
            heading_error / pi,
            np.clip(distance / 5.0, 0, 1),
            self.linear_vel / self.max_linear_vel,
            self.angular_vel / self.max_angular_vel
        ], dtype=np.float32)

        return state

    def _calculate_cross_track_error(self):
        if len(self.path) < 2 or self.current_waypoint_idx == 0:
            if len(self.path) > 0:
                target = self.path[self.current_waypoint_idx]
                return sqrt(
                    (target[0] - self.robot_x)**2 +
                    (target[1] - self.robot_y)**2
                )
            return 0.0

        prev_idx = max(0, self.current_waypoint_idx - 1)
        p1 = np.array(self.path[prev_idx])
        p2 = np.array(self.path[self.current_waypoint_idx])
        robot = np.array([self.robot_x, self.robot_y])

        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len < 0.001:
            return np.linalg.norm(robot - p1)

        line_unitvec = line_vec / line_len
        proj_length = np.clip(np.dot(robot - p1, line_unitvec), 0, line_len)
        closest_point = p1 + proj_length * line_unitvec

        return np.linalg.norm(robot - closest_point)

    def _normalize_angle(self, angle):
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle

    def _get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()

        return action

    def _apply_action(self, action):
        cmd = Twist()

        linear = float(action[0]) * self.max_linear_vel
        angular = float(action[1]) * self.max_angular_vel

        if abs(angular) > 0.5 and abs(linear) < 0.15:
            linear = 0.2
            angular = np.clip(angular, -0.5, 0.5)

        cmd.linear.x = linear
        cmd.angular.z = angular

        self.linear_vel = cmd.linear.x
        self.angular_vel = cmd.angular.z

        self.cmd_vel_pub.publish(cmd)

    def _stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        self.linear_vel = 0.0
        self.angular_vel = 0.0

def main(args=None):
    rclpy.init(args=args)
    node = RLControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()