import numpy as np
from math import sqrt, atan2, pi

class PathFollowingEnv:
    def __init__(self):
        self.observation_shape = 5
        self.num_actions = 2

        self.observation_space = type('obj', (object,), {'n': self.observation_shape})()
        self.action_space = type('obj', (object,), {'n': self.num_actions})()
        self.n_active_features = 1

        

        self.max_linear_vel = 0.5
        self.max_angular_vel = 1.0

        self.dt = 0.1
        self.goal_threshold = 0.4
        self.collision_threshold = 0.3
        self.max_cross_track_error = 1.5

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0

        self.path = []
        self.current_waypoint_idx = 0
        self.obstacles = []

        self.steps = 0
        self.max_steps = 300

        self.prev_distance = 0.0

    def set_path(self, path):
        self.path = path
        self.current_waypoint_idx = 0

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def set_robot_pose(self, x, y, yaw):
        self.robot_x = x
        self.robot_y = y
        self.robot_yaw = yaw

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.steps = 0
        self.current_waypoint_idx = 0
        self.linear_vel = 0.0
        self.angular_vel = 0.0

        if len(self.path) > 0:
            target = self.path[0]
            self.prev_distance = sqrt(
                (target[0] - self.robot_x)**2 +
                (target[1] - self.robot_y)**2
            )

        state = self._get_state()
        info = {}

        return state, info

    def step(self, action):
        self.steps += 1

        action = action.cpu().numpy().flatten()
        self.linear_vel = float(action[0]) * self.max_linear_vel
        self.angular_vel = float(action[1]) * self.max_angular_vel

        self.robot_yaw += self.angular_vel * self.dt
        self.robot_yaw = self._normalize_angle(self.robot_yaw)
        self.robot_x += self.linear_vel * np.cos(self.robot_yaw) * self.dt
        self.robot_y += self.linear_vel * np.sin(self.robot_yaw) * self.dt

        reward, terminated, truncated = self._calculate_reward()

        state = self._get_state()
        info = {
            'waypoint_idx': self.current_waypoint_idx,
            'robot_pose': (self.robot_x, self.robot_y, self.robot_yaw)
        }

        return state, reward, terminated, truncated, info

    def _get_state(self):
        if len(self.path) == 0:
            return np.zeros(self.observation_shape, dtype=np.float32)

        if self.current_waypoint_idx >= len(self.path):
            self.current_waypoint_idx = len(self.path) - 1

        target = self.path[self.current_waypoint_idx]

        cross_track_error = self._calculate_cross_track_error()

        heading_error = self._calculate_heading_error()

        distance_to_goal = sqrt(
            (target[0] - self.robot_x)**2 +
            (target[1] - self.robot_y)**2
        )

        state = np.array([
            np.clip(cross_track_error / self.max_cross_track_error, -1, 1),
            heading_error / pi,
            np.clip(distance_to_goal / 5.0, 0, 1),
            self.linear_vel / self.max_linear_vel,
            self.angular_vel / self.max_angular_vel
        ], dtype=np.float32)

        return state

    def _calculate_cross_track_error(self):
        if len(self.path) < 2 or self.current_waypoint_idx == 0:
            if len(self.path) > 0:
                target = self.path[self.current_waypoint_idx]
                return sqrt((target[0] - self.robot_x)**2 + (target[1] - self.robot_y)**2)
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
        proj_length = np.dot(robot - p1, line_unitvec)
        proj_length = np.clip(proj_length, 0, line_len)

        closest_point = p1 + proj_length * line_unitvec
        cross_track_error = np.linalg.norm(robot - closest_point)

        return cross_track_error

    def _calculate_heading_error(self):
        if len(self.path) == 0:
            return 0.0

        target = self.path[self.current_waypoint_idx]

        desired_yaw = atan2(
            target[1] - self.robot_y,
            target[0] - self.robot_x
        )

        heading_error = desired_yaw - self.robot_yaw
        heading_error = self._normalize_angle(heading_error)

        return heading_error

    def _calculate_reward(self):
        terminated = False
        truncated = False
        reward = 0.0

        if len(self.path) == 0:
            return 0.0, True, False

        target = self.path[self.current_waypoint_idx]
        current_distance = sqrt(
            (target[0] - self.robot_x)**2 +
            (target[1] - self.robot_y)**2
        )

        progress = self.prev_distance - current_distance
        reward += progress * 10.0
        self.prev_distance = current_distance

        cross_track_error = self._calculate_cross_track_error()
        reward -= cross_track_error * 0.5

        heading_error = abs(self._calculate_heading_error())
        reward -= heading_error * 0.3

        if abs(self.angular_vel) > 0.5 and heading_error < 0.3:
            reward -= abs(self.angular_vel) * 0.5

        if self.linear_vel > 0.1:
            reward += 0.1
        elif self.linear_vel < 0:
            reward -= 0.2

        if current_distance < self.goal_threshold:
            reward += 10.0
            self.current_waypoint_idx += 1

            if self.current_waypoint_idx < len(self.path):
                next_target = self.path[self.current_waypoint_idx]
                self.prev_distance = sqrt(
                    (next_target[0] - self.robot_x)**2 +
                    (next_target[1] - self.robot_y)**2
                )

            if self.current_waypoint_idx >= len(self.path):
                reward += 100.0
                terminated = True

        if cross_track_error > self.max_cross_track_error:
            reward -= 50.0
            terminated = True

        for obs in self.obstacles:
            obs_x, obs_y, obs_radius = obs
            dist = sqrt((obs_x - self.robot_x)**2 + (obs_y - self.robot_y)**2)
            if dist < (obs_radius + self.collision_threshold):
                reward -= 100.0
                terminated = True
                break

        if self.steps >= self.max_steps:
            truncated = True

        return reward, terminated, truncated

    def _normalize_angle(self, angle):
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle

def main():
    env = PathFollowingEnv()

    path = [(1, 0), (2, 0), (3, 0)]
    env.set_path(path)
    env.set_robot_pose(0, 0, 0)

    state, info = env.reset()
    print(f"Initial state: {state}")

    import torch
    for i in range(50):
        action = torch.tensor([[0.8, 0.0]])
        state, reward, term, trunc, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, wp_idx={info['waypoint_idx']}")
        if term or trunc:
            print("Episode ended!")
            break

if __name__ == '__main__':
    main()