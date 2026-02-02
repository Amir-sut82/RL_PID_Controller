import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import numpy as np
import heapq
from math import sqrt

class AStarPlannerService(Node):
    def __init__(self):
        super().__init__('astar_planner_node')

        self.declare_parameter('robot_length', 0.4)
        self.declare_parameter('robot_width', 0.46)
        self.declare_parameter('safety_margin', 1.15)

        robot_length = self.get_parameter('robot_length').value
        robot_width = self.get_parameter('robot_width').value
        safety_margin = self.get_parameter('safety_margin').value


        half_length = robot_length / 2.0
        half_width = robot_width / 2.0
        diagonal_radius = sqrt(half_length**2 + half_width**2)
        self.robot_radius = safety_margin * diagonal_radius

        self.get_logger().info(f'Robot footprint: {robot_length}m x {robot_width}m')
        self.get_logger().info(f'Diagonal radius: {diagonal_radius:.3f}m')
        self.get_logger().info(f'Total inflation radius (with margin): {self.robot_radius:.3f}m')

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )

        self.plan_service = self.create_service(
            GetPlan,
            '/plan_path',
            self.plan_callback
        )

        self.path_publisher = self.create_publisher(
            Path,
            '/plan',
            10
        )

        self.map_data = None
        self.inflated_map = None
        self.current_pose = None
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0

        self.get_logger().info('A* Planner Service Started!')
        self.get_logger().info('Service: /plan_path')
        self.get_logger().info('Path topic: /plan')

    def map_callback(self, msg):
        self.map_data = msg
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

        self.get_logger().info(f'Map received! Size: {msg.info.width}x{msg.info.height}')
        self.get_logger().info(f'Resolution: {msg.info.resolution}m/cell')
        self.get_logger().info(f'Origin: ({self.map_origin_x:.2f}, {self.map_origin_y:.2f})')

        self.create_inflated_map()

    def create_inflated_map(self):
        if self.map_data is None:
            return

        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution

        original_map = np.array(self.map_data.data).reshape((height, width))

        inflation_cells = int(np.ceil(self.robot_radius / resolution))

        self.get_logger().info(f'Inflating obstacles by {inflation_cells} cells ({self.robot_radius:.3f}m)')

        self.inflated_map = original_map.copy()

        obstacle_coords = np.where(original_map > 50)

        unknown_coords = np.where(original_map < 0)

        self.get_logger().info(f'Found {len(obstacle_coords[0])} obstacle cells')
        self.get_logger().info(f'Found {len(unknown_coords[0])} unknown cells')

        for oy, ox in zip(obstacle_coords[0], obstacle_coords[1]):
            for dy in range(-inflation_cells, inflation_cells + 1):
                for dx in range(-inflation_cells, inflation_cells + 1):
                    if dx*dx + dy*dy <= inflation_cells*inflation_cells:
                        ny, nx = oy + dy, ox + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if self.inflated_map[ny, nx] < 50:
                                self.inflated_map[ny, nx] = 99

        for oy, ox in zip(unknown_coords[0], unknown_coords[1]):
            for dy in range(-inflation_cells, inflation_cells + 1):
                for dx in range(-inflation_cells, inflation_cells + 1):
                    if dx*dx + dy*dy <= inflation_cells*inflation_cells:
                        ny, nx = oy + dy, ox + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if self.inflated_map[ny, nx] == 0:
                                self.inflated_map[ny, nx] = 50

        self.get_logger().info('Inflation complete!')

    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.get_logger().debug(f'Pose updated: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f})')

    def plan_callback(self, request, response):
        self.get_logger().info('Plan request received!')

        if self.map_data is None or self.inflated_map is None:
            self.get_logger().error('Map not received yet!')
            response.plan = Path()
            return response

        if self.current_pose is None:
            self.get_logger().error('Robot pose not known yet!')
            response.plan = Path()
            return response

        start_x = self.current_pose.position.x
        start_y = self.current_pose.position.y

        goal_x = request.goal.pose.position.x
        goal_y = request.goal.pose.position.y

        self.get_logger().info(f'Planning from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})')

        start_grid = self.world_to_grid(start_x, start_y)
        goal_grid = self.world_to_grid(goal_x, goal_y)

        self.get_logger().info(f'Grid coords - Start: {start_grid}, Goal: {goal_grid}')

        path_grid = self.astar(start_grid[0], start_grid[1], goal_grid[0], goal_grid[1])

        if path_grid is None:
            self.get_logger().error('No path found!')
            response.plan = Path()
            return response

        path_msg = self.create_path_message(path_grid)

        self.path_publisher.publish(path_msg)

        response.plan = path_msg

        self.get_logger().info(f'Path found with {len(path_grid)} waypoints!')
        return response

    def world_to_grid(self, world_x, world_y):
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y

        grid_x = int((world_x - origin_x) / resolution)
        grid_y = int((world_y - origin_y) / resolution)

        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y

        world_x = grid_x * resolution + origin_x + resolution / 2.0
        world_y = grid_y * resolution + origin_y + resolution / 2.0

        return (world_x, world_y)

    def is_valid(self, x, y):
        width = self.map_data.info.width
        height = self.map_data.info.height

        if x < 0 or x >= width or y < 0 or y >= height:
            return False

        cell_value = self.inflated_map[y, x]

        if cell_value >= 50:
            return False

        return True

    def heuristic(self, x1, y1, x2, y2):
        return sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def get_neighbors(self, x, y):
        neighbors = []

        directions = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, 1.414),
            (1, -1, 1.414),
            (-1, 1, 1.414),
            (-1, -1, 1.414)
        ]

        for dx, dy, cost in directions:
            new_x, new_y = x + dx, y + dy
            if self.is_valid(new_x, new_y):
                if abs(dx) == 1 and abs(dy) == 1:
                    if not self.is_valid(x + dx, y) or not self.is_valid(x, y + dy):
                        continue
                neighbors.append((new_x, new_y, cost))

        return neighbors

    def astar(self, start_x, start_y, goal_x, goal_y):

        if not self.is_valid(start_x, start_y):
            self.get_logger().error(f'Start ({start_x}, {start_y}) is not valid!')
            start_x, start_y = self.find_nearest_valid(start_x, start_y)
            if start_x is None:
                return None
            self.get_logger().info(f'Using nearest valid start: ({start_x}, {start_y})')

        if not self.is_valid(goal_x, goal_y):
            self.get_logger().error(f'Goal ({goal_x}, {goal_y}) is not valid!')
            goal_x, goal_y = self.find_nearest_valid(goal_x, goal_y)
            if goal_x is None:
                return None
            self.get_logger().info(f'Using nearest valid goal: ({goal_x}, {goal_y})')

        counter = 0
        open_set = []
        heapq.heappush(open_set, (0, counter, start_x, start_y))
        counter += 1

        g_score = {(start_x, start_y): 0}
        came_from = {}
        closed_set = set()

        while open_set:
            _, _, current_x, current_y = heapq.heappop(open_set)

            if current_x == goal_x and current_y == goal_y:
                return self.reconstruct_path(came_from, current_x, current_y)

            if (current_x, current_y) in closed_set:
                continue

            closed_set.add((current_x, current_y))

            for nx, ny, cost in self.get_neighbors(current_x, current_y):
                if (nx, ny) in closed_set:
                    continue

                tentative_g = g_score[(current_x, current_y)] + cost

                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = (current_x, current_y)
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + self.heuristic(nx, ny, goal_x, goal_y)
                    heapq.heappush(open_set, (f, counter, nx, ny))
                    counter += 1

        return None

    def find_nearest_valid(self, x, y, max_search=20):
        for radius in range(1, max_search):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:
                        nx, ny = x + dx, y + dy
                        if self.is_valid(nx, ny):
                            return nx, ny
        return None, None

    def reconstruct_path(self, came_from, current_x, current_y):
        path = [(current_x, current_y)]

        while (current_x, current_y) in came_from:
            current_x, current_y = came_from[(current_x, current_y)]
            path.append((current_x, current_y))

        path.reverse()
        return path

    def create_path_message(self, path_grid):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for grid_x, grid_y in path_grid:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = path_msg.header.stamp

            world_x, world_y = self.grid_to_world(grid_x, grid_y)
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        return path_msg

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerService()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
