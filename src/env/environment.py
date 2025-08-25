"""
环境模块：工作域、网格、障碍和环境密度生成
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import random


class Workspace:
    """工作域类"""

    def __init__(self, domain, grid_size, visit_radius):
        self.domain = domain  # [x_min, x_max, y_min, y_max]
        self.grid_size = grid_size  # [nx, ny]
        self.visit_radius = visit_radius

        # 生成网格
        self.grid_centers = self._generate_grid()
        self.M = len(self.grid_centers)

    def _generate_grid(self):
        """生成规则网格"""
        x_min, x_max, y_min, y_max = self.domain
        nx, ny = self.grid_size

        x_coords = np.linspace(x_min, x_max, nx)
        y_coords = np.linspace(y_min, y_max, ny)

        grid_centers = []
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                grid_centers.append([x, y])

        return np.array(grid_centers)

    def get_grid_index(self, position):
        """获取位置对应的网格索引"""
        distances = cdist([position], self.grid_centers)[0]
        return np.argmin(distances)

    def is_visited(self, uav_position, grid_center):
        """判断UAV是否访问了网格"""
        distance = np.linalg.norm(uav_position - grid_center)
        return distance <= self.visit_radius


class Obstacle:
    """障碍类"""

    def __init__(self, obstacle_type, params, safety_distance):
        self.type = obstacle_type
        self.params = params
        self.safety_distance = safety_distance
        self.geometry = self._create_geometry()

    def _create_geometry(self):
        """创建障碍几何形状"""
        if self.type == "circle":
            center = self.params["center"]
            radius = self.params["radius"]
            return Point(center).buffer(radius + self.safety_distance)
        elif self.type == "rectangle":
            x, y, w, h = self.params["x"], self.params["y"], self.params["width"], self.params["height"]
            return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]).buffer(self.safety_distance)
        else:
            raise ValueError(f"Unknown obstacle type: {self.type}")

    def contains(self, point):
        """检查点是否在障碍内"""
        return self.geometry.contains(Point(point))

    def distance_to(self, point):
        """计算点到障碍的距离"""
        return self.geometry.distance(Point(point))


class DynamicObstacle(Obstacle):
    """动态障碍类"""

    def __init__(self, obstacle_type, params, safety_distance, velocity, trajectory):
        super().__init__(obstacle_type, params, safety_distance)
        self.velocity = velocity
        self.trajectory = trajectory
        self.current_time = 0

    def update_position(self, dt):
        """更新动态障碍位置"""
        self.current_time += dt
        # 简化的线性轨迹更新
        if self.current_time < len(self.trajectory):
            new_center = self.trajectory[int(self.current_time)]
            if self.type == "circle":
                self.params["center"] = new_center
                self.geometry = self._create_geometry()


class Environment:
    """环境类"""

    def __init__(self, workspace, density_type, obstacles=None):
        self.workspace = workspace
        self.density_type = density_type
        self.obstacles = obstacles or []
        self.density_map = self._generate_density()

    def _generate_density(self):
        """生成环境密度分布"""
        grid_centers = self.workspace.grid_centers

        if self.density_type["type"] == "uniform":
            return np.ones(len(grid_centers))

        elif self.density_type["type"] == "gaussian_mixture":
            density = np.zeros(len(grid_centers))
            centers = self.density_type["centers"]
            stds = self.density_type["stds"]
            weights = self.density_type["weights"]

            for center, std, weight in zip(centers, stds, weights):
                distances = cdist(grid_centers, [center])
                density += weight * np.exp(-distances.flatten() ** 2 / (2 * std ** 2))

            return density

        elif self.density_type["type"] == "corridor":
            density = np.ones(len(grid_centers))
            start = self.density_type["start"]
            end = self.density_type["end"]
            width = self.density_type["width"]
            corridor_density = self.density_type["density"]

            # 创建走廊
            corridor_line = LineString([start, end])
            for i, center in enumerate(grid_centers):
                distance_to_corridor = corridor_line.distance(Point(center))
                if distance_to_corridor <= width / 2:
                    density[i] = corridor_density

            return density

        else:
            raise ValueError(f"Unknown density type: {self.density_type['type']}")

    def get_density_at(self, position):
        """获取指定位置的密度值"""
        grid_index = self.workspace.get_grid_index(position)
        return self.density_map[grid_index]

    def is_free_space(self, position):
        """检查位置是否在自由空间"""
        for obstacle in self.obstacles:
            if obstacle.contains(position):
                return False
        return True

    def get_nearest_obstacle_distance(self, position):
        """获取到最近障碍的距离"""
        if not self.obstacles:
            return float('inf')

        distances = [obstacle.distance_to(position) for obstacle in self.obstacles]
        return min(distances)

    def update_dynamic_obstacles(self, dt):
        """更新动态障碍"""
        for obstacle in self.obstacles:
            if isinstance(obstacle, DynamicObstacle):
                obstacle.update_position(dt)

    def visualize(self, ax=None):
        """可视化环境"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # 绘制网格
        grid_centers = self.workspace.grid_centers
        ax.scatter(grid_centers[:, 0], grid_centers[:, 1],
                   c=self.density_map, cmap='viridis', alpha=0.6, s=20)

        # 绘制障碍
        for obstacle in self.obstacles:
            if obstacle.type == "circle":
                center = obstacle.params["center"]
                radius = obstacle.params["radius"]
                circle = plt.Circle(center, radius, color='red', alpha=0.7)
                ax.add_patch(circle)
            elif obstacle.type == "rectangle":
                x, y, w, h = obstacle.params["x"], obstacle.params["y"], obstacle.params["width"], obstacle.params["height"]
                rect = plt.Rectangle((x, y), w, h, color='red', alpha=0.7)
                ax.add_patch(rect)

        ax.set_xlim(self.workspace.domain[0], self.workspace.domain[1])
        ax.set_ylim(self.workspace.domain[2], self.workspace.domain[3])
        ax.set_aspect('equal')
        ax.set_title('Environment Visualization')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        return ax


def create_environment_from_config(config):
    """从配置创建环境"""
    workspace = Workspace(
        domain=config['workspace']['domain'],
        grid_size=config['workspace']['grid_size'],
        visit_radius=config['workspace']['visit_radius']
    )

    # 创建静态障碍
    obstacles = []
    static_config = config['obstacles']['static']
    count = random.randint(static_config['count'][0], static_config['count'][1])

    for _ in range(count):
        if random.choice(static_config['types']) == "circle":
            center = [
                random.uniform(workspace.domain[0] + 50, workspace.domain[1] - 50),
                random.uniform(workspace.domain[2] + 50, workspace.domain[3] - 50)
            ]
            radius = random.uniform(20, 40)
            obstacle = Obstacle("circle", {"center": center, "radius": radius},
                                static_config['safety_distance'])
        else:
            x = random.uniform(workspace.domain[0] + 50, workspace.domain[1] - 100)
            y = random.uniform(workspace.domain[2] + 50, workspace.domain[3] - 100)
            w = random.uniform(30, 60)
            h = random.uniform(30, 60)
            obstacle = Obstacle("rectangle", {"x": x, "y": y, "width": w, "height": h},
                                static_config['safety_distance'])
        obstacles.append(obstacle)

    # 创建动态障碍
    dynamic_config = config['obstacles']['dynamic']
    count = random.randint(dynamic_config['count'][0], dynamic_config['count'][1])

    for _ in range(count):
        # 简化的线性轨迹
        start = [
            random.uniform(workspace.domain[0] + 50, workspace.domain[1] - 50),
            random.uniform(workspace.domain[2] + 50, workspace.domain[3] - 50)
        ]
        end = [
            random.uniform(workspace.domain[0] + 50, workspace.domain[1] - 50),
            random.uniform(workspace.domain[2] + 50, workspace.domain[3] - 50)
        ]

        trajectory = []
        steps = 100
        for i in range(steps):
            t = i / steps
            pos = [start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])]
            trajectory.append(pos)

        center = start
        radius = random.uniform(15, 25)
        obstacle = DynamicObstacle("circle", {"center": center, "radius": radius},
                                   dynamic_config['safety_distance'],
                                   dynamic_config['velocity'], trajectory)
        obstacles.append(obstacle)

    # 创建环境
    density_config = config['environment']['density_types'][config['tasks']['T1']['density']]
    environment = Environment(workspace, density_config, obstacles)

    return environment




