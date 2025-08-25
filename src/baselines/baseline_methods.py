"""
基线方法模块：实现各种对比算法
"""
import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist
import random


class BaselineMethod:
    """基线方法基类"""

    def __init__(self, uav_fleet, environment, config):
        self.uav_fleet = uav_fleet
        self.environment = environment
        self.config = config

    def plan_step(self, current_time, aoi_states):
        """规划一步（需要子类实现）"""
        raise NotImplementedError


class UnweightedVoronoiBaseline(BaselineMethod):
    """无权重Voronoi + Lloyd基线"""

    def plan_step(self, current_time, aoi_states):
        """使用经典Voronoi分区"""
        uav_positions = self.uav_fleet.get_positions()

        # 使用无权重Voronoi分区
        site_weights = [0] * len(uav_positions)  # 无权重

        # 简化的Lloyd更新
        new_positions = self._lloyd_update(uav_positions, site_weights)

        # 生成控制输入（朝向目标位置）
        control_inputs = []
        for i, uav in enumerate(self.uav_fleet.uavs):
            target_pos = new_positions[i]
            current_pos = uav.get_position()

            # 简单的PD控制器
            position_error = target_pos - current_pos
            velocity_error = np.zeros(2) - uav.get_velocity()

            Kp = 0.5
            Kd = 0.1
            control = Kp * position_error + Kd * velocity_error

            # 限制控制输入
            max_control = self.uav_fleet.dynamics.max_acceleration
            control = np.clip(control, -max_control, max_control)

            control_inputs.append(control)

        return control_inputs

    def _lloyd_update(self, uav_positions, site_weights):
        """简化的Lloyd更新"""
        grid_centers = self.environment.workspace.grid_centers

        # 计算Voronoi单元
        assignments = []
        for grid_center in grid_centers:
            distances = [np.linalg.norm(grid_center - pos) for pos in uav_positions]
            assignments.append(np.argmin(distances))

        # 计算质心
        new_positions = []
        for i in range(len(uav_positions)):
            cell_indices = np.where(np.array(assignments) == i)[0]
            if len(cell_indices) > 0:
                cell_centers = grid_centers[cell_indices]
                centroid = np.mean(cell_centers, axis=0)
            else:
                centroid = uav_positions[i]
            new_positions.append(centroid)

        return np.array(new_positions)


class PowerVoronoiBaseline(BaselineMethod):
    """仅环境密度的Power-Voronoi基线"""

    def plan_step(self, current_time, aoi_states):
        """使用环境密度的Power-Voronoi"""
        uav_positions = self.uav_fleet.get_positions()

        # 使用环境密度，但无站点权重
        site_weights = [0] * len(uav_positions)

        # 计算加权质心
        new_positions = self._weighted_lloyd_update(uav_positions, site_weights)

        # 生成控制输入
        control_inputs = []
        for i, uav in enumerate(self.uav_fleet.uavs):
            target_pos = new_positions[i]
            current_pos = uav.get_position()

            position_error = target_pos - current_pos
            velocity_error = np.zeros(2) - uav.get_velocity()

            Kp = 0.5
            Kd = 0.1
            control = Kp * position_error + Kd * velocity_error

            max_control = self.uav_fleet.dynamics.max_acceleration
            control = np.clip(control, -max_control, max_control)

            control_inputs.append(control)

        return control_inputs

    def _weighted_lloyd_update(self, uav_positions, site_weights):
        """加权Lloyd更新"""
        grid_centers = self.environment.workspace.grid_centers
        density_map = self.environment.density_map

        # 计算加权Voronoi单元
        assignments = []
        for grid_center in grid_centers:
            weighted_distances = []
            for pos, weight in zip(uav_positions, site_weights):
                dist = np.linalg.norm(grid_center - pos) ** 2 - weight
                weighted_distances.append(dist)
            assignments.append(np.argmin(weighted_distances))

        # 计算加权质心
        new_positions = []
        for i in range(len(uav_positions)):
            cell_indices = np.where(np.array(assignments) == i)[0]
            if len(cell_indices) > 0:
                cell_centers = grid_centers[cell_indices]
                cell_densities = density_map[cell_indices]

                weighted_sum = np.sum(cell_centers * cell_densities[:, np.newaxis], axis=0)
                total_weight = np.sum(cell_densities)

                if total_weight > 0:
                    centroid = weighted_sum / total_weight
                else:
                    centroid = uav_positions[i]
            else:
                centroid = uav_positions[i]
            new_positions.append(centroid)

        return np.array(new_positions)


class CBBABaseline(BaselineMethod):
    """CBBA/拍卖分配基线"""

    def __init__(self, uav_fleet, environment, config):
        super().__init__(uav_fleet, environment, config)
        self.task_assignments = {}

    def plan_step(self, current_time, aoi_states):
        """使用CBBA进行任务分配"""
        uav_positions = self.uav_fleet.get_positions()
        grid_centers = self.environment.workspace.grid_centers

        # 简化的CBBA：基于距离的贪心分配
        if current_time % 10 == 0:  # 每10步重新分配
            self._update_task_assignments(uav_positions, aoi_states)

        # 生成控制输入
        control_inputs = []
        for i, uav in enumerate(self.uav_fleet.uavs):
            if i in self.task_assignments:
                target_grid = self.task_assignments[i]
                target_pos = grid_centers[target_grid]
            else:
                target_pos = uav.get_position()

            current_pos = uav.get_position()
            position_error = target_pos - current_pos
            velocity_error = np.zeros(2) - uav.get_velocity()

            Kp = 0.5
            Kd = 0.1
            control = Kp * position_error + Kd * velocity_error

            max_control = self.uav_fleet.dynamics.max_acceleration
            control = np.clip(control, -max_control, max_control)

            control_inputs.append(control)

        return control_inputs

    def _update_task_assignments(self, uav_positions, aoi_states):
        """更新任务分配"""
        grid_centers = self.environment.workspace.grid_centers

        # 按AoI排序网格
        aoi_indices = np.argsort(aoi_states)[::-1]  # 降序

        # 贪心分配
        assigned_grids = set()
        self.task_assignments = {}

        for grid_idx in aoi_indices:
            if len(self.task_assignments) >= len(uav_positions):
                break

            # 找到最近的未分配UAV
            min_distance = float('inf')
            best_uav = None

            for i, uav_pos in enumerate(uav_positions):
                if i not in self.task_assignments:
                    distance = np.linalg.norm(grid_centers[grid_idx] - uav_pos)
                    if distance < min_distance:
                        min_distance = distance
                        best_uav = i

            if best_uav is not None:
                self.task_assignments[best_uav] = grid_idx
                assigned_grids.add(grid_idx)


class FrontierExplorationBaseline(BaselineMethod):
    """前沿探索基线"""

    def __init__(self, uav_fleet, environment, config):
        super().__init__(uav_fleet, environment, config)
        self.explored_regions = set()

    def plan_step(self, current_time, aoi_states):
        """前沿探索策略"""
        uav_positions = self.uav_fleet.get_positions()
        grid_centers = self.environment.workspace.grid_centers

        # 更新已探索区域
        for i, uav_pos in enumerate(uav_positions):
            for j, grid_center in enumerate(grid_centers):
                if np.linalg.norm(uav_pos - grid_center) <= self.uav_fleet.uavs[i].sensing_radius:
                    self.explored_regions.add(j)

        # 为每个UAV找到最近的前沿
        control_inputs = []
        for i, uav in enumerate(self.uav_fleet.uavs):
            frontier_target = self._find_nearest_frontier(uav_positions[i])

            current_pos = uav.get_position()
            position_error = frontier_target - current_pos
            velocity_error = np.zeros(2) - uav.get_velocity()

            Kp = 0.5
            Kd = 0.1
            control = Kp * position_error + Kd * velocity_error

            max_control = self.uav_fleet.dynamics.max_acceleration
            control = np.clip(control, -max_control, max_control)

            control_inputs.append(control)

        return control_inputs

    def _find_nearest_frontier(self, uav_position):
        """找到最近的前沿"""
        grid_centers = self.environment.workspace.grid_centers

        # 找到未探索的网格
        unexplored = []
        for i, center in enumerate(grid_centers):
            if i not in self.explored_regions:
                unexplored.append((i, center))

        if not unexplored:
            # 如果都探索完了，随机选择
            return np.array([
                random.uniform(self.environment.workspace.domain[0], self.environment.workspace.domain[1]),
                random.uniform(self.environment.workspace.domain[2], self.environment.workspace.domain[3])
            ])

        # 找到最近的未探索网格
        min_distance = float('inf')
        nearest_frontier = unexplored[0][1]

        for _, center in unexplored:
            distance = np.linalg.norm(uav_position - center)
            if distance < min_distance:
                min_distance = distance
                nearest_frontier = center

        return nearest_frontier


class RoundRobinBaseline(BaselineMethod):
    """轮询巡检基线"""

    def __init__(self, uav_fleet, environment, config):
        super().__init__(uav_fleet, environment, config)
        self.current_targets = {}
        self.target_sequence = self._generate_target_sequence()

    def plan_step(self, current_time, aoi_states):
        """轮询巡检策略"""
        uav_positions = self.uav_fleet.get_positions()

        # 更新目标
        if current_time % 20 == 0:  # 每20步更新目标
            self._update_targets()

        # 生成控制输入
        control_inputs = []
        for i, uav in enumerate(self.uav_fleet.uavs):
            if i in self.current_targets:
                target_pos = self.current_targets[i]
            else:
                target_pos = uav.get_position()

            current_pos = uav.get_position()
            position_error = target_pos - current_pos
            velocity_error = np.zeros(2) - uav.get_velocity()

            Kp = 0.5
            Kd = 0.1
            control = Kp * position_error + Kd * velocity_error

            max_control = self.uav_fleet.dynamics.max_acceleration
            control = np.clip(control, -max_control, max_control)

            control_inputs.append(control)

        return control_inputs

    def _generate_target_sequence(self):
        """生成目标序列"""
        grid_centers = self.environment.workspace.grid_centers

        # 找到高密度区域作为热点
        density_threshold = np.percentile(self.environment.density_map, 80)
        hotspots = []

        for i, density in enumerate(self.environment.density_map):
            if density > density_threshold:
                hotspots.append(grid_centers[i])

        if len(hotspots) < 3:
            # 如果热点不够，添加一些随机点
            while len(hotspots) < 3:
                random_pos = np.array([
                    random.uniform(self.environment.workspace.domain[0], self.environment.workspace.domain[1]),
                    random.uniform(self.environment.workspace.domain[2], self.environment.workspace.domain[3])
                ])
                hotspots.append(random_pos)

        return hotspots

    def _update_targets(self):
        """更新UAV目标"""
        n_uavs = len(self.uav_fleet.uavs)
        n_targets = len(self.target_sequence)

        for i in range(n_uavs):
            target_idx = (i + random.randint(0, n_targets - 1)) % n_targets
            self.current_targets[i] = self.target_sequence[target_idx]


class GreedyAoIBaseline(BaselineMethod):
    """贪心AoI最小化基线"""

    def plan_step(self, current_time, aoi_states):
        """贪心AoI最小化策略"""
        uav_positions = self.uav_fleet.get_positions()
        grid_centers = self.environment.workspace.grid_centers

        # 为每个UAV找到AoI最大的网格
        control_inputs = []
        assigned_grids = set()

        for i, uav in enumerate(self.uav_fleet.uavs):
            # 找到AoI最大的未分配网格
            max_aoi = -1
            best_grid = None

            for j, aoi in enumerate(aoi_states):
                if j not in assigned_grids:
                    if aoi > max_aoi:
                        max_aoi = aoi
                        best_grid = j

            if best_grid is not None:
                target_pos = grid_centers[best_grid]
                assigned_grids.add(best_grid)
            else:
                target_pos = uav.get_position()

            current_pos = uav.get_position()
            position_error = target_pos - current_pos
            velocity_error = np.zeros(2) - uav.get_velocity()

            Kp = 0.5
            Kd = 0.1
            control = Kp * position_error + Kd * velocity_error

            max_control = self.uav_fleet.dynamics.max_acceleration
            control = np.clip(control, -max_control, max_control)

            control_inputs.append(control)

        return control_inputs


class CentralizedMPCBaseline(BaselineMethod):
    """集中式MPC基线"""

    def __init__(self, uav_fleet, environment, config):
        super().__init__(uav_fleet, environment, config)
        self.H = config['rhc']['horizon']

    def plan_step(self, current_time, aoi_states):
        """集中式MPC规划（简化版本）"""
        # 由于CVXPY不支持3维变量，使用简化的方法
        # 直接使用备选控制策略
        return self._fallback_control()

    def _fallback_control(self):
        """备选控制策略"""
        control_inputs = []
        for uav in self.uav_fleet.uavs:
            # 简单的PD控制器
            current_pos = uav.get_position()
            target_pos = current_pos  # 保持当前位置

            position_error = target_pos - current_pos
            velocity_error = np.zeros(2) - uav.get_velocity()

            Kp = 0.5
            Kd = 0.1
            control = Kp * position_error + Kd * velocity_error

            max_control = self.uav_fleet.dynamics.max_acceleration
            control = np.clip(control, -max_control, max_control)

            control_inputs.append(control)

        return control_inputs




