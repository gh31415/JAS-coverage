"""
Power-Voronoi分区模块：加权Voronoi分区和Lloyd更新
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import cvxpy as cp


class PowerVoronoiPartition:
    """Power-Voronoi分区类"""

    def __init__(self, environment, uav_fleet):
        self.environment = environment
        self.uav_fleet = uav_fleet
        self.workspace = environment.workspace
        self.grid_centers = environment.workspace.grid_centers

        # 分区历史
        self.partition_history = []
        self.quantization_history = []

    def compute_weighted_distance(self, point, uav_position, site_weight):
        """计算加权距离"""
        # 加权距离：d_w(q, p_i) = ||q - p_i||^2 - w_i
        euclidean_distance = np.linalg.norm(point - uav_position) ** 2
        return euclidean_distance - site_weight

    def assign_cells_to_uavs(self, uav_positions, site_weights):
        """将网格分配给UAV"""
        assignments = []

        for grid_center in self.grid_centers:
            min_distance = float('inf')
            assigned_uav = 0

            for i, (uav_pos, site_weight) in enumerate(zip(uav_positions, site_weights)):
                weighted_dist = self.compute_weighted_distance(grid_center, uav_pos, site_weight)

                if weighted_dist < min_distance:
                    min_distance = weighted_dist
                    assigned_uav = i

            assignments.append(assigned_uav)

        return np.array(assignments)

    def compute_voronoi_cells(self, uav_positions, site_weights):
        """计算Power-Voronoi单元"""
        assignments = self.assign_cells_to_uavs(uav_positions, site_weights)

        cells = {}
        for i in range(len(uav_positions)):
            cell_indices = np.where(assignments == i)[0]
            cells[i] = cell_indices

        return cells

    def compute_weighted_centroid(self, cell_indices, uav_position):
        """计算加权质心"""
        if len(cell_indices) == 0:
            return uav_position

        # 获取单元内的网格中心
        cell_centers = self.grid_centers[cell_indices]

        # 获取环境密度
        cell_densities = self.environment.density_map[cell_indices]

        # 计算加权质心
        weighted_sum = np.sum(cell_centers * cell_densities[:, np.newaxis], axis=0)
        total_weight = np.sum(cell_densities)

        if total_weight > 0:
            centroid = weighted_sum / total_weight
        else:
            centroid = uav_position

        return centroid

    def lloyd_update(self, uav_positions, site_weights):
        """Lloyd更新"""
        cells = self.compute_voronoi_cells(uav_positions, site_weights)

        new_positions = []
        for i in range(len(uav_positions)):
            if i in cells:
                centroid = self.compute_weighted_centroid(cells[i], uav_positions[i])
                new_positions.append(centroid)
            else:
                new_positions.append(uav_positions[i])

        return np.array(new_positions)

    def compute_quantization_functional(self, uav_positions, site_weights):
        """计算量化泛函 H(P)"""
        cells = self.compute_voronoi_cells(uav_positions, site_weights)

        H = 0.0
        for i, uav_pos in enumerate(uav_positions):
            if i in cells:
                cell_indices = cells[i]
                cell_centers = self.grid_centers[cell_indices]
                cell_densities = self.environment.density_map[cell_indices]

                # 计算到UAV位置的距离
                distances = np.linalg.norm(cell_centers - uav_pos, axis=1)

                # 加权求和
                H += np.sum(cell_densities * distances ** 2)

        return H

    def optimize_site_weights(self, uav_positions, target_workloads):
        """优化站点权重以平衡工作负载"""
        n_uavs = len(uav_positions)

        # 创建优化问题
        site_weights = cp.Variable(n_uavs)

        # 目标：最小化工作负载差异
        cells = self.compute_voronoi_cells(uav_positions, [0] * n_uavs)  # 初始权重为0
        current_workloads = []

        for i in range(n_uavs):
            if i in cells:
                cell_indices = cells[i]
                workload = np.sum(self.environment.density_map[cell_indices])
            else:
                workload = 0
            current_workloads.append(workload)

        # 目标函数：最小化工作负载差异
        objective = cp.sum_squares(cp.multiply(current_workloads, site_weights) - target_workloads)

        # 约束：权重非负
        constraints = [site_weights >= 0]

        # 求解
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()

        return site_weights.value

    def update_partition(self, uav_positions, site_weights, max_iterations=50, tolerance=1e-6):
        """更新分区（Lloyd迭代）"""
        positions = uav_positions.copy()
        H_history = []

        for iteration in range(max_iterations):
            # 计算当前量化泛函
            H_current = self.compute_quantization_functional(positions, site_weights)
            H_history.append(H_current)

            # Lloyd更新
            new_positions = self.lloyd_update(positions, site_weights)

            # 检查收敛
            position_change = np.linalg.norm(new_positions - positions)
            if position_change < tolerance:
                break

            positions = new_positions

        # 记录历史
        self.partition_history.append(positions.copy())
        self.quantization_history.extend(H_history)

        return positions, H_history

    def get_cell_assignment(self, uav_positions, site_weights):
        """获取当前的分区分配"""
        return self.assign_cells_to_uavs(uav_positions, site_weights)

    def visualize_partition(self, uav_positions, site_weights, ax=None):
        """可视化分区"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))

        # 获取分区分配
        assignments = self.assign_cells_to_uavs(uav_positions, site_weights)

        # 绘制网格，按分配着色
        colors = plt.cm.Set3(np.linspace(0, 1, len(uav_positions)))
        for i, grid_center in enumerate(self.grid_centers):
            uav_id = assignments[i]
            ax.scatter(grid_center[0], grid_center[1],
                       c=[colors[uav_id]], alpha=0.6, s=20)

        # 绘制UAV位置
        ax.scatter(uav_positions[:, 0], uav_positions[:, 1],
                   c='red', s=200, marker='*', edgecolors='black', linewidth=2)

        # 绘制Voronoi边界
        cells = self.compute_voronoi_cells(uav_positions, site_weights)
        for uav_id, cell_indices in cells.items():
            if len(cell_indices) > 0:
                cell_centers = self.grid_centers[cell_indices]
                # 计算凸包作为边界
                if len(cell_centers) >= 3:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(cell_centers)
                    boundary_points = cell_centers[hull.vertices]
                    ax.plot(boundary_points[:, 0], boundary_points[:, 1],
                            'k-', linewidth=2, alpha=0.8)

        ax.set_xlim(self.workspace.domain[0], self.workspace.domain[1])
        ax.set_ylim(self.workspace.domain[2], self.workspace.domain[3])
        ax.set_aspect('equal')
        ax.set_title('Power-Voronoi Partition')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        return ax

    def plot_quantization_convergence(self, ax=None):
        """绘制量化泛函收敛曲线"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.quantization_history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Quantization Functional H(P)')
        ax.set_title('Lloyd Algorithm Convergence')
        ax.grid(True, alpha=0.3)

        return ax

    def get_workload_distribution(self, uav_positions, site_weights):
        """获取工作负载分布"""
        cells = self.compute_voronoi_cells(uav_positions, site_weights)

        workloads = []
        for i in range(len(uav_positions)):
            if i in cells:
                cell_indices = cells[i]
                workload = np.sum(self.environment.density_map[cell_indices])
            else:
                workload = 0
            workloads.append(workload)

        return np.array(workloads)

    def compute_partition_metrics(self, uav_positions, site_weights):
        """计算分区指标"""
        cells = self.compute_voronoi_cells(uav_positions, site_weights)

        metrics = {
            'num_cells': len(cells),
            'cell_sizes': [],
            'workloads': [],
            'coverage_ratio': 0.0
        }

        total_covered = 0
        for uav_id, cell_indices in cells.items():
            metrics['cell_sizes'].append(len(cell_indices))
            workload = np.sum(self.environment.density_map[cell_indices])
            metrics['workloads'].append(workload)
            total_covered += len(cell_indices)

        metrics['coverage_ratio'] = total_covered / len(self.grid_centers)
        metrics['workload_std'] = np.std(metrics['workloads'])
        metrics['workload_mean'] = np.mean(metrics['workloads'])

        return metrics




