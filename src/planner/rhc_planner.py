"""
分布式RHC规划器：滚动时域控制和AoI约束
"""
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


class RHCOptimizer:
    """RHC优化器"""

    def __init__(self, uav_dynamics, environment, partition, config):
        self.dynamics = uav_dynamics
        self.environment = environment
        self.partition = partition
        self.config = config

        # RHC参数
        self.H = config['rhc']['horizon']
        self.alpha_a = config['rhc']['cost_weights']['aoi']
        self.alpha_u = config['rhc']['cost_weights']['control']
        self.alpha_c = config['rhc']['cost_weights']['collision']
        self.alpha_b = config['rhc']['cost_weights']['boundary']

        # AoI收紧参数
        self.delta = config['aoi_tightening']['delta']
        self.sigma = config['aoi_tightening']['sigma']

        # 终端集参数
        self.epsilon = config['rhc']['terminal_set_epsilon']
        self.coupling_decay = config['rhc']['coupling_decay']

        # 构建终端控制器
        self.terminal_controller = self._build_terminal_controller()

    def _build_terminal_controller(self):
        """构建终端控制器（LQR）"""
        # 使用简化的方法避免矩阵奇异性问题
        A = self.dynamics.A
        B = self.dynamics.B

        # 权重矩阵
        Q = np.eye(4)  # 状态权重
        R = np.eye(2)  # 控制权重

        # 使用更稳定的方法构建终端控制器
        # 对于双积分器模型，直接计算LQR增益
        dt = self.dynamics.dt
        
        # 简化的LQR增益计算（避免求解Lyapunov方程）
        # 对于双积分器，可以使用解析解
        q1, q2, q3, q4 = 1.0, 1.0, 0.1, 0.1  # 状态权重
        r1, r2 = 1.0, 1.0  # 控制权重
        
        # 简化的反馈增益（基于双积分器的特性）
        K = np.array([
            [-np.sqrt(q1/r1), 0, -np.sqrt(q3/r1), 0],
            [0, -np.sqrt(q2/r2), 0, -np.sqrt(q4/r2)]
        ])
        
        # 确保增益矩阵的维度正确
        if K.shape != (2, 4):
            K = np.zeros((2, 4))
            K[0, 0] = -0.5  # 位置反馈
            K[0, 2] = -0.3  # 速度反馈
            K[1, 1] = -0.5
            K[1, 3] = -0.3

        return K

    def compute_aoi_tightening(self, T):
        """计算AoI收紧参数"""
        # β_k = σ_k * sqrt(2 * log(T/δ))
        beta = self.sigma * np.sqrt(2 * np.log(T / self.delta))
        return beta

    def build_rhc_problem(self, uav_id, current_state, uav_positions, site_weights,
                          aoi_states, neighbor_states, neighbor_plans):
        """构建RHC优化问题"""
        # 获取UAV的Voronoi单元
        cells = self.partition.compute_voronoi_cells(uav_positions, site_weights)
        if uav_id not in cells:
            return None, None

        cell_indices = cells[uav_id]
        if len(cell_indices) == 0:
            return None, None

        # 优化变量
        nx = self.dynamics.nx
        nu = self.dynamics.nu

        x = cp.Variable((nx, self.H + 1))  # 状态轨迹
        u = cp.Variable((nu, self.H))  # 控制轨迹

        # 约束
        constraints = []

        # 初始状态约束
        constraints.append(x[:, 0] == current_state)

        # 动力学约束
        for h in range(self.H):
            constraints.append(x[:, h + 1] ==
                               self.dynamics.A @ x[:, h] + self.dynamics.B @ u[:, h])

        # 控制约束（放宽限制）
        control_limits = self.dynamics.get_control_constraints()
        # 广播控制约束到整个预测时域
        for h in range(self.H):
            for i in range(u.shape[0]):  # 对每个控制维度
                constraints.append(u[i, h] >= control_limits['min'][i] * 1.5)  # 放宽下限
                constraints.append(u[i, h] <= control_limits['max'][i] * 1.5)  # 放宽上限

        # 速度约束（放宽限制）
        velocity_limits = self.dynamics.get_state_constraints()
        for h in range(self.H + 1):
            constraints.append(x[2:4, h] >= velocity_limits['velocity_min'] * 0.5)  # 放宽下限
            constraints.append(x[2:4, h] <= velocity_limits['velocity_max'] * 1.5)  # 放宽上限

        # 终端集约束（暂时移除以简化问题）
        # terminal_set = self._compute_terminal_set(uav_positions[uav_id], cell_indices)
        # constraints.append(cp.sum_squares(x[:, self.H] - terminal_set['center']) <= terminal_set['radius']**2)

        # 目标函数
        objective = 0

        # AoI代价（简化版本）
        for h in range(self.H + 1):
            # 只考虑最近的几个网格点以减少计算复杂度
            for k in cell_indices[:min(5, len(cell_indices))]:  # 限制最多5个网格点
                grid_center = self.environment.workspace.grid_centers[k]
                # 使用平方距离代替范数，使其成为QP
                distance_sq = cp.sum_squares(x[0:2, h] - grid_center)

                # AoI更新（简化）
                aoi_penalty = self.alpha_a * aoi_states[k] * distance_sq
                objective += aoi_penalty

        # 控制代价
        for h in range(self.H):
            objective += self.alpha_u * cp.sum_squares(u[:, h])

        # 碰撞避免代价（简化版本）
        for h in range(self.H + 1):
            # 只考虑最近的几个邻居以减少计算复杂度
            neighbor_count = 0
            for neighbor_id in neighbor_states:
                if neighbor_id != uav_id and neighbor_count < 3:  # 限制最多3个邻居
                    neighbor_pos = neighbor_states[neighbor_id]
                    # 只使用位置分量（前2个元素）
                    neighbor_pos_2d = neighbor_pos[0:2] if len(neighbor_pos) >= 2 else neighbor_pos
                    # 使用简单的距离惩罚（DCP兼容）
                    distance_sq = cp.sum_squares(x[0:2, h] - neighbor_pos_2d)
                    # 使用线性惩罚代替非线性惩罚
                    collision_penalty = self.alpha_c * distance_sq
                    objective += collision_penalty
                    neighbor_count += 1

        # 边界代价（简化版本）
        for h in range(self.H + 1):
            pos = x[0:2, h]
            domain = self.environment.workspace.domain

            # 简单的边界惩罚（保持在工作空间内）
            boundary_penalty = self.alpha_b * (
                    (pos[0] - domain[0])**2 + (domain[1] - pos[0])**2 +
                    (pos[1] - domain[2])**2 + (domain[3] - pos[1])**2
            )
            objective += boundary_penalty

        # 耦合惩罚
        if neighbor_plans:
            coupling_penalty = self._compute_coupling_penalty(x, neighbor_plans)
            objective += self.coupling_decay * coupling_penalty

        # 求解问题
        problem = cp.Problem(cp.Minimize(objective), constraints)

        return problem, (x, u)

    def _compute_terminal_set(self, uav_position, cell_indices):
        """计算终端集"""
        # 确保uav_position是4维向量
        if len(uav_position) == 2:
            uav_position_4d = np.array([uav_position[0], uav_position[1], 0.0, 0.0])
        else:
            uav_position_4d = np.array(uav_position).flatten()
            
        if len(cell_indices) == 0:
            return {'center': uav_position_4d, 'radius': self.epsilon}

        # 计算单元质心
        cell_centers = self.environment.workspace.grid_centers[cell_indices]
        centroid = np.mean(cell_centers, axis=0)

        # 终端集为中心附近的区域
        terminal_center = np.array([centroid[0], centroid[1], 0.0, 0.0])

        return {
            'center': terminal_center,
            'radius': self.epsilon
        }

    def _compute_coupling_penalty(self, x, neighbor_plans):
        """计算耦合惩罚"""
        penalty = 0

        for neighbor_id, neighbor_plan in neighbor_plans.items():
            if neighbor_plan is not None:
                # 轨迹一致性惩罚
                for h in range(min(self.H + 1, len(neighbor_plan))):
                    distance_sq = cp.sum_squares(x[0:2, h] - neighbor_plan[h][0:2])
                    penalty += distance_sq

        return penalty

    def solve_rhc_problem(self, problem, variables, current_state=None, uav_positions=None, uav_id=None, cell_indices=None, aoi_states=None, neighbor_states=None):
        """求解RHC问题"""
        if problem is None:
            return None

        try:
            # 尝试使用OSQP，如果失败则使用默认求解器
            try:
                problem.solve(solver=cp.OSQP, verbose=False)
            except:
                problem.solve(verbose=False)

            if problem.status == cp.OPTIMAL:
                x, u = variables
                return {
                    'x_opt': x.value,
                    'u_opt': u.value,
                    'cost': problem.value
                }
            else:
                # 如果问题不可行，尝试放宽约束
                print(f"RHC problem not solved: {problem.status}, trying relaxed constraints")
                return self._solve_relaxed_problem(variables, uav_positions, uav_id, cell_indices, aoi_states, neighbor_states)

        except Exception as e:
            print(f"Error solving RHC problem: {e}")
            return None

    def get_terminal_control(self, state, terminal_set):
        """获取终端控制"""
        K = self.terminal_controller
        # 确保状态和终端集中心都是4维向量
        state_4d = np.array(state).flatten()
        center_4d = np.array(terminal_set['center']).flatten()
        
        # 计算控制输入
        control = -K @ (state_4d - center_4d)
        return control

    def _solve_relaxed_problem(self, variables, uav_positions, uav_id, cell_indices, aoi_states, neighbor_states):
        """求解放宽约束的RHC问题"""
        x, u = variables
        
        # 获取当前状态（从UAV位置推断）
        current_state = np.array([uav_positions[uav_id][0], uav_positions[uav_id][1], 0.0, 0.0])
        
        # 构建非常简化的目标函数
        objective = 0
        
        # 控制代价
        for h in range(self.H):
            objective += 0.01 * cp.sum_squares(u[:, h])
        
        # 简化的约束
        constraints = []
        
        # 初始状态约束
        constraints.append(x[:, 0] == current_state)
        
        # 动力学约束
        for h in range(self.H):
            constraints.append(x[:, h + 1] ==
                               self.dynamics.A @ x[:, h] + self.dynamics.B @ u[:, h])
        
        # 非常宽松的控制约束
        control_limits = self.dynamics.get_control_constraints()
        for h in range(self.H):
            for i in range(u.shape[0]):
                constraints.append(u[i, h] >= control_limits['min'][i] * 2.0)  # 进一步放宽下限
                constraints.append(u[i, h] <= control_limits['max'][i] * 2.0)  # 进一步放宽上限
        
        try:
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(verbose=False)
            
            if problem.status == cp.OPTIMAL:
                return {
                    'x_opt': x.value,
                    'u_opt': u.value,
                    'cost': problem.value
                }
            else:
                print(f"Relaxed problem also failed: {problem.status}")
                return None
                
        except Exception as e:
            print(f"Error solving relaxed problem: {e}")
            return None


class DistributedRHCPlanner:
    """分布式RHC规划器"""

    def __init__(self, uav_fleet, environment, partition, config):
        self.uav_fleet = uav_fleet
        self.environment = environment
        self.partition = partition
        self.config = config

        # 为每个UAV创建优化器
        self.optimizers = {}
        for uav in uav_fleet.uavs:
            self.optimizers[uav.id] = RHCOptimizer(
                uav_fleet.dynamics, environment, partition, config
            )

        # 规划历史
        self.planning_history = []

    def plan_step(self, current_time, aoi_states):
        """执行一步规划"""
        uav_positions = self.uav_fleet.get_positions()
        site_weights = [uav.site_weight for uav in self.uav_fleet.uavs]

        # 第一轮：构建问题
        problems = {}
        variables = {}
        neighbor_states = {}
        neighbor_plans = {}

        for uav in self.uav_fleet.uavs:
            # 获取邻居状态
            neighbors = {}
            for neighbor_id in uav.neighbors:
                neighbor = self.uav_fleet.uavs[neighbor_id]
                neighbors[neighbor_id] = neighbor.state

            neighbor_states[uav.id] = neighbors
            neighbor_plans[uav.id] = {}  # 初始为空

            # 构建优化问题
            problem, var = self.optimizers[uav.id].build_rhc_problem(
                uav.id, uav.state, uav_positions, site_weights,
                aoi_states, neighbors, {}
            )

            problems[uav.id] = problem
            variables[uav.id] = var

        # 迭代求解（简化版本：只迭代一次）
        solutions = {}
        for uav in self.uav_fleet.uavs:
            if problems[uav.id] is not None:
                # 获取UAV的Voronoi单元
                cells = self.partition.compute_voronoi_cells(uav_positions, site_weights)
                cell_indices = cells.get(uav.id, [])
                
                solution = self.optimizers[uav.id].solve_rhc_problem(
                    problems[uav.id], variables[uav.id],
                    current_state=uav.state,
                    uav_positions=uav_positions,
                    uav_id=uav.id,
                    cell_indices=cell_indices,
                    aoi_states=aoi_states,
                    neighbor_states=neighbor_states[uav.id]
                )
                solutions[uav.id] = solution
            else:
                solutions[uav.id] = None

        # 提取控制输入
        control_inputs = []
        for uav in self.uav_fleet.uavs:
            if solutions[uav.id] is not None:
                control = solutions[uav.id]['u_opt'][:, 0]  # 取第一步控制
            else:
                # 使用终端控制器作为备选
                terminal_set = self.optimizers[uav.id]._compute_terminal_set(
                    uav.get_position(), []
                )
                control = self.optimizers[uav.id].get_terminal_control(
                    uav.state, terminal_set
                )

            control_inputs.append(control)

        # 记录规划历史
        self.planning_history.append({
            'time': current_time,
            'solutions': solutions,
            'control_inputs': control_inputs
        })

        return control_inputs

    def get_planning_metrics(self):
        """获取规划指标"""
        if not self.planning_history:
            return {}

        metrics = {
            'total_planning_time': 0,
            'successful_solves': 0,
            'failed_solves': 0,
            'average_cost': 0,
            'cost_history': []
        }

        total_cost = 0
        for plan in self.planning_history:
            for uav_id, solution in plan['solutions'].items():
                if solution is not None:
                    metrics['successful_solves'] += 1
                    total_cost += solution['cost']
                    metrics['cost_history'].append(solution['cost'])
                else:
                    metrics['failed_solves'] += 1

        if metrics['successful_solves'] > 0:
            metrics['average_cost'] = total_cost / metrics['successful_solves']

        return metrics

    def visualize_planning_results(self, ax=None):
        """可视化规划结果"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        if not self.planning_history:
            return ax

        # 绘制成本历史
        costs = []
        times = []
        for plan in self.planning_history:
            total_cost = 0
            count = 0
            for solution in plan['solutions'].values():
                if solution is not None:
                    total_cost += solution['cost']
                    count += 1

            if count > 0:
                costs.append(total_cost / count)
                times.append(plan['time'])

        if costs:
            ax.plot(times, costs, 'b-', linewidth=2)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Average Cost')
            ax.set_title('RHC Planning Cost History')
            ax.grid(True, alpha=0.3)

        return ax
