"""
UAV动力学模型：离散双积分模型和干扰生成
"""
import numpy as np
import random
import matplotlib.pyplot as plt


class UAVDynamics:
    """UAV动力学模型"""

    def __init__(self, dt, max_velocity, max_acceleration):
        self.dt = dt
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

        # 状态维度：x = [px, py, vx, vy]^T
        self.nx = 4
        self.nu = 2  # 控制输入：ax, ay

        # 构建离散化矩阵
        self.A, self.B = self._build_discrete_matrices()

    def _build_discrete_matrices(self):
        """构建离散化状态空间矩阵"""
        # 对于双积分器模型，直接构建离散时间矩阵
        dt = self.dt
        
        # 离散时间状态转移矩阵 A
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 离散时间控制矩阵 B
        B = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])
        
        return A, B

    def step(self, x, u, d=None):
        """UAV动力学步进"""
        if d is None:
            d = np.zeros(self.nx)

        # 状态更新
        x_next = self.A @ x + self.B @ u + d

        # 速度约束
        velocity = x_next[2:4]
        if np.linalg.norm(velocity) > self.max_velocity:
            velocity = velocity / np.linalg.norm(velocity) * self.max_velocity
            x_next[2:4] = velocity

        return x_next

    def get_control_constraints(self):
        """获取控制约束"""
        return {
            'min': -self.max_acceleration * np.ones(self.nu),
            'max': self.max_acceleration * np.ones(self.nu)
        }

    def get_state_constraints(self):
        """获取状态约束"""
        return {
            'velocity_min': -self.max_velocity * np.ones(2),
            'velocity_max': self.max_velocity * np.ones(2)
        }


class DisturbanceGenerator:
    """干扰生成器"""

    def __init__(self, disturbance_type, params):
        self.type = disturbance_type
        self.params = params
        self.direction = np.array([1.0, 0.0])  # 初始方向

    def generate(self, t):
        """生成干扰"""
        if self.type == "none":
            return np.zeros(4)

        elif self.type == "bounded":
            max_magnitude = self.params['max_magnitude']
            direction_drift = self.params['direction_drift']

            # 方向缓慢漂移
            self.direction += np.random.normal(0, direction_drift, 2)
            self.direction = self.direction / np.linalg.norm(self.direction)

            # 有界干扰
            magnitude = np.random.uniform(0, max_magnitude)
            disturbance_2d = magnitude * self.direction

            # 扩展到4D状态空间（只影响位置）
            d = np.zeros(4)
            d[0:2] = disturbance_2d

            return d

        elif self.type == "sub_gaussian":
            sigma = self.params['sigma']
            mean = self.params['mean']

            # 亚高斯干扰
            disturbance_2d = np.random.normal(mean, sigma, 2)

            # 扩展到4D状态空间
            d = np.zeros(4)
            d[0:2] = disturbance_2d

            return d

        else:
            raise ValueError(f"Unknown disturbance type: {self.type}")


class UAV:
    """UAV类"""

    def __init__(self, uav_id, initial_state, dynamics, sensing_radius, site_weight,
                 communication_radius, packet_loss_rate, communication_delay):
        self.id = uav_id
        self.state = initial_state
        self.dynamics = dynamics
        self.sensing_radius = sensing_radius
        self.site_weight = site_weight
        self.communication_radius = communication_radius
        self.packet_loss_rate = packet_loss_rate
        self.communication_delay = communication_delay

        # 通信相关
        self.neighbors = []
        self.received_messages = []
        self.message_buffer = []

        # 历史轨迹
        self.trajectory = [initial_state.copy()]
        self.control_history = []

    def update_state(self, control_input, disturbance):
        """更新UAV状态"""
        self.state = self.dynamics.step(self.state, control_input, disturbance)
        self.trajectory.append(self.state.copy())
        self.control_history.append(control_input.copy())

        return self.state

    def get_position(self):
        """获取当前位置"""
        return self.state[0:2]

    def get_velocity(self):
        """获取当前速度"""
        return self.state[2:4]

    def can_communicate_with(self, other_uav):
        """检查是否可以与其他UAV通信"""
        distance = np.linalg.norm(self.get_position() - other_uav.get_position())
        return distance <= self.communication_radius

    def send_message(self, message, target_uav):
        """发送消息"""
        if self.can_communicate_with(target_uav):
            # 模拟丢包
            if random.random() > self.packet_loss_rate:
                # 模拟通信延迟
                delay = random.randint(self.communication_delay[0], self.communication_delay[1])
                message_with_delay = {
                    'message': message,
                    'delay': delay,
                    'sender_id': self.id
                }
                target_uav.received_messages.append(message_with_delay)

    def update_neighbors(self, all_uavs):
        """更新邻居列表"""
        self.neighbors = []
        for uav in all_uavs:
            if uav.id != self.id and self.can_communicate_with(uav):
                self.neighbors.append(uav.id)

    def process_messages(self):
        """处理接收到的消息"""
        # 处理延迟消息
        processed_messages = []
        for msg in self.received_messages:
            msg['delay'] -= 1
            if msg['delay'] <= 0:
                processed_messages.append(msg['message'])
            else:
                self.message_buffer.append(msg)

        self.received_messages = self.message_buffer
        self.message_buffer = []

        return processed_messages


class UAVFleet:
    """UAV机队类"""

    def __init__(self, uav_config, environment):
        self.uavs = []
        self.environment = environment
        self.uav_config = uav_config

        # 创建UAV动力学模型
        self.dynamics = UAVDynamics(
            dt=uav_config['sampling_period'],
            max_velocity=uav_config['max_velocity'],
            max_acceleration=uav_config['max_acceleration']
        )

    def create_uavs(self, count, initial_positions=None):
        """创建UAV机队"""
        if initial_positions is None:
            # 随机初始化位置
            initial_positions = []
            for i in range(count):
                while True:
                    pos = np.array([
                        random.uniform(self.environment.workspace.domain[0] + 50,
                                       self.environment.workspace.domain[1] - 50),
                        random.uniform(self.environment.workspace.domain[2] + 50,
                                       self.environment.workspace.domain[3] - 50)
                    ])
                    if self.environment.is_free_space(pos):
                        initial_positions.append(pos)
                        break

        for i in range(count):
            # 随机选择传感半径和站点权重
            sensing_radius = random.choice(self.uav_config['sensing_radii'])
            site_weight = random.choice(self.uav_config['site_weights'])

            # 初始状态：位置 + 零速度
            initial_state = np.array([
                initial_positions[i][0], initial_positions[i][1], 0.0, 0.0
            ])

            uav = UAV(
                uav_id=i,
                initial_state=initial_state,
                dynamics=self.dynamics,
                sensing_radius=sensing_radius,
                site_weight=site_weight,
                communication_radius=self.uav_config['communication_radius'],
                packet_loss_rate=self.uav_config['packet_loss_rate'],
                communication_delay=self.uav_config['communication_delay']
            )

            self.uavs.append(uav)

    def update_communication_network(self):
        """更新通信网络"""
        for uav in self.uavs:
            uav.update_neighbors(self.uavs)

    def step(self, control_inputs, disturbances):
        """机队步进"""
        for i, uav in enumerate(self.uavs):
            if i < len(control_inputs):
                uav.update_state(control_inputs[i], disturbances[i])

        # 更新通信网络
        self.update_communication_network()

    def get_positions(self):
        """获取所有UAV位置"""
        return [uav.get_position() for uav in self.uavs]

    def get_states(self):
        """获取所有UAV状态"""
        return [uav.state for uav in self.uavs]

    def visualize(self, ax):
        """可视化UAV机队"""
        positions = self.get_positions()
        positions = np.array(positions)

        # 绘制UAV位置
        ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=100, alpha=0.8)

        # 绘制传感范围
        for uav in self.uavs:
            pos = uav.get_position()
            circle = plt.Circle(pos, uav.sensing_radius, color='blue',
                                alpha=0.2, fill=False)
            ax.add_patch(circle)

        # 绘制通信连接
        for uav in self.uavs:
            for neighbor_id in uav.neighbors:
                neighbor = self.uavs[neighbor_id]
                ax.plot([uav.get_position()[0], neighbor.get_position()[0]],
                        [uav.get_position()[1], neighbor.get_position()[1]],
                        'g-', alpha=0.5, linewidth=1)

        return ax
