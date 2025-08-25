"""
评估指标模块：实现各种性能指标计算
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
import pandas as pd
import os  # Added for visualization methods


class CoverageMetrics:
    """覆盖性能指标"""

    def __init__(self, environment, aoi_thresholds):
        self.environment = environment
        self.aoi_thresholds = aoi_thresholds
        self.workspace = environment.workspace

    def compute_aoi_metrics(self, aoi_history, visit_history):
        """计算AoI相关指标"""
        aoi_history = np.array(aoi_history)
        visit_history = np.array(visit_history)

        metrics = {}

        # 平均AoI
        metrics['average_aoi'] = np.mean(aoi_history)

        # 最大AoI
        metrics['max_aoi'] = np.max(aoi_history)

        # Deadline满足率
        deadline_violations = 0
        total_checks = 0

        for t in range(aoi_history.shape[0]):
            for k in range(aoi_history.shape[1]):
                # 根据网格类型确定阈值
                grid_type = self._get_grid_type(k)
                threshold = self.aoi_thresholds[grid_type]

                if aoi_history[t, k] > threshold:
                    deadline_violations += 1
                total_checks += 1

        metrics['deadline_satisfaction_rate'] = 1 - (deadline_violations / total_checks)

        # 重访间隔统计
        revisit_intervals = self._compute_revisit_intervals(visit_history)
        metrics['max_revisit_interval'] = np.max(revisit_intervals) if revisit_intervals else 0
        metrics['mean_revisit_interval'] = np.mean(revisit_intervals) if revisit_intervals else 0

        return metrics

    def _get_grid_type(self, grid_index):
        """获取网格类型"""
        grid_center = self.workspace.grid_centers[grid_index]
        density = self.environment.density_map[grid_index]

        # 根据密度判断网格类型
        density_threshold = np.percentile(self.environment.density_map, 80)

        if density > density_threshold:
            return 'hotspot'
        elif density > np.median(self.environment.density_map):
            return 'corridor'
        else:
            return 'base'

    def _compute_revisit_intervals(self, visit_history):
        """计算重访间隔"""
        intervals = []

        for k in range(visit_history.shape[1]):
            visit_times = np.where(visit_history[:, k] == 1)[0]
            if len(visit_times) > 1:
                for i in range(1, len(visit_times)):
                    interval = visit_times[i] - visit_times[i - 1]
                    intervals.append(interval)

        return intervals

    def compute_coverage_metrics(self, visit_history, uav_positions_history):
        """计算覆盖相关指标"""
        metrics = {}

        # 确保visit_history是numpy数组
        if isinstance(visit_history, list):
            visit_history = np.array(visit_history)
        
        # 覆盖率
        total_visits = np.sum(visit_history, axis=0)
        coverage_ratio = np.sum(total_visits > 0) / visit_history.shape[1]
        metrics['coverage_ratio'] = coverage_ratio

        # 热点优先覆盖率
        hotspot_coverage = self._compute_hotspot_coverage(visit_history)
        metrics['hotspot_coverage'] = hotspot_coverage

        # 空间效率
        spatial_efficiency = self._compute_spatial_efficiency(uav_positions_history)
        metrics['spatial_efficiency'] = spatial_efficiency

        return metrics

    def _compute_hotspot_coverage(self, visit_history):
        """计算热点覆盖率"""
        # 确保visit_history是numpy数组
        if isinstance(visit_history, list):
            visit_history = np.array(visit_history)
            
        hotspot_indices = []

        for k in range(visit_history.shape[1]):
            if self._get_grid_type(k) == 'hotspot':
                hotspot_indices.append(k)

        if not hotspot_indices:
            return 0.0

        hotspot_visits = visit_history[:, hotspot_indices]
        total_hotspot_visits = np.sum(hotspot_visits, axis=0)

        covered_hotspots = np.sum(total_hotspot_visits > 0)
        return covered_hotspots / len(hotspot_indices)

    def _compute_spatial_efficiency(self, uav_positions_history):
        """计算空间效率"""
        if not uav_positions_history:
            return 0.0

        # 计算UAV位置的标准差（衡量分布均匀性）
        positions = np.array(uav_positions_history)
        std_positions = np.std(positions, axis=0)

        # 空间效率 = 1 / (1 + 位置标准差)
        efficiency = 1 / (1 + np.mean(std_positions))

        return efficiency


class RobustnessMetrics:
    """鲁棒性指标"""

    def __init__(self, environment):
        self.environment = environment

    def compute_iss_metrics(self, state_history, disturbance_history):
        """计算ISS相关指标"""
        if not state_history or not disturbance_history:
            return {}

        metrics = {}

        # 计算状态包络
        states = np.array(state_history)
        disturbances = np.array(disturbance_history)

        # 状态范数
        state_norms = np.linalg.norm(states, axis=1)
        disturbance_norms = np.linalg.norm(disturbances, axis=1)

        # ISS关系：||x||_∞ vs ||d||_∞
        max_state_norm = np.max(state_norms)
        max_disturbance_norm = np.max(disturbance_norms)

        metrics['max_state_norm'] = max_state_norm
        metrics['max_disturbance_norm'] = max_disturbance_norm
        metrics['iss_ratio'] = max_state_norm / (1 + max_disturbance_norm)

        # 稳态偏差
        steady_state_norm = np.mean(state_norms[-100:])  # 最后100步的平均
        metrics['steady_state_deviation'] = steady_state_norm

        return metrics

    def compute_aoi_probability_guarantee(self, aoi_history, disturbance_type, delta=0.05):
        """计算AoI概率保证"""
        if not aoi_history:
            return {}

        metrics = {}

        # 计算违反阈值的概率
        aoi_array = np.array(aoi_history)

        # 根据网格类型确定阈值
        violations = 0
        total_checks = 0

        for t in range(aoi_array.shape[0]):
            for k in range(aoi_array.shape[1]):
                grid_type = self._get_grid_type(k)
                threshold = self._get_aoi_threshold(grid_type)

                if aoi_array[t, k] > threshold:
                    violations += 1
                total_checks += 1

        violation_probability = violations / total_checks
        metrics['aoi_violation_probability'] = violation_probability
        metrics['probability_guarantee_satisfied'] = violation_probability <= delta

        return metrics

    def _get_grid_type(self, grid_index):
        """获取网格类型"""
        density = self.environment.density_map[grid_index]
        density_threshold = np.percentile(self.environment.density_map, 80)

        if density > density_threshold:
            return 'hotspot'
        elif density > np.median(self.environment.density_map):
            return 'corridor'
        else:
            return 'base'

    def _get_aoi_threshold(self, grid_type):
        """获取AoI阈值"""
        thresholds = {
            'base': 60,
            'hotspot': 30,
            'corridor': 45
        }
        return thresholds.get(grid_type, 60)

    def compute_collision_metrics(self, uav_positions_history, obstacle_positions_history=None):
        """计算碰撞指标"""
        if not uav_positions_history:
            return {}

        metrics = {}
        positions = np.array(uav_positions_history)

        # UAV间碰撞
        uav_collisions = 0
        for t in range(positions.shape[0]):
            for i in range(positions.shape[1]):
                for j in range(i + 1, positions.shape[1]):
                    distance = np.linalg.norm(positions[t, i] - positions[t, j])
                    if distance < 5:  # 安全距离
                        uav_collisions += 1

        metrics['uav_collisions'] = uav_collisions

        # 与障碍碰撞
        obstacle_collisions = 0
        if obstacle_positions_history:
            for t in range(positions.shape[0]):
                for i in range(positions.shape[1]):
                    uav_pos = positions[t, i]
                    for obstacle in self.environment.obstacles:
                        if obstacle.contains(uav_pos):
                            obstacle_collisions += 1
                            break

        metrics['obstacle_collisions'] = obstacle_collisions
        metrics['total_collisions'] = uav_collisions + obstacle_collisions

        return metrics


class PerformanceMetrics:
    """性能指标"""

    def __init__(self):
        pass

    def compute_computational_metrics(self, planning_times):
        """计算计算性能指标"""
        if not planning_times:
            return {}

        times = np.array(planning_times)

        metrics = {
            'mean_planning_time': np.mean(times),
            'max_planning_time': np.max(times),
            'min_planning_time': np.min(times),
            'std_planning_time': np.std(times),
            'p90_planning_time': np.percentile(times, 90),
            'p95_planning_time': np.percentile(times, 95)
        }

        return metrics

    def compute_communication_metrics(self, message_counts, message_sizes):
        """计算通信指标"""
        if not message_counts or not message_sizes:
            return {}

        counts = np.array(message_counts)
        sizes = np.array(message_sizes)

        metrics = {
            'total_messages': np.sum(counts),
            'average_messages_per_step': np.mean(counts),
            'total_communication_bytes': np.sum(sizes),
            'average_bytes_per_step': np.mean(sizes)
        }

        return metrics

    def compute_energy_metrics(self, control_inputs_history):
        """计算能耗指标"""
        if not control_inputs_history:
            return {}

        energy_consumption = []
        trajectory_lengths = []

        for step_inputs in control_inputs_history:
            # 控制输入平方和作为能耗代理
            step_energy = np.sum([np.sum(u ** 2) for u in step_inputs])
            energy_consumption.append(step_energy)

            # 轨迹长度
            if len(step_inputs) > 0:
                step_length = np.sum([np.linalg.norm(u) for u in step_inputs])
                trajectory_lengths.append(step_length)

        metrics = {
            'total_energy_consumption': np.sum(energy_consumption),
            'average_energy_per_step': np.mean(energy_consumption),
            'total_trajectory_length': np.sum(trajectory_lengths),
            'average_trajectory_length': np.mean(trajectory_lengths)
        }

        return metrics


class StatisticalAnalysis:
    """统计分析"""

    def __init__(self):
        pass

    def perform_wilcoxon_test(self, method1_metrics, method2_metrics, metric_name):
        """执行Wilcoxon符号秩检验"""
        if len(method1_metrics) != len(method2_metrics):
            return None

        # 提取指定指标
        values1 = [m[metric_name] for m in method1_metrics if metric_name in m]
        values2 = [m[metric_name] for m in method2_metrics if metric_name in m]

        if len(values1) != len(values2) or len(values1) < 3:
            return None

        # 执行检验
        statistic, p_value = stats.wilcoxon(values1, values2)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    def perform_friedman_test(self, all_methods_metrics, metric_name):
        """执行Friedman检验"""
        # 提取所有方法的指定指标
        method_values = []
        method_names = []

        for method_name, metrics_list in all_methods_metrics.items():
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            if values:
                method_values.append(values)
                method_names.append(method_name)

        if len(method_values) < 2:
            return None

        # 确保所有方法有相同数量的数据
        min_length = min(len(values) for values in method_values)
        method_values = [values[:min_length] for values in method_values]

        # 执行Friedman检验
        statistic, p_value = stats.friedmanchisquare(*method_values)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'method_names': method_names
        }

    def compute_confidence_intervals(self, metrics_list, metric_name, confidence=0.95):
        """计算置信区间"""
        values = [m[metric_name] for m in metrics_list if metric_name in m]

        if len(values) < 2:
            return None

        # Bootstrap方法计算置信区间
        n_bootstrap = 5000
        bootstrap_means = []

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        # 计算置信区间
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence': confidence
        }


class VisualizationTools:
    """可视化工具"""

    def __init__(self):
        # 定义颜色方案
        self.colors = {
            'Ours_Full': '#1f77b4',  # 蓝色
            'Ours_Light': '#ff7f0e',  # 橙色
            'Unweighted_Voronoi': '#2ca02c',  # 绿色
            'Power_Voronoi': '#d62728',  # 红色
            'CBBA': '#9467bd',  # 紫色
            'Frontier_Exploration': '#8c564b',  # 棕色
            'Round_Robin': '#e377c2',  # 粉色
            'Greedy_AoI': '#7f7f7f',  # 灰色
            'Centralized_MPC': '#bcbd22'  # 黄绿色
        }

        self.line_styles = {
            'Ours_Full': '-',
            'Ours_Light': '--',
            'Unweighted_Voronoi': '-.',
            'Power_Voronoi': ':',
            'CBBA': '-',
            'Frontier_Exploration': '--',
            'Round_Robin': '-.',
            'Greedy_AoI': ':',
            'Centralized_MPC': '-'
        }

    def plot_aoi_comparison_curves(self, results_data, task_name, uav_count, output_dir):
        """绘制多个方法的AoI对比曲线（均值+阴影）"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 收集所有方法的数据
        method_data = {}
        time_steps = None

        for seed, seed_results in results_data.items():
            for method, method_results in seed_results.items():
                if method not in method_data:
                    method_data[method] = []

                aoi_history = method_results['data']['aoi_history']
                if time_steps is None:
                    time_steps = np.arange(len(aoi_history))

                # 计算平均AoI
                aoi_array = np.array(aoi_history)
                mean_aoi = np.mean(aoi_array, axis=1)
                method_data[method].append(mean_aoi)

        # 绘制对比曲线
        for method, data_list in method_data.items():
            if method in self.colors:
                data_array = np.array(data_list)
                mean_curve = np.mean(data_array, axis=0)
                std_curve = np.std(data_array, axis=0)

                # 计算95%置信区间
                ci_95 = 1.96 * std_curve / np.sqrt(len(data_list))

                # 绘制均值曲线
                ax1.plot(time_steps, mean_curve,
                         color=self.colors[method],
                         linestyle=self.line_styles[method],
                         linewidth=2,
                         label=method)

                # 绘制阴影区域
                ax1.fill_between(time_steps,
                                 mean_curve - ci_95,
                                 mean_curve + ci_95,
                                 color=self.colors[method],
                                 alpha=0.3)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Average AoI (s)')
        ax1.set_title(f'AoI Comparison - {task_name} (UAV Count: {uav_count})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 绘制最大AoI对比
        method_max_data = {}
        for seed, seed_results in results_data.items():
            for method, method_results in seed_results.items():
                if method not in method_max_data:
                    method_max_data[method] = []

                aoi_history = method_results['data']['aoi_history']
                aoi_array = np.array(aoi_history)
                max_aoi = np.max(aoi_array, axis=1)
                method_max_data[method].append(max_aoi)

        for method, data_list in method_max_data.items():
            if method in self.colors:
                data_array = np.array(data_list)
                mean_curve = np.mean(data_array, axis=0)
                std_curve = np.std(data_array, axis=0)
                ci_95 = 1.96 * std_curve / np.sqrt(len(data_list))

                ax2.plot(time_steps, mean_curve,
                         color=self.colors[method],
                         linestyle=self.line_styles[method],
                         linewidth=2,
                         label=method)

                ax2.fill_between(time_steps,
                                 mean_curve - ci_95,
                                 mean_curve + ci_95,
                                 color=self.colors[method],
                                 alpha=0.3)

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Max AoI (s)')
        ax2.set_title(f'Max AoI Comparison - {task_name} (UAV Count: {uav_count})')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'aoi_comparison_{task_name}_uav{uav_count}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_deadline_violation_comparison(self, results_data, task_name, uav_count, output_dir):
        """绘制Deadline违反率对比曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))

        method_data = {}
        time_steps = None

        for seed, seed_results in results_data.items():
            for method, method_results in seed_results.items():
                if method not in method_data:
                    method_data[method] = []

                aoi_history = method_results['data']['aoi_history']
                if time_steps is None:
                    time_steps = np.arange(len(aoi_history))

                # 计算违反率（简化版本）
                aoi_array = np.array(aoi_history)
                violation_rates = []
                for t in range(aoi_array.shape[0]):
                    # 假设阈值为30秒
                    violations = np.sum(aoi_array[t, :] > 30)
                    violation_rate = violations / aoi_array.shape[1]
                    violation_rates.append(violation_rate)

                method_data[method].append(violation_rates)

        # 绘制对比曲线
        for method, data_list in method_data.items():
            if method in self.colors:
                data_array = np.array(data_list)
                mean_curve = np.mean(data_array, axis=0)
                std_curve = np.std(data_array, axis=0)
                ci_95 = 1.96 * std_curve / np.sqrt(len(data_list))

                ax.plot(time_steps, mean_curve,
                        color=self.colors[method],
                        linestyle=self.line_styles[method],
                        linewidth=2,
                        label=method)

                ax.fill_between(time_steps,
                                mean_curve - ci_95,
                                mean_curve + ci_95,
                                color=self.colors[method],
                                alpha=0.3)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Deadline Violation Rate')
        ax.set_title(f'Deadline Violation Rate Comparison - {task_name} (UAV Count: {uav_count})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.05, color='k', linestyle='--', alpha=0.7, label='δ = 0.05')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'deadline_violation_{task_name}_uav{uav_count}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_computation_time_comparison(self, results_data, task_name, uav_count, output_dir):
        """绘制计算时间对比（箱线图+均值曲线）"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 箱线图
        method_times = {}
        for seed, seed_results in results_data.items():
            for method, method_results in seed_results.items():
                if method not in method_times:
                    method_times[method] = []

                planning_times = method_results['data']['planning_times']
                method_times[method].extend(planning_times)

        # 准备箱线图数据
        data = []
        labels = []
        for method, times in method_times.items():
            if method in self.colors:
                data.append(times)
                labels.append(method)

        bp = ax1.boxplot(data, labels=labels, patch_artist=True)
        for patch, method in zip(bp['boxes'], labels):
            if method in self.colors:
                patch.set_facecolor(self.colors[method])
                patch.set_alpha(0.7)

        ax1.set_ylabel('Planning Time (s)')
        ax1.set_title(f'Computation Time Distribution - {task_name} (UAV Count: {uav_count})')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # 时间序列图
        method_time_series = {}
        time_steps = None

        for seed, seed_results in results_data.items():
            for method, method_results in seed_results.items():
                if method not in method_time_series:
                    method_time_series[method] = []

                planning_times = method_results['data']['planning_times']
                if time_steps is None:
                    time_steps = np.arange(len(planning_times))

                method_time_series[method].append(planning_times)

        for method, data_list in method_time_series.items():
            if method in self.colors:
                data_array = np.array(data_list)
                mean_curve = np.mean(data_array, axis=0)
                std_curve = np.std(data_array, axis=0)
                ci_95 = 1.96 * std_curve / np.sqrt(len(data_list))

                ax2.plot(time_steps, mean_curve,
                         color=self.colors[method],
                         linestyle=self.line_styles[method],
                         linewidth=2,
                         label=method)

                ax2.fill_between(time_steps,
                                 mean_curve - ci_95,
                                 mean_curve + ci_95,
                                 color=self.colors[method],
                                 alpha=0.3)

        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Planning Time (s)')
        ax2.set_title(f'Computation Time Over Time - {task_name} (UAV Count: {uav_count})')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'computation_time_{task_name}_uav{uav_count}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_coverage_comparison(self, results_data, task_name, uav_count, output_dir):
        """绘制覆盖率对比曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))

        method_data = {}
        time_steps = None

        for seed, seed_results in results_data.items():
            for method, method_results in seed_results.items():
                if method not in method_data:
                    method_data[method] = []

                visit_history = method_results['data']['visit_history']
                if time_steps is None:
                    time_steps = np.arange(len(visit_history))

                # 计算覆盖率
                coverage_rates = []
                total_grids = visit_history[0].shape[0] if visit_history else 0

                for t in range(len(visit_history)):
                    visited_grids = np.sum(visit_history[t] > 0)
                    coverage_rate = visited_grids / total_grids if total_grids > 0 else 0
                    coverage_rates.append(coverage_rate)

                method_data[method].append(coverage_rates)

        # 绘制对比曲线
        for method, data_list in method_data.items():
            if method in self.colors:
                data_array = np.array(data_list)
                mean_curve = np.mean(data_array, axis=0)
                std_curve = np.std(data_array, axis=0)
                ci_95 = 1.96 * std_curve / np.sqrt(len(data_list))

                ax.plot(time_steps, mean_curve,
                        color=self.colors[method],
                        linestyle=self.line_styles[method],
                        linewidth=2,
                        label=method)

                ax.fill_between(time_steps,
                                mean_curve - ci_95,
                                mean_curve + ci_95,
                                color=self.colors[method],
                                alpha=0.3)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Coverage Rate')
        ax.set_title(f'Coverage Rate Comparison - {task_name} (UAV Count: {uav_count})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'coverage_comparison_{task_name}_uav{uav_count}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_energy_consumption_comparison(self, results_data, task_name, uav_count, output_dir):
        """绘制能耗对比曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))

        method_data = {}
        time_steps = None

        for seed, seed_results in results_data.items():
            for method, method_results in seed_results.items():
                if method not in method_data:
                    method_data[method] = []

                control_inputs = method_results['data']['control_inputs_history']
                if time_steps is None:
                    time_steps = np.arange(len(control_inputs))

                # 计算能耗（基于控制输入的平方和）
                energy_consumption = []
                for t in range(len(control_inputs)):
                    inputs = np.array(control_inputs[t])
                    energy = np.sum(inputs ** 2)  # 简化的能耗模型
                    energy_consumption.append(energy)

                method_data[method].append(energy_consumption)

        # 绘制对比曲线
        for method, data_list in method_data.items():
            if method in self.colors:
                data_array = np.array(data_list)
                mean_curve = np.mean(data_array, axis=0)
                std_curve = np.std(data_array, axis=0)
                ci_95 = 1.96 * std_curve / np.sqrt(len(data_list))

                ax.plot(time_steps, mean_curve,
                        color=self.colors[method],
                        linestyle=self.line_styles[method],
                        linewidth=2,
                        label=method)

                ax.fill_between(time_steps,
                                mean_curve - ci_95,
                                mean_curve + ci_95,
                                color=self.colors[method],
                                alpha=0.3)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy Consumption')
        ax.set_title(f'Energy Consumption Comparison - {task_name} (UAV Count: {uav_count})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'energy_consumption_{task_name}_uav{uav_count}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_quantization_functional_comparison(self, results_data, task_name, uav_count, output_dir):
        """绘制量化函数对比曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))

        method_data = {}
        time_steps = None

        for seed, seed_results in results_data.items():
            for method, method_results in seed_results.items():
                if method not in method_data:
                    method_data[method] = []

                # 从指标中获取量化函数值
                metrics = method_results['metrics']
                if 'quantization_functional' in metrics:
                    quantization_values = [metrics['quantization_functional']]  # 简化，实际应该是时间序列
                else:
                    quantization_values = [0]  # 默认值

                if time_steps is None:
                    time_steps = np.arange(len(quantization_values))

                method_data[method].append(quantization_values)

        # 绘制对比曲线
        for method, data_list in method_data.items():
            if method in self.colors:
                data_array = np.array(data_list)
                mean_curve = np.mean(data_array, axis=0)
                std_curve = np.std(data_array, axis=0)
                ci_95 = 1.96 * std_curve / np.sqrt(len(data_list))

                ax.plot(time_steps, mean_curve,
                        color=self.colors[method],
                        linestyle=self.line_styles[method],
                        linewidth=2,
                        label=method)

                ax.fill_between(time_steps,
                                mean_curve - ci_95,
                                mean_curve + ci_95,
                                color=self.colors[method],
                                alpha=0.3)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Quantization Functional H(P)')
        ax.set_title(f'Quantization Functional Comparison - {task_name} (UAV Count: {uav_count})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'quantization_functional_{task_name}_uav{uav_count}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_aoi_trajectory(self, aoi_history, time_steps, ax=None):
        """绘制AoI轨迹"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        aoi_array = np.array(aoi_history)

        # 绘制平均AoI和最大AoI
        mean_aoi = np.mean(aoi_array, axis=1)
        max_aoi = np.max(aoi_array, axis=1)

        ax.plot(time_steps, mean_aoi, 'b-', linewidth=2, label='Average AoI')
        ax.plot(time_steps, max_aoi, 'r-', linewidth=2, label='Max AoI')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Age of Information (s)')
        ax.set_title('AoI Trajectory Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_deadline_violations(self, aoi_history, thresholds, ax=None):
        """绘制Deadline违反率"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        aoi_array = np.array(aoi_history)

        # 计算违反率
        violation_rates = []
        for t in range(aoi_array.shape[0]):
            violations = 0
            total = 0

            for k in range(aoi_array.shape[1]):
                grid_type = self._get_grid_type(k)
                threshold = thresholds[grid_type]

                if aoi_array[t, k] > threshold:
                    violations += 1
                total += 1

            violation_rates.append(violations / total)

        ax.plot(violation_rates, 'r-', linewidth=2)
        ax.axhline(y=0.05, color='k', linestyle='--', label='δ = 0.05')

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Deadline Violation Rate')
        ax.set_title('Deadline Violation Rate Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_computation_time_boxplot(self, all_methods_times, ax=None):
        """绘制计算时间箱线图"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # 准备数据
        data = []
        labels = []

        for method_name, times in all_methods_times.items():
            data.append(times)
            labels.append(method_name)

        ax.boxplot(data, labels=labels)
        ax.set_ylabel('Planning Time (s)')
        ax.set_title('Computation Time Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_trajectory_heatmap(self, uav_positions_history, environment, ax=None):
        """绘制轨迹热力图"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))

        positions = np.array(uav_positions_history)

        # 创建位置密度图
        x_positions = positions[:, :, 0].flatten()
        y_positions = positions[:, :, 1].flatten()

        # 创建2D直方图
        H, xedges, yedges = np.histogram2d(x_positions, y_positions, bins=50)

        # 绘制热力图
        im = ax.imshow(H.T, origin='lower', extent=[
            environment.workspace.domain[0], environment.workspace.domain[1],
            environment.workspace.domain[2], environment.workspace.domain[3]
        ], cmap='hot', alpha=0.7)

        # 添加颜色条
        plt.colorbar(im, ax=ax, label='Visit Frequency')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('UAV Trajectory Heatmap')

        return ax

    def plot_comprehensive_summary(self, results_data, task_name, uav_count, output_dir):
        """绘制综合性能对比总结图"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Comprehensive Performance Comparison - {task_name} (UAV Count: {uav_count})',
                     fontsize=16, fontweight='bold')

        # 收集所有方法的汇总指标
        method_summary = {}

        for seed, seed_results in results_data.items():
            for method, method_results in seed_results.items():
                if method not in method_summary:
                    method_summary[method] = {
                        'avg_aoi': [],
                        'max_aoi': [],
                        'deadline_rate': [],
                        'coverage_rate': [],
                        'planning_time': [],
                        'energy_consumption': []
                    }

                metrics = method_results['metrics']

                # 收集各项指标
                method_summary[method]['avg_aoi'].append(metrics.get('average_aoi', 0))
                method_summary[method]['max_aoi'].append(metrics.get('max_aoi', 0))
                method_summary[method]['deadline_rate'].append(metrics.get('deadline_satisfaction_rate', 0))
                method_summary[method]['coverage_rate'].append(metrics.get('coverage_rate', 0))
                method_summary[method]['planning_time'].append(metrics.get('mean_planning_time', 0))
                method_summary[method]['energy_consumption'].append(metrics.get('total_energy_consumption', 0))

        # 计算均值和置信区间
        summary_stats = {}
        for method, data in method_summary.items():
            if method in self.colors:
                summary_stats[method] = {}
                for metric, values in data.items():
                    values_array = np.array(values)
                    mean_val = np.mean(values_array)
                    std_val = np.std(values_array)
                    ci_95 = 1.96 * std_val / np.sqrt(len(values))

                    summary_stats[method][metric] = {
                        'mean': mean_val,
                        'std': std_val,
                        'ci_95': ci_95
                    }

        # 绘制各项指标对比
        metrics_to_plot = [
            ('avg_aoi', 'Average AoI (s)', axes[0, 0]),
            ('max_aoi', 'Max AoI (s)', axes[0, 1]),
            ('deadline_rate', 'Deadline Satisfaction Rate', axes[0, 2]),
            ('coverage_rate', 'Coverage Rate', axes[1, 0]),
            ('planning_time', 'Mean Planning Time (s)', axes[1, 1]),
            ('energy_consumption', 'Total Energy Consumption', axes[1, 2])
        ]

        for metric_key, metric_name, ax in metrics_to_plot:
            methods = []
            means = []
            cis = []
            colors_list = []

            for method, stats in summary_stats.items():
                if metric_key in stats:
                    methods.append(method)
                    means.append(stats[metric_key]['mean'])
                    cis.append(stats[metric_key]['ci_95'])
                    colors_list.append(self.colors[method])

            # 绘制柱状图
            x_pos = np.arange(len(methods))
            bars = ax.bar(x_pos, means, yerr=cis, capsize=5,
                          color=colors_list, alpha=0.7, edgecolor='black')

            ax.set_xlabel('Methods')
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            # 添加数值标签
            for i, (bar, mean_val) in enumerate(zip(bars, means)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + cis[i] + 0.01,
                        f'{mean_val:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comprehensive_summary_{task_name}_uav{uav_count}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_method_ranking(self, results_data, task_name, uav_count, output_dir):
        """绘制方法排名对比图"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # 计算每个方法的综合得分
        method_scores = {}

        for seed, seed_results in results_data.items():
            for method, method_results in seed_results.items():
                if method not in method_scores:
                    method_scores[method] = []

                metrics = method_results['metrics']

                # 计算综合得分（归一化后的加权平均）
                # 这里使用简化的评分方法，实际可以根据具体需求调整
                avg_aoi = metrics.get('average_aoi', 0)
                deadline_rate = metrics.get('deadline_satisfaction_rate', 0)
                coverage_rate = metrics.get('coverage_rate', 0)
                planning_time = metrics.get('mean_planning_time', 0)

                # 归一化得分（越小越好或越大越好）
                score = (deadline_rate + coverage_rate) / (1 + avg_aoi / 100 + planning_time / 10)
                method_scores[method].append(score)

        # 计算平均得分和置信区间
        final_scores = {}
        for method, scores in method_scores.items():
            if method in self.colors:
                scores_array = np.array(scores)
                mean_score = np.mean(scores_array)
                std_score = np.std(scores_array)
                ci_95 = 1.96 * std_score / np.sqrt(len(scores))

                final_scores[method] = {
                    'mean': mean_score,
                    'ci_95': ci_95
                }

        # 按得分排序
        sorted_methods = sorted(final_scores.items(), key=lambda x: x[1]['mean'], reverse=True)

        methods = [item[0] for item in sorted_methods]
        means = [item[1]['mean'] for item in sorted_methods]
        cis = [item[1]['ci_95'] for item in sorted_methods]
        colors_list = [self.colors[method] for method in methods]

        # 绘制排名图
        x_pos = np.arange(len(methods))
        bars = ax.barh(x_pos, means, xerr=cis, capsize=5,
                       color=colors_list, alpha=0.7, edgecolor='black')

        ax.set_xlabel('Comprehensive Performance Score')
        ax.set_ylabel('Methods')
        ax.set_title(f'Method Ranking - {task_name} (UAV Count: {uav_count})')
        ax.set_yticks(x_pos)
        ax.set_yticklabels(methods)
        ax.grid(True, alpha=0.3, axis='x')

        # 添加排名标签
        for i, (bar, mean_val, rank) in enumerate(zip(bars, means, range(1, len(methods) + 1))):
            width = bar.get_width()
            ax.text(width + cis[i] + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'#{rank}', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'method_ranking_{task_name}_uav{uav_count}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _get_grid_type(self, grid_index):
        """获取网格类型（简化版本）"""
        return 'base'  # 简化实现
