"""
主实验脚本：整合所有模块并执行完整实验
"""
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from src.env.environment import create_environment_from_config
from src.models.uav_dynamics import UAVFleet, DisturbanceGenerator
from src.partition.power_voronoi import PowerVoronoiPartition
from src.planner.rhc_planner import DistributedRHCPlanner
from src.baselines.baseline_methods import (
    UnweightedVoronoiBaseline, PowerVoronoiBaseline, CBBABaseline,
    FrontierExplorationBaseline, RoundRobinBaseline, GreedyAoIBaseline,
    CentralizedMPCBaseline
)
from src.eval.metrics import (
    CoverageMetrics, RobustnessMetrics, PerformanceMetrics,
    StatisticalAnalysis, VisualizationTools
)

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}

    def _load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def setup_experiment(self, task_name, uav_count, random_seed):
        """设置实验环境"""
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 获取任务配置
        task_config = self.config['tasks'][task_name]

        # 创建环境
        environment = create_environment_from_config(self.config)

        # 创建UAV机队
        uav_fleet = UAVFleet(self.config['uav'], environment)
        uav_fleet.create_uavs(uav_count)

        # 创建分区器
        partition = PowerVoronoiPartition(environment, uav_fleet)

        # 创建RHC规划器
        rhc_planner = DistributedRHCPlanner(uav_fleet, environment, partition, self.config)

        # 创建基线方法
        baselines = {
            'Unweighted_Voronoi': UnweightedVoronoiBaseline(uav_fleet, environment, self.config),
            'Power_Voronoi': PowerVoronoiBaseline(uav_fleet, environment, self.config),
            'CBBA': CBBABaseline(uav_fleet, environment, self.config),
            'Frontier_Exploration': FrontierExplorationBaseline(uav_fleet, environment, self.config),
            'Round_Robin': RoundRobinBaseline(uav_fleet, environment, self.config),
            'Greedy_AoI': GreedyAoIBaseline(uav_fleet, environment, self.config),
            'Centralized_MPC': CentralizedMPCBaseline(uav_fleet, environment, self.config)
        }

        # 创建干扰生成器
        disturbance_config = self.config['disturbance'][task_config['disturbance']]
        disturbance_generator = DisturbanceGenerator(
            task_config['disturbance'], disturbance_config
        )

        # 创建评估指标
        aoi_thresholds = {
            'base': self.config['aoi_thresholds']['base'],
            'hotspot': self.config['aoi_thresholds']['hotspot'],
            'corridor': self.config['aoi_thresholds']['corridor']
        }

        coverage_metrics = CoverageMetrics(environment, aoi_thresholds)
        robustness_metrics = RobustnessMetrics(environment)
        performance_metrics = PerformanceMetrics()
        statistical_analysis = StatisticalAnalysis()
        visualization_tools = VisualizationTools()

        return {
            'environment': environment,
            'uav_fleet': uav_fleet,
            'partition': partition,
            'rhc_planner': rhc_planner,
            'baselines': baselines,
            'disturbance_generator': disturbance_generator,
            'coverage_metrics': coverage_metrics,
            'robustness_metrics': robustness_metrics,
            'performance_metrics': performance_metrics,
            'statistical_analysis': statistical_analysis,
            'visualization_tools': visualization_tools,
            'task_config': task_config
        }

    def run_single_experiment(self, method_name, components, duration):
        """运行单个实验"""
        environment = components['environment']
        uav_fleet = components['uav_fleet']
        partition = components['partition']
        rhc_planner = components['rhc_planner']
        baselines = components['baselines']
        disturbance_generator = components['disturbance_generator']

        # 初始化历史记录
        aoi_history = []
        visit_history = []
        uav_positions_history = []
        control_inputs_history = []
        planning_times = []
        state_history = []
        disturbance_history = []

        # 初始化AoI状态
        num_grids = environment.workspace.M
        aoi_states = np.zeros(num_grids)

        # 主循环
        for t in tqdm(range(duration), desc=f"Running {method_name}"):
            start_time = time.time()

            # 生成干扰
            disturbance = disturbance_generator.generate(t)
            disturbance_history.append(disturbance)

            # 选择规划方法
            if method_name == 'Ours_Full':
                control_inputs = rhc_planner.plan_step(t, aoi_states)
            elif method_name == 'Ours_Light':
                # 简化版本：不使用动态障碍前馈收紧
                control_inputs = rhc_planner.plan_step(t, aoi_states)
            elif method_name in baselines:
                control_inputs = baselines[method_name].plan_step(t, aoi_states)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            # 记录规划时间
            planning_time = time.time() - start_time
            planning_times.append(planning_time)

            # 更新UAV状态
            uav_fleet.step(control_inputs, [disturbance] * len(uav_fleet.uavs))

            # 更新环境
            environment.update_dynamic_obstacles(self.config['uav']['sampling_period'])

            # 更新AoI和访问记录
            visit_indicator = np.zeros(num_grids)
            for i, uav in enumerate(uav_fleet.uavs):
                uav_pos = uav.get_position()
                for k, grid_center in enumerate(environment.workspace.grid_centers):
                    if environment.workspace.is_visited(uav_pos, grid_center):
                        visit_indicator[k] = 1
                        aoi_states[k] = 0
                    else:
                        aoi_states[k] += 1

            # 记录历史
            aoi_history.append(aoi_states.copy())
            visit_history.append(visit_indicator.copy())
            uav_positions_history.append(uav_fleet.get_positions())
            control_inputs_history.append(control_inputs)
            state_history.append(uav_fleet.get_states())

        return {
            'aoi_history': aoi_history,
            'visit_history': visit_history,
            'uav_positions_history': uav_positions_history,
            'control_inputs_history': control_inputs_history,
            'planning_times': planning_times,
            'state_history': state_history,
            'disturbance_history': disturbance_history
        }

    def compute_metrics(self, experiment_data, components):
        """计算实验指标"""
        coverage_metrics = components['coverage_metrics']
        robustness_metrics = components['robustness_metrics']
        performance_metrics = components['performance_metrics']

        # 覆盖指标
        aoi_metrics = coverage_metrics.compute_aoi_metrics(
            experiment_data['aoi_history'],
            experiment_data['visit_history']
        )

        coverage_metrics_result = coverage_metrics.compute_coverage_metrics(
            experiment_data['visit_history'],
            experiment_data['uav_positions_history']
        )

        # 鲁棒性指标
        iss_metrics = robustness_metrics.compute_iss_metrics(
            experiment_data['state_history'],
            experiment_data['disturbance_history']
        )

        aoi_probability_metrics = robustness_metrics.compute_aoi_probability_guarantee(
            experiment_data['aoi_history'],
            'bounded'
        )

        collision_metrics = robustness_metrics.compute_collision_metrics(
            experiment_data['uav_positions_history']
        )

        # 性能指标
        computational_metrics = performance_metrics.compute_computational_metrics(
            experiment_data['planning_times']
        )

        energy_metrics = performance_metrics.compute_energy_metrics(
            experiment_data['control_inputs_history']
        )

        # 合并所有指标
        all_metrics = {}
        all_metrics.update(aoi_metrics)
        all_metrics.update(coverage_metrics_result)
        all_metrics.update(iss_metrics)
        all_metrics.update(aoi_probability_metrics)
        all_metrics.update(collision_metrics)
        all_metrics.update(computational_metrics)
        all_metrics.update(energy_metrics)

        return all_metrics

    def run_task(self, task_name):
        """运行完整任务"""
        print(f"\n=== Running Task: {task_name} ===")

        task_config = self.config['tasks'][task_name]
        uav_counts = task_config['uav_count']
        random_seeds = self.config['experiment']['random_seeds']
        duration = self.config['experiment']['duration']

        task_results = {}

        for uav_count in uav_counts:
            print(f"\n--- UAV Count: {uav_count} ---")
            uav_count_results = {}

            for seed in random_seeds:
                print(f"\nSeed: {seed}")
                seed_results = {}

                # 设置实验
                components = self.setup_experiment(task_name, uav_count, seed)

                # 定义要测试的方法
                methods = [
                    'Ours_Full',
                    'Ours_Light',
                    'Unweighted_Voronoi',
                    'Power_Voronoi',
                    'CBBA',
                    'Frontier_Exploration',
                    'Round_Robin',
                    'Greedy_AoI',
                    'Centralized_MPC'
                ]

                for method in methods:
                    print(f"  Testing {method}...")

                    # 重新设置实验（确保每个方法从相同初始状态开始）
                    components = self.setup_experiment(task_name, uav_count, seed)

                    # 运行实验
                    experiment_data = self.run_single_experiment(method, components, duration)

                    # 计算指标
                    metrics = self.compute_metrics(experiment_data, components)

                    # 保存结果
                    seed_results[method] = {
                        'metrics': metrics,
                        'data': experiment_data
                    }

                uav_count_results[seed] = seed_results

            task_results[uav_count] = uav_count_results

        return task_results

    def run_all_tasks(self):
        """运行所有任务"""
        print("Starting comprehensive experiment...")

        all_results = {}

        for task_name in self.config['tasks'].keys():
            task_results = self.run_task(task_name)
            all_results[task_name] = task_results

        self.results = all_results
        return all_results

    def save_results(self, output_dir="results"):
        """保存实验结果"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存原始结果
        results_file = os.path.join(output_dir, f"experiment_results_{timestamp}.npz")
        np.savez_compressed(results_file, results=self.results)

        # 保存汇总指标
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.csv")
        self._save_summary_csv(summary_file)

        # 生成可视化
        self._generate_visualizations(output_dir, timestamp)

        print(f"Results saved to {output_dir}")

    def _save_summary_csv(self, filename):
        """保存汇总CSV"""
        summary_data = []

        for task_name, task_results in self.results.items():
            for uav_count, uav_results in task_results.items():
                for seed, seed_results in uav_results.items():
                    for method, method_results in seed_results.items():
                        metrics = method_results['metrics']

                        row = {
                            'task': task_name,
                            'uav_count': uav_count,
                            'seed': seed,
                            'method': method
                        }
                        row.update(metrics)
                        summary_data.append(row)

        df = pd.DataFrame(summary_data)
        df.to_csv(filename, index=False)

    def _generate_visualizations(self, output_dir, timestamp):
        """生成可视化图表"""
        visualization_tools = VisualizationTools()

        # 为每个任务生成图表
        for task_name, task_results in self.results.items():
            task_dir = os.path.join(output_dir, f"task_{task_name}_{timestamp}")
            os.makedirs(task_dir, exist_ok=True)

            print(f"\nGenerating visualizations for {task_name}...")

            # 为每个UAV数量生成对比图表
            for uav_count, uav_results in task_results.items():
                print(f"  UAV Count: {uav_count}")

                # 生成AoI对比曲线
                visualization_tools.plot_aoi_comparison_curves(
                    uav_results, task_name, uav_count, task_dir
                )

                # 生成Deadline违反率对比曲线
                visualization_tools.plot_deadline_violation_comparison(
                    uav_results, task_name, uav_count, task_dir
                )

                # 生成计算时间对比
                visualization_tools.plot_computation_time_comparison(
                    uav_results, task_name, uav_count, task_dir
                )

                # 生成覆盖率对比曲线
                visualization_tools.plot_coverage_comparison(
                    uav_results, task_name, uav_count, task_dir
                )

                # 生成能耗对比曲线
                visualization_tools.plot_energy_consumption_comparison(
                    uav_results, task_name, uav_count, task_dir
                )

                # 生成量化函数对比曲线
                visualization_tools.plot_quantization_functional_comparison(
                    uav_results, task_name, uav_count, task_dir
                )

                # 生成综合性能对比总结图
                visualization_tools.plot_comprehensive_summary(
                    uav_results, task_name, uav_count, task_dir
                )

                # 生成方法排名对比图
                visualization_tools.plot_method_ranking(
                    uav_results, task_name, uav_count, task_dir
                )

            print(f"Visualizations for {task_name} saved to {task_dir}")

    def print_summary(self):
        """打印实验摘要"""
        print("\n=== Experiment Summary ===")

        for task_name, task_results in self.results.items():
            print(f"\nTask: {task_name}")

            for uav_count, uav_results in task_results.items():
                print(f"  UAV Count: {uav_count}")

                # 计算平均指标
                method_metrics = {}

                for seed, seed_results in uav_results.items():
                    for method, method_results in seed_results.items():
                        if method not in method_metrics:
                            method_metrics[method] = []
                        method_metrics[method].append(method_results['metrics'])

                # 打印平均结果
                for method, metrics_list in method_metrics.items():
                    avg_aoi = np.mean([m['average_aoi'] for m in metrics_list])
                    avg_deadline_rate = np.mean([m['deadline_satisfaction_rate'] for m in metrics_list])
                    avg_planning_time = np.mean([m['mean_planning_time'] for m in metrics_list])

                    print(f"    {method}: Avg AoI={avg_aoi:.2f}, "
                          f"Deadline Rate={avg_deadline_rate:.3f}, "
                          f"Planning Time={avg_planning_time:.3f}s")


def main():
    """主函数"""
    # 配置文件路径
    config_path = "../config/experiment_config.yaml"

    # 创建实验运行器
    runner = ExperimentRunner(config_path)

    # 运行所有任务
    results = runner.run_all_tasks()

    # 保存结果
    runner.save_results()

    # 打印摘要
    runner.print_summary()

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
