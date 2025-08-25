#!/usr/bin/env python3
"""
简化实验运行脚本
用于快速测试和验证实验设置
"""
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from tqdm import tqdm

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from env.environment import create_environment_from_config
from models.uav_dynamics import UAVFleet, DisturbanceGenerator
from partition.power_voronoi import PowerVoronoiPartition
from planner.rhc_planner import DistributedRHCPlanner
from baselines.baseline_methods import UnweightedVoronoiBaseline, GreedyAoIBaseline
from eval.metrics import CoverageMetrics, RobustnessMetrics, PerformanceMetrics

def run_quick_test():
    """运行快速测试"""
    print("=== 快速测试模式 ===")
    
    # 加载配置
    config_path = "config/experiment_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 创建环境
    print("创建环境...")
    environment = create_environment_from_config(config)
    
    # 创建UAV机队
    print("创建UAV机队...")
    uav_fleet = UAVFleet(config['uav'], environment)
    uav_fleet.create_uavs(6)  # 使用6个UAV
    
    # 创建分区器
    partition = PowerVoronoiPartition(environment, uav_fleet)
    
    # 创建RHC规划器
    rhc_planner = DistributedRHCPlanner(uav_fleet, environment, partition, config)
    
    # 创建基线方法
    baseline = GreedyAoIBaseline(uav_fleet, environment, config)
    
    # 创建干扰生成器
    disturbance_generator = DisturbanceGenerator("bounded", config['disturbance']['bounded'])
    
    # 创建评估指标
    aoi_thresholds = {
        'base': config['aoi_thresholds']['base'],
        'hotspot': config['aoi_thresholds']['hotspot'],
        'corridor': config['aoi_thresholds']['corridor']
    }
    coverage_metrics = CoverageMetrics(environment, aoi_thresholds)
    
    # 运行短时间实验
    duration = 100  # 100步测试
    num_grids = environment.workspace.M
    aoi_states = np.zeros(num_grids)
    
    # 测试我们的方法
    print("测试我们的方法...")
    our_results = run_method_test("Ours", rhc_planner, uav_fleet, environment, 
                                disturbance_generator, aoi_states, duration)
    
    # 测试基线方法
    print("测试基线方法...")
    baseline_results = run_method_test("Baseline", baseline, uav_fleet, environment,
                                     disturbance_generator, aoi_states, duration)
    
    # 计算指标
    print("计算指标...")
    our_metrics = compute_metrics(our_results, coverage_metrics)
    baseline_metrics = compute_metrics(baseline_results, coverage_metrics)
    
    # 打印结果
    print("\n=== 测试结果 ===")
    print(f"我们的方法:")
    print(f"  平均AoI: {our_metrics['average_aoi']:.2f}")
    print(f"  Deadline满足率: {our_metrics['deadline_satisfaction_rate']:.3f}")
    print(f"  覆盖率: {our_metrics['coverage_ratio']:.3f}")
    
    print(f"\n基线方法:")
    print(f"  平均AoI: {baseline_metrics['average_aoi']:.2f}")
    print(f"  Deadline满足率: {baseline_metrics['deadline_satisfaction_rate']:.3f}")
    print(f"  覆盖率: {baseline_metrics['coverage_ratio']:.3f}")
    
    # 可视化
    print("\n生成可视化...")
    create_visualizations(our_results, baseline_results, environment)
    
    print("\n快速测试完成!")

def run_method_test(method_name, planner, uav_fleet, environment, 
                   disturbance_generator, aoi_states, duration):
    """运行方法测试"""
    # 初始化历史记录
    aoi_history = []
    visit_history = []
    uav_positions_history = []
    planning_times = []
    
    # 主循环
    for t in tqdm(range(duration), desc=f"Running {method_name}"):
        start_time = time.time()
        
        # 生成干扰
        disturbance = disturbance_generator.generate(t)
        
        # 规划
        if method_name == "Ours":
            control_inputs = planner.plan_step(t, aoi_states)
        else:
            control_inputs = planner.plan_step(t, aoi_states)
        
        # 记录规划时间
        planning_time = time.time() - start_time
        planning_times.append(planning_time)
        
        # 更新UAV状态
        uav_fleet.step(control_inputs, [disturbance] * len(uav_fleet.uavs))
        
        # 更新环境
        environment.update_dynamic_obstacles(0.5)
        
        # 更新AoI和访问记录
        num_grids = environment.workspace.M
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
    
    return {
        'aoi_history': aoi_history,
        'visit_history': visit_history,
        'uav_positions_history': uav_positions_history,
        'planning_times': planning_times
    }

def compute_metrics(results, coverage_metrics):
    """计算指标"""
    aoi_metrics = coverage_metrics.compute_aoi_metrics(
        results['aoi_history'],
        results['visit_history']
    )
    
    coverage_metrics_result = coverage_metrics.compute_coverage_metrics(
        results['visit_history'],
        results['uav_positions_history']
    )
    
    # 合并指标
    all_metrics = {}
    all_metrics.update(aoi_metrics)
    all_metrics.update(coverage_metrics_result)
    
    return all_metrics

def create_visualizations(our_results, baseline_results, environment):
    """创建可视化"""
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. AoI轨迹对比
    ax1 = axes[0, 0]
    our_aoi = np.array(our_results['aoi_history'])
    baseline_aoi = np.array(baseline_results['aoi_history'])
    
    our_mean_aoi = np.mean(our_aoi, axis=1)
    baseline_mean_aoi = np.mean(baseline_aoi, axis=1)
    
    time_steps = range(len(our_mean_aoi))
    ax1.plot(time_steps, our_mean_aoi, 'b-', linewidth=2, label='我们的方法')
    ax1.plot(time_steps, baseline_mean_aoi, 'r-', linewidth=2, label='基线方法')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('平均AoI')
    ax1.set_title('AoI轨迹对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 覆盖率对比
    ax2 = axes[0, 1]
    our_coverage = np.sum(our_results['visit_history'], axis=1) / our_results['visit_history'][0].shape[0]
    baseline_coverage = np.sum(baseline_results['visit_history'], axis=1) / baseline_results['visit_history'][0].shape[0]
    
    ax2.plot(time_steps, our_coverage, 'b-', linewidth=2, label='我们的方法')
    ax2.plot(time_steps, baseline_coverage, 'r-', linewidth=2, label='基线方法')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('覆盖率')
    ax2.set_title('覆盖率对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 环境可视化
    ax3 = axes[1, 0]
    environment.visualize(ax3)
    ax3.set_title('环境地图')
    
    # 4. UAV轨迹
    ax4 = axes[1, 1]
    our_positions = np.array(our_results['uav_positions_history'])
    
    # 绘制UAV轨迹
    for i in range(our_positions.shape[1]):  # 每个UAV
        trajectory = our_positions[:, i, :]
        ax4.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.7, linewidth=1)
    
    # 绘制最终位置
    final_positions = our_positions[-1]
    ax4.scatter(final_positions[:, 0], final_positions[:, 1], c='red', s=100, marker='*')
    
    ax4.set_xlim(environment.workspace.domain[0], environment.workspace.domain[1])
    ax4.set_ylim(environment.workspace.domain[2], environment.workspace.domain[3])
    ax4.set_aspect('equal')
    ax4.set_title('UAV轨迹')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    
    plt.tight_layout()
    plt.savefig('quick_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("可视化结果已保存为 'quick_test_results.png'")

def run_single_task_test():
    """运行单个任务测试"""
    print("=== 单个任务测试 ===")
    
    # 加载配置
    config_path = "config/experiment_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 选择任务T1进行测试
    task_name = "T1"
    task_config = config['tasks'][task_name]
    
    print(f"测试任务: {task_name}")
    print(f"任务描述: {task_config['name']}")
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 创建环境
    environment = create_environment_from_config(config)
    
    # 创建UAV机队
    uav_count = task_config['uav_count'][0]  # 使用第一个UAV数量
    uav_fleet = UAVFleet(config['uav'], environment)
    uav_fleet.create_uavs(uav_count)
    
    print(f"UAV数量: {uav_count}")
    print(f"环境网格数: {environment.workspace.M}")
    
    # 测试分区
    partition = PowerVoronoiPartition(environment, uav_fleet)
    uav_positions = uav_fleet.get_positions()
    site_weights = [uav.site_weight for uav in uav_fleet.uavs]
    
    print("测试Power-Voronoi分区...")
    cells = partition.compute_voronoi_cells(uav_positions, site_weights)
    print(f"分区单元数: {len(cells)}")
    
    # 计算量化泛函
    H_value = partition.compute_quantization_functional(uav_positions, site_weights)
    print(f"量化泛函值: {H_value:.2f}")
    
    # 测试Lloyd更新
    print("测试Lloyd更新...")
    new_positions, H_history = partition.update_partition(uav_positions, site_weights, max_iterations=10)
    print(f"更新后量化泛函值: {H_history[-1]:.2f}")
    print(f"量化泛函下降: {H_history[0] - H_history[-1]:.2f}")
    
    print("\n单个任务测试完成!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行实验测试')
    parser.add_argument('--mode', choices=['quick', 'single'], default='quick',
                       help='测试模式: quick(快速测试) 或 single(单个任务测试)')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_test()
    else:
        run_single_task_test()




