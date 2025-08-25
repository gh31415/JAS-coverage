#!/usr/bin/env python3
"""
测试可视化功能的脚本
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from src.eval.metrics import VisualizationTools

def generate_sample_data():
    """生成示例数据用于测试可视化"""
    # 模拟多个随机种子和方法的实验结果
    sample_results = {}
    
    methods = ['Ours_Full', 'Ours_Light', 'Unweighted_Voronoi', 'Power_Voronoi', 'CBBA']
    seeds = [1, 2, 3, 4, 5]
    
    for seed in seeds:
        sample_results[seed] = {}
        
        for method in methods:
            # 生成模拟的实验数据
            duration = 100
            num_grids = 50
            
            # 生成AoI历史
            aoi_history = []
            for t in range(duration):
                # 模拟不同方法的AoI表现
                if 'Ours' in method:
                    base_aoi = 15 + np.random.normal(0, 2)
                elif 'Voronoi' in method:
                    base_aoi = 25 + np.random.normal(0, 3)
                else:
                    base_aoi = 35 + np.random.normal(0, 4)
                
                aoi_values = np.maximum(0, base_aoi + np.random.normal(0, 5, num_grids))
                aoi_history.append(aoi_values)
            
            # 生成访问历史
            visit_history = []
            for t in range(duration):
                visits = np.random.choice([0, 1], size=num_grids, p=[0.7, 0.3])
                visit_history.append(visits)
            
            # 生成规划时间
            if 'Ours' in method:
                planning_times = np.random.normal(0.5, 0.1, duration)
            else:
                planning_times = np.random.normal(0.2, 0.05, duration)
            
            # 生成控制输入历史
            control_inputs_history = []
            for t in range(duration):
                inputs = np.random.normal(0, 1, (3, 2))  # 3个UAV，每个2维控制输入
                control_inputs_history.append(inputs)
            
            # 生成UAV位置历史
            uav_positions_history = []
            for t in range(duration):
                positions = np.random.uniform(0, 100, (3, 2))  # 3个UAV，每个2维位置
                uav_positions_history.append(positions)
            
            # 生成模拟指标
            metrics = {
                'average_aoi': np.mean([np.mean(aoi) for aoi in aoi_history]),
                'max_aoi': np.max([np.max(aoi) for aoi in aoi_history]),
                'deadline_satisfaction_rate': np.random.uniform(0.8, 0.95),
                'coverage_rate': np.random.uniform(0.6, 0.9),
                'mean_planning_time': np.mean(planning_times),
                'total_energy_consumption': np.random.uniform(100, 500),
                'quantization_functional': np.random.uniform(10, 50)
            }
            
            sample_results[seed][method] = {
                'metrics': metrics,
                'data': {
                    'aoi_history': aoi_history,
                    'visit_history': visit_history,
                    'planning_times': planning_times,
                    'control_inputs_history': control_inputs_history,
                    'uav_positions_history': uav_positions_history
                }
            }
    
    return sample_results

def test_visualizations():
    """测试所有可视化功能"""
    print("生成示例数据...")
    sample_data = generate_sample_data()
    
    print("创建可视化工具...")
    viz_tools = VisualizationTools()
    
    # 创建输出目录
    output_dir = "test_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成可视化图表...")
    
    # 测试各种可视化功能
    task_name = "T1_Static_Environment"
    uav_count = 3
    
    # 1. AoI对比曲线
    print("  生成AoI对比曲线...")
    viz_tools.plot_aoi_comparison_curves(sample_data, task_name, uav_count, output_dir)
    
    # 2. Deadline违反率对比
    print("  生成Deadline违反率对比...")
    viz_tools.plot_deadline_violation_comparison(sample_data, task_name, uav_count, output_dir)
    
    # 3. 计算时间对比
    print("  生成计算时间对比...")
    viz_tools.plot_computation_time_comparison(sample_data, task_name, uav_count, output_dir)
    
    # 4. 覆盖率对比
    print("  生成覆盖率对比...")
    viz_tools.plot_coverage_comparison(sample_data, task_name, uav_count, output_dir)
    
    # 5. 能耗对比
    print("  生成能耗对比...")
    viz_tools.plot_energy_consumption_comparison(sample_data, task_name, uav_count, output_dir)
    
    # 6. 量化函数对比
    print("  生成量化函数对比...")
    viz_tools.plot_quantization_functional_comparison(sample_data, task_name, uav_count, output_dir)
    
    # 7. 综合性能总结
    print("  生成综合性能总结...")
    viz_tools.plot_comprehensive_summary(sample_data, task_name, uav_count, output_dir)
    
    # 8. 方法排名
    print("  生成方法排名...")
    viz_tools.plot_method_ranking(sample_data, task_name, uav_count, output_dir)
    
    print(f"\n所有可视化图表已保存到 {output_dir} 目录")
    print("生成的文件包括：")
    
    files = os.listdir(output_dir)
    for file in sorted(files):
        print(f"  - {file}")

if __name__ == "__main__":
    test_visualizations()




