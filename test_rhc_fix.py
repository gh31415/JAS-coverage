#!/usr/bin/env python3
"""
测试RHC规划器的修复
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.models.uav_dynamics import UAVDynamics
from src.planner.rhc_planner import RHCOptimizer


def test_rhc_optimizer():
    """测试RHC优化器"""
    print("测试RHC优化器...")

    try:
        # 创建UAV动力学模型
        dynamics = UAVDynamics(dt=0.5, max_velocity=8.0, max_acceleration=4.0)
        print("✓ UAV动力学模型创建成功")

        # 创建简化的配置
        config = {
            'rhc': {
                'horizon': 10,
                'cost_weights': {
                    'aoi': 1.0,
                    'control': 0.01,
                    'collision': 5.0,
                    'boundary': 2.0
                },
                'terminal_set_epsilon': 0.1,
                'coupling_decay': 0.9
            },
            'aoi_tightening': {
                'delta': 0.05,
                'sigma': 0.1
            }
        }

        # 创建简化的环境和分区对象（模拟）
        class MockEnvironment:
            def __init__(self):
                self.workspace = MockWorkspace()

        class MockWorkspace:
            def __init__(self):
                self.grid_centers = np.random.uniform(0, 100, (50, 2))

        class MockPartition:
            def compute_voronoi_cells(self, positions, weights):
                return {0: list(range(10))}  # 简化的单元分配

        environment = MockEnvironment()
        partition = MockPartition()

        # 创建RHC优化器
        optimizer = RHCOptimizer(dynamics, environment, partition, config)
        print("✓ RHC优化器创建成功")

        # 测试终端控制器
        K = optimizer.terminal_controller
        print(f"终端控制器形状: {K.shape}")
        print(f"终端控制器: \n{K}")
        print("✓ 终端控制器构建成功")

        # 测试AoI收紧计算
        beta = optimizer.compute_aoi_tightening(100)
        print(f"AoI收紧参数: {beta}")
        print("✓ AoI收紧计算成功")

        return True

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("开始测试RHC规划器修复...")

    success = test_rhc_optimizer()

    if success:
        print("\n🎉 所有测试通过！RHC规划器修复成功。")
    else:
        print("\n❌ 测试失败，需要进一步检查。")




