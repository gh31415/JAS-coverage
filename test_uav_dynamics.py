#!/usr/bin/env python3
"""
测试UAV动力学模型的修复
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.uav_dynamics import UAVDynamics, DisturbanceGenerator


def test_uav_dynamics():
    """测试UAV动力学模型"""
    print("测试UAV动力学模型...")

    # 创建UAV动力学模型
    dt = 0.5
    max_velocity = 8.0
    max_acceleration = 4.0

    try:
        dynamics = UAVDynamics(dt, max_velocity, max_acceleration)
        print("✓ UAV动力学模型创建成功")

        # 测试状态转移矩阵
        print(f"A矩阵形状: {dynamics.A.shape}")
        print(f"B矩阵形状: {dynamics.B.shape}")
        print("✓ 离散化矩阵构建成功")

        # 测试状态更新
        x = np.array([0.0, 0.0, 1.0, 1.0])  # 初始状态
        u = np.array([0.5, 0.3])  # 控制输入

        x_next = dynamics.step(x, u)
        print(f"初始状态: {x}")
        print(f"控制输入: {u}")
        print(f"下一状态: {x_next}")
        print("✓ 状态更新成功")

        # 测试约束
        constraints = dynamics.get_control_constraints()
        print(f"控制约束: {constraints}")
        print("✓ 约束获取成功")

        return True

    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def test_disturbance_generator():
    """测试干扰生成器"""
    print("\n测试干扰生成器...")

    try:
        # 测试无干扰
        params = {}
        generator = DisturbanceGenerator("none", params)
        d = generator.generate(0)
        print(f"无干扰: {d}")

        # 测试有界干扰
        params = {
            'max_magnitude': 1.0,
            'direction_drift': 0.1
        }
        generator = DisturbanceGenerator("bounded", params)
        d = generator.generate(0)
        print(f"有界干扰: {d}")

        # 测试亚高斯干扰
        params = {
            'sigma': 0.5,
            'mean': 0.0
        }
        generator = DisturbanceGenerator("sub_gaussian", params)
        d = generator.generate(0)
        print(f"亚高斯干扰: {d}")

        print("✓ 干扰生成器测试成功")
        return True

    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


if __name__ == "__main__":
    import numpy as np

    print("开始测试UAV动力学模型修复...")

    success1 = test_uav_dynamics()
    success2 = test_disturbance_generator()

    if success1 and success2:
        print("\n🎉 所有测试通过！UAV动力学模型修复成功。")
    else:
        print("\n❌ 部分测试失败，需要进一步检查。")




