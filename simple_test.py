#!/usr/bin/env python3
"""
简单测试脚本：验证基本功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import yaml

def test_basic_imports():
    """测试基本导入"""
    print("测试基本导入...")
    
    try:
        from src.models.uav_dynamics import UAVDynamics, UAVFleet
        from src.env.environment import create_environment_from_config
        from src.partition.power_voronoi import PowerVoronoiPartition
        from src.planner.rhc_planner import DistributedRHCPlanner
        print("✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入错误: {e}")
        return False

def test_uav_dynamics():
    """测试UAV动力学"""
    print("\n测试UAV动力学...")
    
    try:
        dynamics = UAVDynamics(dt=0.5, max_velocity=8.0, max_acceleration=4.0)
        print(f"✓ UAV动力学创建成功，A矩阵形状: {dynamics.A.shape}")
        return True
    except Exception as e:
        print(f"✗ UAV动力学错误: {e}")
        return False

def test_environment():
    """测试环境创建"""
    print("\n测试环境创建...")
    
    try:
        # 加载配置
        with open("config/experiment_config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建环境
        environment = create_environment_from_config(config)
        print(f"✓ 环境创建成功，网格数量: {environment.workspace.M}")
        return True
    except Exception as e:
        print(f"✗ 环境创建错误: {e}")
        return False

def test_uav_fleet():
    """测试UAV机队"""
    print("\n测试UAV机队...")
    
    try:
        # 加载配置
        with open("config/experiment_config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建环境
        environment = create_environment_from_config(config)
        
        # 创建UAV机队
        uav_fleet = UAVFleet(config['uav'], environment)
        uav_fleet.create_uavs(3)  # 创建3个UAV
        
        print(f"✓ UAV机队创建成功，UAV数量: {len(uav_fleet.uavs)}")
        return True
    except Exception as e:
        print(f"✗ UAV机队错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_partition():
    """测试分区算法"""
    print("\n测试分区算法...")
    
    try:
        # 加载配置
        with open("config/experiment_config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建环境
        environment = create_environment_from_config(config)
        
        # 创建UAV机队
        uav_fleet = UAVFleet(config['uav'], environment)
        uav_fleet.create_uavs(3)
        
        # 创建分区器
        partition = PowerVoronoiPartition(environment, uav_fleet)
        print("✓ 分区器创建成功")
        return True
    except Exception as e:
        print(f"✗ 分区器错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始简单测试...")
    
    tests = [
        test_basic_imports,
        test_uav_dynamics,
        test_environment,
        test_uav_fleet,
        test_partition
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有基本功能测试通过！")
    else:
        print("❌ 部分测试失败，需要进一步检查。")




