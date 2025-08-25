# 多无人机分布式覆盖优化实验代码

本项目实现了论文"Distributed Power Voronoi-Based Spatiotemporal Coverage Optimization with Temporal Feasibility and Robust Receding Horizon Control for Multi-UAV Systems"的完整实验代码。

## 项目结构

```
code/
├── config/
│   └── experiment_config.yaml    # 实验配置文件
├── src/
│   ├── env/
│   │   └── environment.py        # 环境模块（工作域、网格、障碍）
│   ├── models/
│   │   └── uav_dynamics.py       # UAV动力学模型和干扰生成
│   ├── partition/
│   │   └── power_voronoi.py      # Power-Voronoi分区算法
│   ├── planner/
│   │   └── rhc_planner.py        # 分布式RHC规划器
│   ├── baselines/
│   │   └── baseline_methods.py   # 基线方法实现
│   └── eval/
│       └── metrics.py            # 评估指标和可视化工具
├── src/
│   └── main_experiment.py        # 主实验脚本
├── requirements.txt              # 依赖包列表
└── README.md                     # 项目说明
```

## 核心功能

### 1. 环境建模
- **工作域**: 1000×1000m区域，30×30规则网格
- **环境密度**: 均匀分布、聚类分布、走廊分布
- **障碍**: 静态圆形/矩形障碍，动态移动障碍
- **AoI阈值**: 基础区60s，热点区30s，走廊45s

### 2. UAV系统
- **动力学**: 离散双积分模型
- **异构性**: 不同传感半径(12-18m)、站点权重(0-2)
- **通信**: 180m通信半径，10%丢包率，1-2步延迟
- **约束**: 最大速度8m/s，最大加速度4m/s²

### 3. 核心算法
- **Power-Voronoi分区**: 加权Voronoi + Lloyd更新
- **分布式RHC**: 滚动时域控制 + AoI约束
- **鲁棒收紧**: 亚高斯干扰下的概率保证
- **终端控制**: LQR终端控制器 + 终端集约束

### 4. 基线方法
- Unweighted Voronoi + Lloyd
- Power-Voronoi (仅环境密度)
- CBBA/拍卖分配
- 前沿探索
- 轮询巡检
- 贪心AoI最小化
- 集中式MPC

### 5. 评估指标
- **时效性**: 平均/最大AoI、Deadline满足率、重访间隔
- **空间效率**: 覆盖率、热点优先覆盖率、量化泛函收敛
- **鲁棒性**: ISS稳定性、AoI概率保证、碰撞统计
- **性能**: 计算时间、通信负载、能耗

## 实验任务

| 任务 | 描述 | UAV数量 | 干扰类型 | 障碍类型 |
|------|------|---------|----------|----------|
| T1 | 均匀分布+静态障碍 | 6/10 | 无/有界 | 静态 |
| T2 | 聚类分布+热点 | 10/15 | 有界 | 静态 |
| T3 | 走廊分布+狭通道 | 10 | 有界 | 静态+窄隘 |
| T4 | 动态障碍 | 10/15 | 有界/亚高斯 | 动态 |
| T5 | 通信不稳定 | 10 | 有界 | 静态 |
| T6 | 重负载(小τ_k) | 15 | 有界 | 静态 |

## 安装和运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行实验
```bash
cd src
python main_experiment.py
```

### 3. 测试可视化功能
```bash
python test_visualization.py
```
此脚本会生成示例数据并测试所有可视化功能，输出图表保存到 `test_visualizations/` 目录。

### 4. 查看结果
实验结果将保存在`results/`目录下：
- `experiment_results_YYYYMMDD_HHMMSS.npz`: 原始数据
- `summary_YYYYMMDD_HHMMSS.csv`: 汇总指标
- `task_*/`: 各任务的可视化图表

## 配置说明

主要配置参数在`config/experiment_config.yaml`中：

```yaml
# 实验参数
experiment:
  duration: 1200        # 仿真时长(步数)
  random_seeds: [42, 123, ...]  # 随机种子

# UAV参数
uav:
  sampling_period: 0.5  # 采样周期
  max_velocity: 8.0     # 最大速度
  max_acceleration: 4.0 # 最大加速度

# RHC参数
rhc:
  horizon: 20           # 预测步长
  cost_weights:         # 代价权重
    aoi: 1.0
    control: 0.01
    collision: 5.0
```

## 核心算法实现

### Power-Voronoi分区
```python
# 加权距离计算
d_w(q, p_i) = ||q - p_i||^2 - w_i

# Lloyd更新
p_i^+ = C_i(P) = ∫_{V_i^w(P)} ρ(q) q dq / ∫_{V_i^w(P)} ρ(q) dq
```

### 分布式RHC
```python
# 优化问题
min_{u_i(0:H-1)} Σ_{h=0}^{H-1} Σ_{k=1}^M w_k ℓ(A_k(t+h))

# 约束条件
x_{i,t+1} = A x_{i,t} + B u_{i,t} + d_t
A_k(t+h) ≤ τ_k (收紧约束)
x_i(H) ∈ X_{f,i} (终端集约束)
```

### AoI收紧
```python
# 概率保证
β_k = σ_k √(2 log(T/δ))
Pr{A_k(t) ≤ τ_k, ∀t ∈ [0,T]} ≥ 1 - δ
```

## 理论验证

### Theorem 1: AoI上界
- 验证重访间隔 ≤ τ_k 时，AoI ≤ τ_k
- 实现: `CoverageMetrics.compute_aoi_metrics()`

### Theorem 2: 量化泛函收敛
- 验证H(P)单调下降，收敛到加权质心配置
- 实现: `PowerVoronoiPartition.update_partition()`

### Theorem 3: ISS稳定性
- 验证组合Lyapunov函数的差分负定
- 实现: `RobustnessMetrics.compute_iss_metrics()`

### Theorem 4: AoI概率保证
- 验证亚高斯干扰下的概率约束
- 实现: `RobustnessMetrics.compute_aoi_probability_guarantee()`

### Theorem 5: RHC性能界
- 验证递归可行性和性能上界
- 实现: `DistributedRHCPlanner.plan_step()`

## 结果分析

### 统计检验
- Wilcoxon符号秩检验: 成对方法比较
- Friedman检验: 多方法比较
- 置信区间: Bootstrap方法

### 可视化
系统会为每个任务和UAV数量生成以下对比图表：

#### 时间序列对比图（均值+阴影）
- **AoI对比曲线**：平均AoI和最大AoI的时间演化
- **Deadline违反率对比**：时间窗口内的违反率变化
- **覆盖率对比曲线**：环境覆盖率随时间的变化
- **能耗对比曲线**：总能耗消耗的时间序列
- **量化函数对比**：Power-Voronoi量化函数H(P)的收敛性

#### 性能分布图
- **计算时间对比**：箱线图显示规划时间分布，时间序列显示计算负载变化
- **综合性能总结**：6个子图展示所有关键指标的对比
- **方法排名图**：基于综合得分的水平柱状图排名

#### 图表特点
- **多实验统计**：基于多个随机种子的实验结果
- **置信区间**：95%置信区间的阴影区域
- **颜色编码**：每种方法使用独特的颜色和线型
- **高分辨率**：300 DPI输出，适合论文发表

## 扩展功能

### 消融实验
- A1 (–site): 移除站点权重
- A2 (–ρ): 移除环境密度
- A3 (–AoI-tight): 移除AoI收紧
- A4 (–Ψ): 移除耦合惩罚
- A5 (短H): 缩短预测步长
- A6 (–X_f/–κ): 移除终端集/控制器

### 自定义实验
可以通过修改配置文件添加新的：
- 环境密度分布
- 干扰类型
- 基线方法
- 评估指标

## 注意事项

1. **计算资源**: 完整实验需要较长时间，建议使用多核处理器
2. **内存使用**: 大规模实验可能需要大量内存存储历史数据
3. **数值稳定性**: RHC求解可能遇到数值问题，已实现备选控制策略
4. **随机性**: 所有实验使用固定随机种子确保可重现性

## 引用

如果使用本代码，请引用相关论文：

```bibtex
@article{gao2025distributed,
  title={Distributed Power Voronoi-Based Spatiotemporal Coverage Optimization with Temporal Feasibility and Robust Receding Horizon Control for Multi-UAV Systems},
  author={Gao, Hao and Zhou, Siyi and Gao, Yun and Cheng, Xianzhe},
  journal={IEEE Journal on Selected Areas in Communications},
  year={2025}
}
```

## 联系方式

如有问题或建议，请联系：
- Hao Gao: ghalfred39@gmail.com
- Xianzhe Cheng: chengxianzhe11@nudt.edu.cn
