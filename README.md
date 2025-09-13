BrainLGN-X
==========

一个基于 BrainState/BrainPy 的 LGN（外侧膝状体）建模项目，目标与 BMTK FilterNet 数值/接口保持一致（Parity）。

项目概述
- 使用 BrainState/BrainPy 逐步重构传统的 LN（线性→非线性）LGN 流水线。
- 通过“透传”方式直接复用 BMTK 的 LGN 滤波与传递函数，确保结果与 BMTK 完全一致。
- 最小可用路径（MVP）：漂移光栅刺激 → 可分离 LN（空间→时间）→ 放电率（Hz）→ 泊松采样生成尖峰。

当前亮点
- 与 BMTK 完全对齐：空间/时间/时空滤波与传递函数均直接复用 BMTK 实现。
- 简洁接口：`evaluate(stimulus, separable=True, downsample=1, threshold=None, frame_rate=...)`。
- 已提供数值一致性测试（可分离/不可分离），绝对误差目标 1e-12。

仓库结构
- `brainlgn_x/`
  - `neuron.py`：LGN 神经元（等价 LNUnit）的接口与计算管线。
  - `filters.py`：BMTK 滤波器透传（GaussianSpatialFilter、TemporalFilterCosineBump、SpatioTemporalFilter 等）。
  - `transfer.py`：BMTK 传递函数透传（ScalarTransferFunction、MultiTransferFunction）。
- `tests/`
  - `test_bmtk_parity.py`：与 BMTK 的数值对齐（可分离）
  - `test_bmtk_parity_more.py`：更多对齐场景（不可分离、下采样、OFF 单元、偏置、双子场求和）
- `docs/`
  - `DEVELOPMENT.md`：英文开发日志与变更记录
- `notebooks/`
  - `visualize_parity.ipynb`：可视化 BMTK 与本实现的差异（叠加、残差、散点、直方图）

依赖环境
- Python 3.9+
- `brainstate`、`brainpy`、`jax/jaxlib`、`numpy`、`scipy`、`pytest`
- `bmtk`（用于透传实现，保证与 BMTK 完全一致）

安装示例（conda）
1) 创建与激活环境：
- `conda create -n brainlgn python=3.11 -y`
- `conda activate brainlgn`
2) 安装依赖：
- `pip install brainstate brainpy "jax[cpu]" numpy scipy pytest bmtk`

快速校验
- 运行测试：`pytest -q`
- 对齐测试会在相同滤波堆栈与刺激下比较：
  - 参考：BMTK `LNUnit + Cursor`（可分离或不可分离）
  - 本实现：`LGNNeuron.evaluate(...)`
  - 目标：绝对误差 ≤ 1e-12（测试中已覆盖多种场景）

使用要点
- 刺激张量形状：`(t, y, x)`，时间步长由 `frame_rate`（单位秒）定义。
- 输出为放电率（Hz），非负性由传递函数保证（推荐用 `Max(0, s)` 或 `Max(0, s+b)`）。
- ON/OFF 通过 `SpatioTemporalFilter.amplitude` 的正负确定（>0 为 ON，<0 为 OFF）。
- BMTK 时间核默认 1 ms 采样（`nkt=600`），请在调用侧对齐 `frame_rate/dt`。
- 空间坐标：`translate=(x, y)`，与帧索引 `[t, y, x]` 一致（与 BMTK 处理相同）。

路线图
- MVP：可分离 LN 对齐、泊松尖峰、最小 HDF5/CSV 输出。
- 下一步：刺激生成器（光栅/闪烁）、多神经元网络、JSON 配置解析、SONATA 兼容输出。

说明
- 更详细的进展与变更请见 `docs/DEVELOPMENT.md`（英文）。
