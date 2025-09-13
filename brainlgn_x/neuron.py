"""
BrainState 版本的 LGN 神经元（计算逻辑对齐 BMTK LNUnit）。

@author: gray
@date: 2025-09-12
@e-mail: oswin0001@qq.com
@Liu-Lab

约定：
- 输入刺激形状为 (t, y, x)，t 单位为秒采样（由 frame_rate 决定）。
- 输出为放电率 rate[t]（Hz），非负由传递函数保证（例如 Heaviside/ReLU）。
- 仅实现可分离（separable=True）管线：空间卷积 -> 时间卷积 -> 传递函数。
"""

import os
import numpy as np
# Optional brainstate dependency:
# Some environments (e.g., Python < 3.10) may fail when importing brainstate/brainevent
# due to type-annotation features. To keep tests runnable and parity-focused, fall back
# to a minimal stub if any exception occurs while importing brainstate.
try:
    import brainstate as bs
    if not hasattr(bs, 'DynamicalSystem'):
        class _Dyn(object):
            pass
        bs.DynamicalSystem = _Dyn
except Exception:
    class _BS:
        class DynamicalSystem(object):
            pass
    bs = _BS()

from .filters import GaussianSpatialFilter, TemporalFilterCosineBump, SpatioTemporalFilter


class LGNNeuron(bs.DynamicalSystem):
    """
    LGN 神经元（对应 BMTK 的 LNUnit）。
    实现线性-非线性（LN）神经元模型的最小计算管线。
    """

    def __init__(self, spatial_filter, temporal_filter, transfer_function, amplitude=1.0):
        super().__init__()
        self.spatial_filter = spatial_filter
        self.temporal_filter = temporal_filter
        self.transfer_function = transfer_function
        self.amplitude = float(amplitude)

        # 创建时空滤波器（线性部分的幅值由 amplitude 定义，决定 ON/OFF）
        self.linear_filter = SpatioTemporalFilter(
            spatial_filter=spatial_filter,
            temporal_filter=temporal_filter,
            amplitude=self.amplitude,
        )

    def evaluate(self, stimulus, separable=True, downsample=1, threshold=None, frame_rate=1000.0, backend=None):
        """
        计算神经元对刺激的响应（放电率时间序列）。

        Args:
            stimulus: 输入刺激 (t, y, x)
            separable (bool): 仅支持 True（空间->时间）
            downsample (int): 输出下采样步长（>=1）
            threshold (float|None): 线性响应的偏置/阈值（可选），等价于 s+b
        Returns:
            rate: (t,) 非负放电率（Hz）
        """
        # 后端选择：参数优先，其次环境变量，默认 bmtk
        _backend = backend or os.getenv('BRAINLGN_BACKEND', 'bmtk').lower()

        if _backend in ('brainstate', 'jax', 'bs'):
            from . import bs_backend as bsx
            if not separable:
                return bsx.eval_nonseparable(self.linear_filter, self.transfer_function,
                                             np.array(stimulus, copy=False), float(frame_rate),
                                             downsample=int(downsample))
            return bsx.eval_separable(self.linear_filter, self.transfer_function,
                                      np.array(stimulus, copy=False), float(frame_rate),
                                      downsample=int(downsample))

        # 默认：BMTK 参考实现（Parity）
        from bmtk.simulator.filternet.lgnmodel.lnunit import LNUnit
        from bmtk.simulator.filternet.lgnmodel.movie import Movie

        movie = Movie(np.array(stimulus, copy=False), frame_rate=float(frame_rate))
        ln = LNUnit(self.linear_filter, self.transfer_function)

        if separable:
            t_vals, y_vals = ln.get_cursor(movie, separable=True).evaluate()
            rate = np.array(y_vals)
            if downsample and downsample > 1:
                rate = rate[::int(downsample)]
            return rate

        t_vals, rate = ln.get_cursor(movie, separable=False, threshold=(threshold or 0.0)).evaluate(
            downsample=int(downsample)
        )
        return rate

    # 兼容 update 命名（调用 evaluate）
    def update(self, stimulus):
        return self.evaluate(stimulus)


class OnUnit(LGNNeuron):
    """ON 型 LGN 单元：对亮度增强敏感（线性部分幅值 > 0）。"""

    def __init__(self, spatial_filter, temporal_filter, transfer_function, amplitude=1.0):
        assert amplitude > 0, "ON unit requires positive linear amplitude"
        super().__init__(spatial_filter, temporal_filter, transfer_function, amplitude=amplitude)


class OffUnit(LGNNeuron):
    """OFF 型 LGN 单元：对亮度减弱敏感（线性部分幅值 < 0）。"""

    def __init__(self, spatial_filter, temporal_filter, transfer_function, amplitude=-1.0):
        assert amplitude < 0, "OFF unit requires negative linear amplitude"
        super().__init__(spatial_filter, temporal_filter, transfer_function, amplitude=amplitude)


class TwoSubfieldLinearCell(bs.DynamicalSystem):
    """
    双感受野线性细胞（主/次子场）。

    约定：两个子场均为完整 LN 单元（含空间/时间/幅值/传递函数）。
    MVP：各自通过非线性后相加（等价于 Heaviside(x)*x + Heaviside(y)*y）。
    """

    def __init__(self, dominant_unit: LGNNeuron, nondominant_unit: LGNNeuron,
                 subfield_separation=10.0, onoff_axis_angle=45.0,
                 dominant_subfield_location=(30.0, 40.0)):
        super().__init__()
        self.dominant_unit = dominant_unit
        self.nondominant_unit = nondominant_unit
        self.subfield_separation = float(subfield_separation)
        self.onoff_axis_angle = float(onoff_axis_angle)
        self.dominant_subfield_location = tuple(dominant_subfield_location)

        # 几何参数由各自的空间滤波器管理（如 translate）；此处不强制修改，留给构造处完成。

    def evaluate(self, stimulus, separable=True, downsample=1):
        dom = self.dominant_unit.evaluate(stimulus, separable=separable, downsample=downsample)
        ndom = self.nondominant_unit.evaluate(stimulus, separable=separable, downsample=downsample)
        return dom + ndom

    def update(self, stimulus):
        return self.evaluate(stimulus)


def create_simple_on_unit():
    """
    创建一个简单的 ON 单元用于测试（占位接口）。

    Returns:
        OnUnit: 配置好的 ON 型 LGN 神经元
    """
    # 空间滤波器：高斯
    spatial_filter = GaussianSpatialFilter(
        size=(20, 20),
        sigma=3.0,
        amplitude=1.0,
    )

    # 时间滤波器：余弦 bump（占位参数接口）
    temporal_filter = TemporalFilterCosineBump(
        duration=0.5,
        phase=0.0,
        amplitude=1.0,
    )

    # 传递函数：简单线性/Heaviside（占位）
    from .transfer import LinearTransferFunction
    transfer_function = LinearTransferFunction()

    return OnUnit(spatial_filter, temporal_filter, transfer_function, amplitude=1.0)
