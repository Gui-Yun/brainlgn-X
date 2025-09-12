"""
BrainState版本的LGN神经元实现

基于原始BMTK lgnmodel的LNUnit，使用BrainState重写

@
"""

import numpy as np
import brainstate as bs
import jax.numpy as jnp
from .filters import GaussianSpatialFilter, TemporalFilterCosineBump, SpatioTemporalFilter


class LGNNeuron(bs.DynamicalSystem):
    """
    LGN神经元 - BrainState版本
    
    对应BMTK中的LNUnit类，实现线性-非线性(LN)神经元模型
    """
    
    def __init__(self, spatial_filter, temporal_filter, transfer_function, amplitude=1.0):
        super().__init__()
        self.spatial_filter = spatial_filter
        self.temporal_filter = temporal_filter  
        self.transfer_function = transfer_function
        self.amplitude = amplitude
        
        # 创建时空滤波器
        self.linear_filter = SpatioTemporalFilter(
            spatial_filter=spatial_filter,
            temporal_filter=temporal_filter,
            amplitude=amplitude
        )
    
    def update(self, stimulus):
        """
        处理视觉刺激，返回神经元响应
        
        Args:
            stimulus: 输入刺激 (time x height x width)
            
        Returns:
            response: 神经元响应时间序列
        """
        # 线性滤波
        linear_response = self.linear_filter.convolve(stimulus)
        
        # 非线性传递函数
        nonlinear_response = self.transfer_function.apply(linear_response)
        
        return nonlinear_response


class OnUnit(LGNNeuron):
    """ON型LGN神经元 - 对光增强敏感"""
    
    def __init__(self, spatial_filter, temporal_filter, transfer_function):
        # ON单元要求正振幅
        assert spatial_filter.amplitude > 0, "ON unit requires positive amplitude"
        super().__init__(spatial_filter, temporal_filter, transfer_function)


class OffUnit(LGNNeuron):  
    """OFF型LGN神经元 - 对光减弱敏感"""
    
    def __init__(self, spatial_filter, temporal_filter, transfer_function):
        # OFF单元要求负振幅
        assert spatial_filter.amplitude < 0, "OFF unit requires negative amplitude"
        super().__init__(spatial_filter, temporal_filter, transfer_function)


class TwoSubfieldLinearCell(bs.DynamicalSystem):
    """
    双感受野线性细胞
    
    包含主导和非主导两个子感受野
    """
    
    def __init__(self, dominant_filter, nondominant_filter, 
                 subfield_separation=10, onoff_axis_angle=45, 
                 dominant_subfield_location=(30, 40)):
        super().__init__()
        self.dominant_filter = dominant_filter
        self.nondominant_filter = nondominant_filter
        self.subfield_separation = subfield_separation
        self.onoff_axis_angle = onoff_axis_angle
        self.dominant_subfield_location = dominant_subfield_location
        
        # 创建两个LGN单元
        self.dominant_unit = LGNNeuron(dominant_filter, None, None)
        self.nondominant_unit = LGNNeuron(nondominant_filter, None, None)
    
    def update(self, stimulus):
        """处理双子感受野响应"""
        dominant_response = self.dominant_unit.update(stimulus)
        nondominant_response = self.nondominant_unit.update(stimulus) 
        
        # 线性组合两个子感受野的响应
        combined_response = dominant_response + nondominant_response
        return combined_response


def create_simple_on_unit():
    """
    创建一个简单的ON单元用于测试
    
    Returns:
        OnUnit: 配置好的ON型LGN神经元
    """
    # 空间滤波器 - 高斯型
    spatial_filter = GaussianSpatialFilter(
        size=(20, 20),
        sigma=3.0,
        amplitude=1.0
    )
    
    # 时间滤波器 - 余弦波形
    temporal_filter = TemporalFilterCosineBump(
        duration=0.5,
        phase=0.0,
        amplitude=1.0
    )
    
    # 传递函数 - 简单线性
    from .transfer import LinearTransferFunction
    transfer_function = LinearTransferFunction()
    
    return OnUnit(spatial_filter, temporal_filter, transfer_function)