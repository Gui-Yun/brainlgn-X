"""
BrainState版本的滤波器实现

基于BMTK lgnmodel的滤波器，使用JAX重写以获得性能提升
"""

import numpy as np
import jax.numpy as jnp
import brainstate as bs


class GaussianSpatialFilter:
    """高斯空间滤波器 - 对应BMTK的GaussianSpatialFilter"""
    
    def __init__(self, size=(20, 20), sigma=3.0, amplitude=1.0, translate=(0, 0)):
        self.size = size
        self.sigma = sigma  
        self.amplitude = amplitude
        self.translate = translate
        
        # 预计算高斯核
        self.kernel = self._create_gaussian_kernel()
    
    def _create_gaussian_kernel(self):
        """创建高斯核"""
        height, width = self.size
        center_y, center_x = height // 2, width // 2
        
        y, x = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing='ij')
        
        # 高斯公式
        dist_sq = (x - center_x - self.translate[0])**2 + (y - center_y - self.translate[1])**2
        kernel = self.amplitude * jnp.exp(-dist_sq / (2 * self.sigma**2))
        
        return kernel
    
    def convolve(self, image):
        """空间卷积"""
        # 使用JAX的卷积操作
        return jnp.convolve(image, self.kernel, mode='same')


class TemporalFilterCosineBump:
    """余弦凸起时间滤波器 - 对应BMTK的TemporalFilterCosineBump"""
    
    def __init__(self, duration=0.5, phase=0.0, amplitude=1.0, frame_rate=1000.0):
        self.duration = duration
        self.phase = phase
        self.amplitude = amplitude
        self.frame_rate = frame_rate
        
        # 创建时间核
        self.kernel = self._create_temporal_kernel()
    
    def _create_temporal_kernel(self):
        """创建时间滤波核"""
        dt = 1.0 / self.frame_rate
        t = jnp.arange(0, self.duration, dt)
        
        # 余弦凸起滤波器公式
        kernel = self.amplitude * jnp.cos(2 * jnp.pi * t / self.duration + self.phase)
        # 只保留正值部分（凸起）
        kernel = jnp.maximum(kernel, 0)
        
        return kernel
    
    def convolve(self, signal):
        """时间卷积"""
        return jnp.convolve(signal, self.kernel, mode='same')


class SpatioTemporalFilter:
    """时空滤波器 - 对应BMTK的SpatioTemporalFilter"""
    
    def __init__(self, spatial_filter, temporal_filter, amplitude=1.0):
        self.spatial_filter = spatial_filter
        self.temporal_filter = temporal_filter
        self.amplitude = amplitude
    
    def convolve(self, stimulus):
        """
        时空卷积
        
        Args:
            stimulus: 输入刺激 (time x height x width)
            
        Returns:
            response: 滤波后的响应
        """
        # 先进行空间滤波
        spatial_response = []
        for t in range(stimulus.shape[0]):
            frame = stimulus[t]
            spatial_filtered = self.spatial_filter.convolve(frame)
            # 对空间滤波结果求和得到该时刻的响应值
            spatial_response.append(jnp.sum(spatial_filtered))
        
        spatial_response = jnp.array(spatial_response)
        
        # 再进行时间滤波
        if self.temporal_filter is not None:
            temporal_response = self.temporal_filter.convolve(spatial_response)
        else:
            temporal_response = spatial_response
            
        return temporal_response * self.amplitude


class LinearTransferFunction:
    """线性传递函数 - 对应BMTK的ScalarTransferFunction('s')"""
    
    def __init__(self, slope=1.0, offset=0.0):
        self.slope = slope
        self.offset = offset
    
    def apply(self, x):
        """应用线性传递函数"""
        return self.slope * x + self.offset


class RectifyingTransferFunction:
    """整流传递函数 - 对应BMTK的'Heaviside(s)*s'"""
    
    def apply(self, x):
        """应用整流传递函数 - 只保留正值"""
        return jnp.maximum(x, 0)