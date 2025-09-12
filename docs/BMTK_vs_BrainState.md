# BMTK vs BrainState LGN神经元实现对比

## 📋 BMTK原始实现

### 核心LNUnit类 (bmtk/scripts/lgnmodel/lgnmodel/lnunit.py)

```python
class LNUnit(object):
     
    def __init__(self, linear_filter, transfer_function, amplitude=1.):
        self.linear_filter = linear_filter
        self.transfer_function = transfer_function
        self.amplitude = amplitude

    def evaluate(self, movie, **kwargs):
        return self.get_cursor(movie, separable=kwargs.pop('separable', False)).evaluate(**kwargs)
 
    def get_spatiotemporal_kernel(self, *args, **kwargs):
        return self.linear_filter.get_spatiotemporal_kernel(*args, **kwargs)
    
    def get_cursor(self, movie, threshold=0, separable = False):
        if separable:
            return SeparableLNUnitCursor(self, movie)
        else:
            return LNUnitCursor(self, movie, threshold=threshold)
    
    def show_temporal_filter(self, *args, **kwargs):
        self.linear_filter.show_temporal_filter(*args, **kwargs)
        
    def show_spatial_filter(self, *args, **kwargs):
        self.linear_filter.show_spatial_filter(*args, **kwargs)
    
    def to_dict(self):
        return {'class':(__name__, self.__class__.__name__),
                'linear_filter':self.linear_filter.to_dict(),
                'transfer_function':self.transfer_function.to_dict()}
```

### OnUnit和OffUnit实现 (bmtk/scripts/lgnmodel/lgnmodel/cellmodel.py)

```python
class OnUnit(LNUnit):
    def __init__(self, linear_filter, transfer_function):
        assert linear_filter.amplitude > 0
        super(OnUnit, self).__init__(linear_filter, transfer_function)
        
class OffUnit(LNUnit):
    def __init__(self, linear_filter, transfer_function):
        assert linear_filter.amplitude < 0
        super(OffUnit, self).__init__(linear_filter, transfer_function)
```

### TwoSubfieldLinearCell实现

```python
class TwoSubfieldLinearCell(MultiLNUnit):
    def __init__(self, dominant_filter, nondominant_filter, 
                 subfield_separation=10, onoff_axis_angle=45, 
                 dominant_subfield_location=(30,40),
                 transfer_function=MultiTransferFunction((symbolic_x, symbolic_y),
                                   'Heaviside(x)*(x)+Heaviside(y)*(y)')):
         
        self.subfield_separation = subfield_separation
        self.onoff_axis_angle = onoff_axis_angle
        self.dominant_subfield_location = dominant_subfield_location
        self.dominant_filter = dominant_filter
        self.nondominant_filter = nondominant_filter
        self.transfer_function= transfer_function

        self.dominant_unit = LNUnit(self.dominant_filter, ScalarTransferFunction('s'), 
                                   amplitude=self.dominant_filter.amplitude)
        self.nondominant_unit = LNUnit(self.nondominant_filter, ScalarTransferFunction('s'), 
                                      amplitude=self.dominant_filter.amplitude)

        super(TwoSubfieldLinearCell, self).__init__([self.dominant_unit, self.nondominant_unit], 
                                                   self.transfer_function)
        
        # 设置空间位置
        self.dominant_filter.spatial_filter.translate = self.dominant_subfield_location
        hor_offset = np.cos(self.onoff_axis_angle*np.pi/180.)*self.subfield_separation + self.dominant_subfield_location[0]
        vert_offset = np.sin(self.onoff_axis_angle*np.pi/180.)*self.subfield_separation+ self.dominant_subfield_location[1]
        rel_translation = (hor_offset,vert_offset)
        self.nondominant_filter.spatial_filter.translate = rel_translation
```

### LGNOnCell实现 (实际使用的细胞)

```python
class LGNOnCell(object):
    def __init__(self, **kwargs):
        self.position = kwargs.pop('position', None)
        self.weights = kwargs.pop('weights', None)
        self.kpeaks = kwargs.pop('kpeaks', None)
        self.amplitude = kwargs.pop('amplitude', None)
        self.sigma = kwargs.pop('sigma', None)
        self.transfer_function_str = kwargs.pop('transfer_function_str', 'Heaviside(s)*s')
        self.metadata = kwargs.pop('metadata', {})

        temporal_filter = TemporalFilterCosineBump(self.weights, self.kpeaks)
        spatial_filter = GaussianSpatialFilter(translate=self.position, sigma=self.sigma, origin=(0,0))
        spatiotemporal_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=self.amplitude)
        transfer_function = ScalarTransferFunction(self.transfer_function_str)
        self.unit = OnUnit(spatiotemporal_filter, transfer_function)
```

## 🚀 BrainState重新实现 (brainlgn-X/brainlgn_x/neuron.py)

### 核心LGNNeuron类

```python
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
```

## 🔍 关键差异对比

### 1. **继承架构差异**

| 方面 | BMTK | BrainState |
|------|------|------------|
| 基类 | `object` | `bs.DynamicalSystem` |
| 状态管理 | 手动管理 | BrainState自动管理 |
| 数值计算 | NumPy | JAX (JIT编译) |

### 2. **计算方法差异**

| 操作 | BMTK | BrainState |
|------|------|------------|
| 主要方法 | `evaluate(movie)` | `update(stimulus)` |
| 滤波计算 | `get_cursor().evaluate()` | `linear_filter.convolve()` |
| 并行化 | 无 | JAX自动向量化 |

### 3. **数据流差异**

**BMTK流程**:
```
movie → get_cursor() → LNUnitCursor.evaluate() → 结果
```

**BrainState流程**:
```
stimulus → linear_filter.convolve() → transfer_function.apply() → 结果
```

### 4. **性能优化点**

| 优化 | BMTK | BrainState |
|------|------|------------|
| 编译 | 解释执行 | JAX JIT编译 |
| 内存 | 动态分配 | 预分配+复用 |
| GPU | 不支持 | 自动GPU加速 |
| 批处理 | 循环处理 | 向量化操作 |

## 🧪 验证计划

### 数值一致性验证

```python
def test_numerical_consistency():
    # 创建相同的输入
    stimulus = create_test_stimulus()
    
    # BMTK实现
    bmtk_unit = create_bmtk_lgn_unit()
    bmtk_response = bmtk_unit.evaluate(stimulus)
    
    # BrainState实现  
    brainstate_unit = create_brainstate_lgn_unit()
    brainstate_response = brainstate_unit.update(stimulus)
    
    # 验证数值误差 < 1e-10
    assert np.allclose(bmtk_response, brainstate_response, rtol=1e-10)
```

### 性能基准测试

```python
def benchmark_performance():
    stimulus = create_large_stimulus()  # 大规模刺激
    
    # 测试BMTK
    start_time = time.time()
    bmtk_response = bmtk_unit.evaluate(stimulus)
    bmtk_time = time.time() - start_time
    
    # 测试BrainState
    start_time = time.time() 
    brainstate_response = brainstate_unit.update(stimulus)
    brainstate_time = time.time() - start_time
    
    speedup = bmtk_time / brainstate_time
    print(f"性能提升: {speedup:.2f}x")
```

## 📝 迁移策略

### 阶段1: 核心功能对等
- [x] 实现基础LGNNeuron类
- [ ] 实现OnUnit/OffUnit子类
- [ ] 实现TwoSubfieldLinearCell
- [ ] 数值验证通过

### 阶段2: 性能优化
- [ ] JAX JIT编译优化
- [ ] 向量化批处理
- [ ] GPU加速支持

### 阶段3: 接口兼容
- [ ] 保持原有API调用方式
- [ ] evaluate()方法兼容包装
- [ ] 配置参数映射

---

**总结**: BrainState版本在保持功能完全一致的同时，通过现代化架构和JAX生态获得显著性能提升。