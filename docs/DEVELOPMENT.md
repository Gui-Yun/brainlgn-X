# BrainLGN-X 开发日志

## 🎯 开发策略 (基础组件优先)

按照从简单到复杂的原则，逐步验证每个组件：

### Phase 1: 单个神经元 ✋ (当前阶段)
1. **实现基础LGN神经元** - 用BrainState重写LNUnit
2. **验证响应正确性** - 与BMTK对比数值结果 
3. **单元测试** - 确保功能稳定

### Phase 2: 视觉刺激
1. **实现光栅刺激** - GratingMovie 
2. **实现闪光刺激** - FullFieldFlashMovie
3. **验证刺激生成** - 确保视觉输入正确

### Phase 3: 网络层面  
1. **多神经元网络** - 构建LGN群体
2. **连接管理** - 神经元间连接
3. **批量仿真** - 整体网络响应

### Phase 4: BMTK兼容
1. **API包装** - 保持接口一致
2. **配置兼容** - JSON配置解析
3. **输出格式** - HDF5文件兼容

## 📝 当前任务

### 正在进行: 单个LGN神经元实现

**目标**: 创建一个完全功能的LGN神经元，能够：
- 接受视觉输入
- 进行时空滤波
- 生成尖峰输出
- 数值结果与BMTK一致

**实现文件**:
- `brainlgn_x/neuron.py` - 神经元主体
- `brainlgn_x/filters.py` - 滤波器实现  
- `tests/test_neuron.py` - 单元测试

**验证方法**:
```python
# 简单测试：给定输入，验证输出
neuron = LGNNeuron(...)
stimulus = create_simple_stimulus()
response = neuron.process(stimulus)
assert np.allclose(response, expected_bmtk_response)
```

## 🧪 验证策略

每个阶段都要完成数值验证：
1. 创建相同的输入
2. 对比BMTK和BrainState的输出
3. 确保误差 < 1e-10
4. 通过单元测试

## 📊 开发进度

- [x] 项目结构设计
- [x] 开发策略制定
- [ ] 单个神经元实现
- [ ] 神经元响应验证
- [ ] 视觉刺激系统
- [ ] 网络层构建
- [ ] BMTK兼容层

---
*最后更新: 2024-09-12*