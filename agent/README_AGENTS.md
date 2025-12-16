# Agent 使用说明

本目录包含两个独立的增强型 Agent，可根据需要选择：

## 1. BasicEnhancedAgent (基础增强版)
**文件**: `basic_enhanced_agent.py`

**特点**:
- ✅ 神经网络基础评分（前14维特征）
- ✅ 基础规则加成（成方、跳吃、安全性、准方）
- ❌ 不包含褡裢特征
- 适合对比实验或不需要褡裢策略的场景

**使用示例**:
```python
from agent.basic_enhanced_agent import BasicEnhancedAgent

# 创建Agent
agent = BasicEnhancedAgent(
    model_path='exp/jcar_sft_2025_balanced/checkpoint_best.pt',
    device='cuda',
    square_weight=3.0,   # 成方权重
    eat_weight=2.5,      # 跳吃权重
    safety_weight=1.0,   # 安全性权重
    triple_weight=2.0,   # 准方权重
    break_weight=1.5,    # 破坏对方权重
    verbose=False
)

# 选择走法
move, info = agent.select_move(game_state)
print(f"Value: {info['value']}, Prob: {info['prob']}")
```

## 2. DalianEnhancedAgent (褡裢增强版)
**文件**: `dalian_enhanced_agent.py`

**特点**:
- ✅ 神经网络基础评分（前14维特征）
- ✅ 完整规则加成（成方、跳吃、安全性、准方）
- ✅ 褡裢策略增强（形成、利用、破坏褡裢）
- 适合实际对战，能够发挥褡裢战术优势

**使用示例**:
```python
from agent.dalian_enhanced_agent import DalianEnhancedAgent

# 创建Agent
agent = DalianEnhancedAgent(
    model_path='exp/jcar_sft_2025_balanced/checkpoint_best.pt',
    device='cuda',
    # 基础规则权重
    square_weight=3.0,
    eat_weight=2.5,
    safety_weight=1.0,
    triple_weight=0.8,
    break_weight=1.2,
    capture_weight=1.5,
    # 褡裢权重（褡裢是关键战术）
    dalian_create_weight=12.0,   # 形成褡裢
    dalian_use_weight=20.0,      # 利用褡裢吃子（最高优先级）
    dalian_break_weight=10.0,    # 破坏对方褡裢
    pre_dalian_weight=5.0,       # 准褡裢（布局阶段）
    verbose=False
)

# 选择走法
move, info = agent.select_move(game_state)
print(f"Value: {info['value']}, Prob: {info['prob']}")

# 查看褡裢相关信息
if 'creates_dalian' in info:
    print(f"形成褡裢: {info['creates_dalian']}")
    print(f"利用褡裢: {info['uses_dalian']}")
```

## 快速对比测试

```python
from jiu.jiuboard_fast import GameState
from agent.basic_enhanced_agent import BasicEnhancedAgent
from agent.dalian_enhanced_agent import DalianEnhancedAgent

# 创建两个Agent
basic_agent = BasicEnhancedAgent(
    'exp/jcar_sft_2025_balanced/checkpoint_best.pt',
    device='cpu'
)

dalian_agent = DalianEnhancedAgent(
    'exp/jcar_sft_2025_balanced/checkpoint_best.pt', 
    device='cpu'
)

# 简单测试
state = GameState.new_game(14)

# 基础版选择
move1, info1 = basic_agent.select_move(state)
print(f"Basic: {move1}, Value: {info1['value']:.3f}")

# 褡裢版选择
move2, info2 = dalian_agent.select_move(state)
print(f"Dalian: {move2}, Value: {info2['value']:.3f}")

# 可能会选择相同或不同的走法
print(f"选择相同: {move1 == move2}")
```

## 特征维度说明

两个Agent都使用增强特征，但利用的维度不同：

### 共同使用的特征 (0-25维):
- [0-13]: 基础特征（模型输入，14维）
- [14-25]: 增强特征（规则加成，12维）
  - 成方、跳吃、安全性、准方、破坏对方等

### DalianEnhancedAgent 额外使用 (26-31维):
- [26]: my_dalian_count_norm - 我方褡裢数
- [27]: opp_dalian_count_norm - 对方褡裢数
- [28]: creates_dalian - 是否形成褡裢
- [29]: uses_dalian - 是否利用褡裢吃子
- [30]: breaks_opp_dalian - 是否破坏对方褡裢
- [31]: creates_pre_dalian - 是否形成准褡裢

## 性能对比

根据测试，DalianEnhancedAgent（含褡裢）vs BasicEnhancedAgent（不含褡裢）：
- 褡裢版胜率显著提升
- 褡裢策略能够创造战术优势
- 在关键时刻利用褡裢可以扭转局势

建议：
- **训练/调试**: 使用 BasicEnhancedAgent 快速迭代
- **实战/评估**: 使用 DalianEnhancedAgent 发挥最大实力
