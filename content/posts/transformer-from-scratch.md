---
title: 从零实现 Transformer：3000 行 NumPy 代码构建完整语言模型
date: 2026-10-07
description: 深入实践 - 使用纯 NumPy 从零实现 Transformer、BPE 分词器和训练流程，理解大语言模型的每一个细节
series: LLM 从零实现
seriesOrder: 2
categories:
  - 技术
tags:
  - 深度学习
  - Transformer
  - NLP
  - Python
  - NumPy
  - 机器学习
hidden: false
haloPublished: true
---

# 从零实现 Transformer：3000 行 NumPy 代码构建完整语言模型

> 💡 **项目地址**: [github.com/liangqianxing/lm-lab](https://github.com/liangqianxing/lm-lab)  
> 📊 **完成度**: BPE 分词器（71 tests ✓）、Transformer 架构、优化器、完整文档

在[上一篇文章](https://liangqianxing.github.io/posts/bpe-tokenizer-from-scratch)中，我们实现了 BPE 分词器。今天，我将分享如何**从零实现一个完整的 Transformer 语言模型**，使用纯 NumPy，不依赖任何深度学习框架。

## 🎯 为什么要从零实现？

在使用 PyTorch/TensorFlow 时，我们常常：
- 调用 `nn.MultiheadAttention()` 却不知道内部如何计算
- 使用 `Adam` 优化器但不理解偏差校正
- 看到梯度消失/爆炸却不知道如何推导

**从零实现的价值**：
- 🧠 **深度理解**：手动推导每个公式，理解每个操作
- 🔍 **透明性**：没有黑盒，每行代码都清晰可见
- 🐛 **调试能力**：知道哪里可能出错，如何修复
- 📚 **教育意义**：最佳的学习方式

## 📊 项目概览

```
lm-lab/
├── tokenization/   ✅ BPE 分词器（71 tests passed）
├── model/          ✅ Transformer 架构
│   ├── attention.py      # 因果自注意力
│   ├── layer_norm.py     # 层归一化
│   ├── feedforward.py    # 前馈网络
│   ├── embedding.py      # 位置编码
│   └── transformer.py    # 完整模型
├── optimizer/      ✅ SGD, Adam, AdamW
├── training/       ⏳ 训练流程（进行中）
└── docs/           ✅ 完整文档（5 份）
```

**技术栈**：
- 🐍 Python 3.11+
- 🔢 纯 NumPy 实现
- ✅ Pytest 测试
- 📖 完整文档

## 🏗️ 核心架构

### 1. 因果自注意力机制

注意力是 Transformer 的核心。我们实现的因果自注意力确保每个 token 只能看到它之前的信息。

```python
class CausalSelfAttention:
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, seq_len, d_model)
        output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. 投影到 Q, K, V
        qkv = x @ self.W_qkv + self.b_qkv  # (batch, seq_len, 3*d_model)
        q, k, v = np.split(qkv, 3, axis=-1)
        
        # 2. 重塑为多头格式
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        q = q.transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        k = k.transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        v = v.transpose(0, 2, 1, 3)
        
        # 3. 缩放点积注意力
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        
        # 4. 因果掩码（关键！）
        mask = np.tril(np.ones((seq_len, seq_len)))
        scores = np.where(mask == 1, scores, -1e10)
        
        # 5. Softmax
        attn_weights = self._softmax(scores)
        
        # 6. 应用注意力
        out = attn_weights @ v
        
        # 7. 拼接多头
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # 8. 输出投影
        return out @ self.W_out + self.b_out
```

**关键点**：
- ✅ 因果掩码用下三角矩阵实现
- ✅ 缩放因子 `1/sqrt(d_k)` 防止梯度消失
- ✅ 数值稳定的 softmax（减去最大值）

### 2. 反向传播推导

这是最难的部分！让我们看看注意力的梯度计算：

```python
def backward(self, grad_output: np.ndarray) -> np.ndarray:
    """完整的反向传播"""
    # 梯度通过输出投影
    grad_out_pre_proj = grad_output @ self.W_out.T
    self.grad_W_out = grad_output.reshape(-1, d_model).T @ \
                      out_pre_proj.reshape(-1, d_model)
    
    # 梯度通过多头拼接
    grad_out_pre_proj = grad_out_pre_proj.reshape(
        batch_size, seq_len, self.num_heads, self.d_k
    ).transpose(0, 2, 1, 3)
    
    # 梯度通过 attn_weights @ v
    grad_attn_weights = grad_out_pre_proj @ v.transpose(0, 1, 3, 2)
    grad_v = attn_weights.transpose(0, 1, 3, 2) @ grad_out_pre_proj
    
    # 梯度通过 softmax（重要！）
    grad_scores = attn_weights * (
        grad_attn_weights - 
        np.sum(grad_attn_weights * attn_weights, axis=-1, keepdims=True)
    )
    
    # 应用因果掩码到梯度
    grad_scores = np.where(mask == 1, grad_scores, 0)
    
    # 梯度通过缩放
    grad_scores = grad_scores / math.sqrt(self.d_k)
    
    # 梯度通过 QK^T
    grad_q = grad_scores @ k
    grad_k = grad_scores.transpose(0, 1, 3, 2) @ q
    
    # ... 继续向后传播
    return grad_x
```

**关键推导**：

Softmax 的梯度是最复杂的部分：

$$
\frac{\partial L}{\partial s_i} = a_i \left( \frac{\partial L}{\partial a_i} - \sum_j a_j \frac{\partial L}{\partial a_j} \right)
$$

其中 $a = \text{softmax}(s)$。

### 3. Layer Normalization

层归一化对训练稳定性至关重要：

```python
class LayerNorm:
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        归一化最后一个维度
        x: (..., d_model)
        """
        # 计算均值和方差
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        
        # 归一化
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # 学习的缩放和平移
        return self.gamma * x_norm + self.beta
```

**为什么需要 LayerNorm？**
- 🎯 稳定训练：防止激活值过大或过小
- 🚀 加速收敛：允许更大的学习率
- 🔧 Pre-LN vs Post-LN：我们使用 Pre-LN（更稳定）

### 4. 前馈网络

简单但重要的两层 MLP：

```python
class FeedForward:
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        FFN(x) = GELU(xW1 + b1)W2 + b2
        """
        # 第一层 + GELU 激活
        hidden = x @ self.W1 + self.b1
        hidden = self._gelu(hidden)
        
        # 第二层
        return hidden @ self.W2 + self.b2
    
    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        """
        GELU 激活函数（比 ReLU 更平滑）
        """
        return 0.5 * x * (1.0 + np.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
        ))
```

**为什么用 GELU？**
- 📈 平滑的非线性
- 🎯 更好的梯度流
- ✅ GPT 系列的标准选择

### 5. 完整的 Transformer Block

组合所有组件：

```python
class TransformerBlock:
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Pre-LN Transformer Block
        """
        # 自注意力 + 残差
        attn_input = self.ln1.forward(x)
        attn_output = self.attention.forward(attn_input)
        x = x + attn_output
        
        # 前馈 + 残差
        ffn_input = self.ln2.forward(x)
        ffn_output = self.ffn.forward(ffn_input)
        x = x + ffn_output
        
        return x
```

**Pre-LN 架构**：
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

优于 Post-LN（原始论文）：
- ✅ 训练更稳定
- ✅ 不需要学习率预热
- ✅ GPT-2/GPT-3 使用的架构

### 6. 完整语言模型

```python
class TransformerLM:
    def __init__(self, vocab_size, d_model, num_heads, 
                 num_layers, d_ff, max_seq_len):
        # Token 嵌入
        self.token_embeddings = np.random.randn(vocab_size, d_model) * 0.02
        
        # 位置编码
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_len)
        
        # N 个 Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len)
            for _ in range(num_layers)
        ]
        
        # 最终归一化
        self.ln_f = LayerNorm(d_model)
        
        # 权重绑定：输出层共享输入嵌入
        # logits = x @ token_embeddings.T
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        input_ids: (batch, seq_len)
        output: (batch, seq_len, vocab_size)
        """
        # 1. Token 嵌入
        x = self.token_embeddings[input_ids]
        
        # 2. 添加位置编码
        x = self.pos_encoding.forward(x)
        
        # 3. 通过所有 Transformer blocks
        for block in self.blocks:
            x = block.forward(x)
        
        # 4. 最终归一化
        x = self.ln_f.forward(x)
        
        # 5. 投影到词表（权重绑定）
        logits = x @ self.token_embeddings.T
        
        return logits
```

## 🔧 优化器实现

### Adam 优化器

```python
class Adam:
    def step(self, params: dict, grads: dict):
        """Adam 更新规则"""
        self.t += 1
        
        for name, param in params.items():
            grad = grads[name]
            
            # 更新一阶矩（动量）
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            
            # 更新二阶矩（RMSProp）
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad**2
            
            # 偏差校正（重要！）
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # 参数更新
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
```

**关键概念**：
- 🎯 **一阶矩**：梯度的指数移动平均（动量）
- 📊 **二阶矩**：梯度平方的指数移动平均（自适应学习率）
- ⚖️ **偏差校正**：补偿初始化为零的偏差

### AdamW（推荐）

```python
class AdamW(Adam):
    def step(self, params: dict, grads: dict):
        """AdamW：解耦权重衰减"""
        self.t += 1
        
        for name, param in params.items():
            grad = grads[name]
            
            # Adam 步骤（不含权重衰减）
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad**2
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            
            # 解耦的权重衰减
            param -= self.learning_rate * self.weight_decay * param
```

**AdamW vs Adam**：
- ❌ Adam：权重衰减加到梯度上（次优）
- ✅ AdamW：权重衰减直接应用于参数（更好）

## 🎓 数值稳定性技巧

### 1. 稳定的 Softmax

```python
def stable_softmax(x, axis=-1):
    """防止数值溢出"""
    # 减去最大值
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    
    # 指数和归一化
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

**为什么要减去最大值？**
- $e^{1000}$ 会溢出 → Inf
- $e^{1000 - 1000} = e^0 = 1$ ✓

### 2. 梯度裁剪

```python
def clip_grad_norm(gradients, max_norm=1.0):
    """全局范数裁剪"""
    # 计算总范数
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
    
    # 裁剪系数
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for g in gradients:
            g *= clip_coef
    
    return total_norm
```

**防止梯度爆炸**：
- 计算所有梯度的全局范数
- 如果超过阈值，按比例缩放
- 保持梯度方向不变

### 3. 权重初始化

```python
# Xavier/Glorot 初始化
scale = 1.0 / math.sqrt(d_model)
W = np.random.randn(d_model, d_ff) * scale

# 或使用 He 初始化（ReLU）
scale = np.sqrt(2.0 / d_model)
W = np.random.randn(d_model, d_ff) * scale
```

## 📊 完整训练示例

```python
import numpy as np
from lm_lab.tokenization import train_bpe, BPETokenizer
from lm_lab.model import TransformerLM
from lm_lab.optimizer import AdamW
from lm_lab.utils import set_seed

# 1. 设置随机种子
set_seed(42)

# 2. 训练分词器
texts = ["The cat sat on the mat.", "Machine learning is fun.", ...]
model = train_bpe(texts, vocab_size=500, pretokenization="simple")
tokenizer = BPETokenizer(model)

# 3. 创建模型
model = TransformerLM(
    vocab_size=500,
    d_model=256,
    num_heads=8,
    num_layers=4,
    d_ff=1024,
    max_seq_len=128
)

# 4. 创建优化器
optimizer = AdamW(learning_rate=3e-4, weight_decay=0.01)

# 5. 训练循环
for epoch in range(num_epochs):
    # 准备批次
    inputs, targets = prepare_batch(texts, tokenizer)
    
    # 前向传播
    logits = model.forward(inputs)
    
    # 计算损失和梯度
    loss, grad_logits = compute_loss(logits, targets)
    
    # 反向传播
    model.backward(grad_logits)
    
    # 梯度裁剪
    grads = model.gradients()
    clip_grad_norm(grads, max_norm=1.0)
    
    # 优化步骤
    params = model.parameters()
    param_dict = {f'p{i}': p for i, p in enumerate(params)}
    grad_dict = {f'p{i}': g for i, g in enumerate(grads)}
    optimizer.step(param_dict, grad_dict)
    
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

## 🔍 调试技巧

### 1. 梯度检查

```python
def numerical_gradient(f, x, eps=1e-5):
    """数值梯度（用于验证）"""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        x[idx] = old_value + eps
        fxh_plus = f(x)
        
        x[idx] = old_value - eps
        fxh_minus = f(x)
        
        grad[idx] = (fxh_plus - fxh_minus) / (2 * eps)
        x[idx] = old_value
        it.iternext()
    
    return grad

# 验证注意力的梯度
analytical_grad = attention.backward(grad_output)
numerical_grad = numerical_gradient(lambda x: attention.forward(x), x)

# 相对误差应该很小（< 1e-5）
rel_error = np.abs(analytical_grad - numerical_grad) / \
            (np.abs(analytical_grad) + np.abs(numerical_grad) + 1e-8)
print(f"Gradient check: max relative error = {rel_error.max()}")
```

### 2. 形状调试

```python
from lm_lab.utils import debug_shapes

# 打印所有中间张量的形状
print(debug_shapes(
    q=q, k=k, v=v,
    scores=scores,
    attn_weights=attn_weights,
    output=output
))

# 输出：
# q: (32, 8, 128, 64)
# k: (32, 8, 128, 64)
# v: (32, 8, 128, 64)
# scores: (32, 8, 128, 128)
# attn_weights: (32, 8, 128, 128)
# output: (32, 128, 512)
```

### 3. 过拟合小数据集

```python
# 测试模型是否能学习
tiny_dataset = ["Hello world"] * 100

# 模型应该能完美过拟合
for epoch in range(100):
    loss = train_step(tiny_dataset)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 预期：损失应该降到接近 0
```

## 📈 性能对比

| 实现方式 | 训练速度 | 可读性 | 教育价值 |
|---------|---------|--------|---------|
| NumPy（本项目） | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| PyTorch | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| TensorFlow | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

**NumPy 实现的优势**：
- 🎓 **教育目的**：每个操作都透明
- 🔍 **调试友好**：可以打印任何中间结果
- 📚 **理解深度**：必须理解每个公式
- 🚀 **可移植**：纯 Python，无复杂依赖

**劣势**：
- ⏱️ 速度慢（无 GPU 加速）
- 💾 内存效率低
- ⚠️ 数值稳定性需要手动处理

## 🎯 学到的核心概念

### 理论层面

1. **注意力机制**
   - Scaled dot-product attention 的数学原理
   - 为什么需要缩放（`1/sqrt(d_k)`）
   - 因果掩码如何实现自回归

2. **反向传播**
   - Softmax 梯度的链式法则
   - 残差连接的梯度流
   - 多头注意力的梯度分配

3. **归一化**
   - LayerNorm vs BatchNorm
   - Pre-LN vs Post-LN 的区别
   - 为什么归一化能稳定训练

4. **优化算法**
   - Adam 的一阶和二阶动量
   - 偏差校正的必要性
   - AdamW 为什么更好

### 实现层面

1. **NumPy 技巧**
   - 张量操作和形状变换
   - 广播机制的应用
   - 数值稳定性处理

2. **调试方法**
   - 梯度检查
   - 形状断言
   - 小数据集过拟合测试

3. **软件工程**
   - 模块化设计
   - 接口设计
   - 测试驱动开发

## 🚀 扩展方向

### 短期（1-2 周）

- [ ] 完成训练流程实现
- [ ] 添加更多测试
- [ ] 在小数据集上验证收敛

### 中期（1 个月）

- [ ] 实现 Flash Attention
- [ ] 添加分布式训练支持
- [ ] 性能优化

### 长期（2-3 个月）

- [ ] 迁移到 GPU（CUDA/Triton）
- [ ] 实现更多优化技巧
- [ ] Scaling laws 实验

## 📚 推荐资源

### 论文

1. **Attention is All You Need** (Vaswani et al., 2017)
   - Transformer 原始论文
   - 必读经典

2. **GPT-2** (Radford et al., 2019)
   - 自回归语言建模
   - Pre-LN 架构

3. **AdamW** (Loshchilov & Hutter, 2017)
   - 解耦权重衰减
   - 为什么 AdamW 比 Adam 好

### 课程

- **CS336: Language Modeling from Scratch** (Stanford)
- **CS224N: NLP with Deep Learning** (Stanford)
- **The Illustrated Transformer** (Jay Alammar)

### 代码

- **本项目**: [github.com/liangqianxing/lm-lab](https://github.com/liangqianxing/lm-lab)
- **nanoGPT**: Andrej Karpathy 的最小 GPT 实现
- **minGPT**: 另一个教育性 GPT 实现

## 💡 总结

通过这个项目，我们：

✅ **实现了完整的 Transformer**（~3000 行代码）  
✅ **手动推导了所有梯度**（深入理解反向传播）  
✅ **验证了分词器**（71 tests passed）  
✅ **创建了完整文档**（5 份详细文档）  
✅ **使用纯 NumPy**（无框架依赖）

**关键收获**：
- 🧠 理解了 Transformer 的每一个细节
- 🔍 掌握了数值稳定性技巧
- 📐 学会了手动推导梯度
- 🎯 具备了调试深度学习模型的能力

**适合人群**：
- 想深入理解 Transformer 的学习者
- 准备面试深度学习职位的工程师
- 对大语言模型原理感兴趣的研究者
- CS336 课程的学习者

## 🎉 下一步

1. **克隆项目**：
```bash
git clone https://github.com/liangqianxing/lm-lab
cd lm-lab
```

2. **安装依赖**：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -e ".[dev]"
```

3. **运行测试**：
```bash
pytest tests/tokenization/ -v  # 71 tests
```

4. **阅读文档**：
- [快速开始](https://github.com/liangqianxing/lm-lab/blob/master/docs/quickstart.md)
- [架构设计](https://github.com/liangqianxing/lm-lab/blob/master/docs/architecture.md)
- [API 参考](https://github.com/liangqianxing/lm-lab/blob/master/docs/api_reference.md)

5. **开始学习**：
从简单的 `LayerNorm` 开始，逐步理解每个组件，最后掌握完整的 Transformer！

---

**感谢阅读！** 如果这篇文章对你有帮助，欢迎：
- ⭐ Star 项目：[github.com/liangqianxing/lm-lab](https://github.com/liangqianxing/lm-lab)
- 💬 提出问题和建议
- 📢 分享给更多人

让我们一起探索大语言模型的奥秘！🚀
