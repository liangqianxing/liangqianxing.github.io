---
title: "DPO 精读：不用 PPO，如何直接从偏好数据对齐语言模型"
date: 2026-07-19 09:15:00
description: 从 KL 约束的 RLHF 目标出发，推导 DPO 的闭式策略映射、偏好损失、训练流程、实验边界与复现要点。
series: 三大会论文精读
seriesOrder: 2
categories:
  - AI
tags:
  - LLM
  - DPO
  - RLHF
  - 偏好对齐
  - 模型后训练
  - NeurIPS
hidden: true
haloPublished: true
---

DPO 解决的问题很直接：**已有成对偏好数据时，能否跳过显式奖励模型和 PPO，直接把语言模型训练成更偏向获胜回答的策略？**

论文给出的答案是：在 Bradley-Terry 偏好模型和 KL 约束的奖励最大化目标下，可以把奖励函数改写成策略与参考策略的对数概率比。代回偏好模型后，未知的配分函数会相消，最终只剩一个二分类形式的损失。

这让偏好优化从“奖励模型 + 在线采样 + 强化学习”缩短为一次离线监督式训练。但它没有消除偏好数据、参考模型和评估的难题，也没有证明 DPO 在所有规模、分布和偏好噪声下都优于 PPO。本文会把论文结论与我的工程判断分开。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** |
| 作者 | Rafael Rafailov、Archit Sharma、Eric Mitchell、Christopher D. Manning、Stefano Ermon、Chelsea Finn |
| 会议 | NeurIPS 2023 |
| 主题 | 偏好学习、RLHF、语言模型后训练、离线策略优化 |
| 官方论文 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html) |
| 官方补充材料 | [NeurIPS Supplemental](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Supplemental-Conference.pdf) |
| 作者版本与许可 | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)，CC BY 4.0 |
| 官方代码 | [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization) |

**选题理由**：DPO 把一个复杂的 RLHF 系统问题压缩成可直接实现的损失函数，同时保留了参考策略和 KL 正则背后的统计含义。它既适合从公式理解，也适合从数据、显存和训练循环理解，是工程读者进入偏好优化的关键论文。

## 问题背景：RLHF 为什么复杂

经典 RLHF 通常包含三段：

1. 用高质量示范做监督微调，得到 SFT 策略 $\pi_{\mathrm{SFT}}$；
2. 对同一提示采样多个回答，请标注者给出偏好，再训练奖励模型 $r_\phi(x,y)$；
3. 用 PPO 等强化学习算法最大化奖励，同时用 KL 惩罚限制策略不要离参考模型太远。

第三阶段常写成：

$$
\max_{\pi_\theta}
\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot|x)}
\left[r_\phi(x,y)\right]
-\beta D_{\mathrm{KL}}
\left(\pi_\theta(\cdot|x)\,\|\,\pi_{\mathrm{ref}}(\cdot|x)\right)
$$

其中 $\beta$ 控制偏离参考策略的代价。KL 约束不是装饰：奖励模型只在有限数据分布上可靠，如果策略为了高分跑到分布外，可能利用奖励模型漏洞，生成高奖励但低质量的文本。

系统复杂性来自几个方向：

- 需要同时维护策略、参考模型、奖励模型，有时还要价值模型；
- PPO 训练过程中要反复从当前策略采样，数据分布随参数变化；
- 离散文本不能直接对采样结果求梯度，需要策略梯度和方差控制；
- 奖励尺度、KL 系数、采样温度、优势估计和裁剪都会影响稳定性。

![RLHF 与 DPO 的偏好学习流程对比](/images/posts/dpo-direct-preference-optimization/dpo-vs-rlhf.png)

*图源：Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html), Figure 1, NeurIPS 2023；取自作者 [CC BY 4.0 arXiv 源码](https://arxiv.org/abs/2305.18290)。原图用于论文解读。*

Figure 1 最重要的差别不是少画了几个框，而是 DPO 的训练数据可以保持离线：每条样本已经包含提示 $x$、偏好回答 $y_w$ 和非偏好回答 $y_l$，训练循环中不必再调用当前策略生成回答，也不显式拟合一个独立奖励模型。

## 核心贡献

论文的贡献可以拆成四层：

1. **闭式策略映射**：给出 KL 约束奖励最大化问题的最优策略形式，并反向把奖励写成策略概率比。
2. **直接偏好损失**：把 Bradley-Terry 奖励似然改写为策略似然，得到只依赖策略、参考策略和偏好对的二分类损失。
3. **隐式奖励解释**：证明语言模型的对数概率比可以表示偏好模型中的奖励等价类，不因这种参数化丢失所需的一般性。
4. **跨任务实证**：在情感控制、摘要和单轮对话上比较 DPO、PPO、Preferred-FT、Unlikelihood 与 Best of N，并补充人类评估和分布外摘要测试。

**论文结论**：在论文的任务、模型规模和评估协议下，DPO 的效果与 PPO 相当或更好，训练管线更简单，且对采样温度更稳健。

**我的判断**：真正有长期价值的不是“某个 win rate 高了几点”，而是 DPO 把参考策略相对概率直接放进目标函数，使偏好优化变成标准反向传播问题。这降低了实现和调试门槛，也让后续研究可以围绕损失、偏好噪声和正则化快速迭代。

## 方法总览：从奖励到策略概率比

### 1. Bradley-Terry 偏好模型

给定同一提示 $x$ 的两个回答 $y_1,y_2$，Bradley-Terry 模型假设人更偏好 $y_1$ 的概率为：

$$
p^*(y_1 \succ y_2|x)
=\frac{\exp(r^*(x,y_1))}
{\exp(r^*(x,y_1))+\exp(r^*(x,y_2))}
=\sigma\left(r^*(x,y_1)-r^*(x,y_2)\right)
$$

传统奖励模型训练就是对偏好对做二分类：提高获胜回答 $y_w$ 的奖励，降低落败回答 $y_l$ 的奖励。

这个模型只关心奖励差。如果给同一个提示下的所有回答都加上同一个 $f(x)$，偏好概率不会变化。论文把这称为奖励函数的等价类。

### 2. KL 约束目标的最优策略

对任意奖励函数 $r(x,y)$，前面的 KL 约束目标有闭式最优解：

$$
\pi_r(y|x)
=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y|x)
\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

其中

$$
Z(x)=\sum_y \pi_{\mathrm{ref}}(y|x)
\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

是配分函数。直接计算 $Z(x)$ 几乎不可能，因为回答空间包含所有可能的 token 序列。但把上式取对数并移项，可得：

$$
r(x,y)
=\beta\log\frac{\pi_r(y|x)}{\pi_{\mathrm{ref}}(y|x)}
+\beta\log Z(x)
$$

### 3. 配分函数为什么会消失

把奖励改写式代入 Bradley-Terry 模型时，同一个提示下两个回答都含有 $\beta\log Z(x)$。偏好概率只看奖励差，因此这项相消：

$$
p^*(y_w\succ y_l|x)
=\sigma\left(
\beta\log\frac{\pi^*(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
-\beta\log\frac{\pi^*(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}
\right)
$$

于是可以直接用参数化策略 $\pi_\theta$ 做最大似然估计，而不需要求 $Z(x)$，也不需要先得到 $r_\phi$。

### 4. DPO 损失

对数据集 $\mathcal{D}=\{(x,y_w,y_l)\}$，DPO 的损失为：

$$
\mathcal{L}_{\mathrm{DPO}}
=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[
\log\sigma\left(
\beta
\left(
\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
-\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}
\right)
\right)
\right]
$$

实现时，序列对数概率是回答 token 的条件对数概率之和。定义：

$$
\Delta_\theta
=\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x)
$$

$$
\Delta_{\mathrm{ref}}
=\log\pi_{\mathrm{ref}}(y_w|x)-\log\pi_{\mathrm{ref}}(y_l|x)
$$

则单个样本的代码几乎就是：

```python
loss = -logsigmoid(beta * (delta_policy - delta_reference))
```

这行代码的含义不是单纯“提高 $y_w$、压低 $y_l$”，而是要求当前策略相对于参考策略，更明显地偏向 $y_w$。参考项提供了锚点，避免目标退化成无约束地压低落败回答概率。

## 隐式奖励与动态样本权重

DPO 隐式定义了一个奖励：

$$
\hat r_\theta(x,y)
=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}
$$

如果当前隐式奖励已经正确拉开 $y_w$ 和 $y_l$，sigmoid 权重会变小；如果模型仍把 $y_l$ 排在 $y_w$ 前面，梯度权重会更大。论文据此解释 DPO 的更新：

- 增大偏好回答 $y_w$ 的对数概率；
- 减小非偏好回答 $y_l$ 的对数概率；
- 根据当前隐式奖励把样本“排错了多少”动态调节更新强度。

论文还证明，在 Bradley-Terry / Plackett-Luce 偏好模型下，每个与偏好分布一致的奖励等价类，都能由某个 $\beta\log(\pi/\pi_{\mathrm{ref}})$ 表示。这里的边界很重要：证明依赖偏好模型和温和的支持集假设，并不意味着任意现实人类偏好都严格服从 Bradley-Terry。

## 训练与推理流程

### 训练前准备

1. 准备监督微调模型，并把它冻结为参考策略 $\pi_{\mathrm{ref}}$。
2. 对提示采样成对回答并收集偏好标签，或直接使用公开偏好数据。
3. 若公开数据对应的原始 SFT 模型不可用，论文先在偏好回答 $y_w$ 上做最大似然训练，得到分布更匹配的参考模型。

第三步不是可有可无。若偏好回答对参考模型来说严重离分布，概率比会混入很强的模型分布差异，偏好优化就不再只是学习“哪个回答更好”。

### 每个 DPO 训练步

1. 将同一提示与 $y_w,y_l$ 拼成两条序列。
2. 分别计算策略模型对两条回答的序列 log probability。
3. 计算冻结参考模型的对应 log probability；参考模型固定时，这些值可预先缓存以节省计算。
4. 得到 $\Delta_\theta-\Delta_{\mathrm{ref}}$，送入 `logsigmoid`。
5. 只更新策略模型参数，参考模型不反向传播。

论文默认使用 $\beta=0.1$、batch size 64、Adam、学习率 $10^{-6}$，前 150 步从 0 线性 warmup 到 $10^{-6}$；TL;DR 摘要任务使用 $\beta=0.5$。

### 推理

部署时只需要训练后的策略模型。参考模型、偏好对和 DPO loss 都不在推理链路中，因此 DPO 本身不会增加单次生成的模型数量或额外解码步骤。

不过，DPO 也不会自动降低策略模型的推理成本；吞吐和显存仍由基础模型架构、精度、上下文长度和推理引擎决定。

## 实验设置

论文覆盖三个开放式文本任务：

| 任务 | 模型与数据 | 偏好来源 | 主要评估 |
| --- | --- | --- | --- |
| 正向情感生成 | GPT-2-large；IMDb 前缀 | 情感分类器为生成结果排序 | 真值奖励与相对参考策略的 KL 前沿 |
| Reddit TL;DR 摘要 | GPT-J 6B SFT；人工偏好数据 | Stiennon 等人收集的人类偏好 | GPT-4 对人写摘要的胜率；补充人评 |
| 单轮对话 | Pythia-2.8B；Anthropic Helpful-Harmless | 约 17 万段对话末尾的偏好对 | GPT-4 对数据集中获胜回答的胜率 |

情感实验先对 IMDb 子集做 1 个 epoch 的 SFT，再对 25,000 个前缀各采样 4 个回答，由此构造每个前缀 6 个偏好对。作者比较了 DPO、PPO、可访问真实奖励的 PPO-GT、Preferred-FT 和 Unlikelihood，并围绕保守程度做了 22 次训练运行。

真实偏好任务无法访问“真奖励”，所以论文用 GPT-4 作为自动裁判。作者没有把它当成无条件可靠的真值，而是另外做了人类研究，检查 GPT-4 与人的一致性。

## 主要结果

### 1. 情感控制：同等 KL 下取得更高奖励

![IMDb 情感生成的奖励-KL 前沿](/images/posts/dpo-direct-preference-optimization/reward-kl-frontier.png)

*图源：Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html), Figure 2 (left), NeurIPS 2023；取自作者 [CC BY 4.0 arXiv 源码](https://arxiv.org/abs/2305.18290)。原图用于论文解读。*

横轴是策略相对参考策略的序列级 KL，纵轴是情感分类器给出的真实奖励。不能只比较最高奖励，因为更大的 KL 意味着策略离参考分布更远。

**论文结论**：在这组受控实验里，DPO 的奖励-KL 前沿严格优于各个 PPO 版本，甚至优于直接访问真实奖励的 PPO-GT。这更像是在说明 DPO 对目标的优化更容易，而不是说明隐式奖励比真实奖励“信息更多”。

### 2. TL;DR 摘要：约 61% 对 57%

![TL;DR 摘要胜率随采样温度变化](/images/posts/dpo-direct-preference-optimization/tldr-winrate-temperature.png)

*图源：Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html), Figure 2 (right), NeurIPS 2023；取自作者 [CC BY 4.0 arXiv 源码](https://arxiv.org/abs/2305.18290)。原图用于论文解读。*

在 GPT-4 自动评估下，DPO 以温度 0 采样时对人写参考摘要的胜率约为 61%，PPO 的最佳结果约为 57%，同样出现在温度 0。DPO 在高温下也会退化，但曲线整体比 PPO 稳定。

人类头对头评估中，温度 0.25 的 DPO 摘要相对温度 0 的 PPO 摘要获得 58% 胜率。GPT-4 的简洁版提示给出 47%，强调简洁性的提示给出 54%，说明裁判提示本身会改变结论。

### 3. 单轮对话：超过数据集中的偏好回答

![Anthropic-HH 单轮对话胜率随采样温度变化](/images/posts/dpo-direct-preference-optimization/dialogue-winrate-temperature.png)

*图源：Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html), Figure 3 (left), NeurIPS 2023；取自作者 [CC BY 4.0 arXiv 源码](https://arxiv.org/abs/2305.18290)。原图用于论文解读。*

虚线 0.5 表示与数据集中获胜回答打平。DPO 和 Best of 128 越过了这条线；Preferred-FT 和 2-shot Pythia-2.8B 没有。Figure 3 中 DPO 在温度 1.0 的胜率读图约为 0.63，但这是 GPT-4 自动裁判下的图示值，不应当解释为通用对话能力提高 63%。

Best of 128 每次推理需要采样并打分 128 个回答，成本很高；DPO 只需一次正常策略采样。这使“相近质量下的推理成本”成为 DPO 更有工程意义的比较维度。

### 4. 分布外摘要

作者把 Reddit TL;DR 上训练的模型直接用于 CNN/DailyMail 新闻摘要：

| 方法 | 温度 0 | 温度 0.25 |
| --- | ---: | ---: |
| DPO | **0.36** | **0.31** |
| PPO | 0.26 | 0.23 |

这是 GPT-4 对数据集真值摘要的胜率。它提供了初步的分布外证据，但只覆盖一个新数据集和两个温度，论文也明确要求更全面的研究。

## 消融与关键对照

### 动态权重不是普通 Unlikelihood

最接近 DPO 的朴素目标，是提高 $\log\pi(y_w|x)$ 并直接压低 $\log\pi(y_l|x)$。论文的 Unlikelihood 基线在情感控制上能工作，但在摘要和对话上生成大量重复的 `when`，失去可读性，因此没有进入这两项主实验。

**论文观察**：DPO 通过参考策略概率比和 sigmoid 动态权重约束更新；去掉这些结构，单纯降低 $y_l$ 的概率会使复杂生成任务退化。

**我的判断**：这也是实现 DPO 时不能只写“chosen loss - rejected loss”的原因。负样本概率被压低到哪里、相对哪个模型压低、样本已经排对时是否继续施压，决定了训练是否稳定。

### $\beta$ 控制的不是一个固定 KL 值

受控实验搜索 $\beta\in\{0.05,0.1,1,5\}$，主实验默认 0.1，TL;DR 用 0.5。理论上，更大的 $\beta$ 对应更强的 KL 约束；实践中，最终 KL 还受偏好数据、优化步数、学习率和模型分布影响，不能把 $\beta$ 当成可直接指定的 KL 距离。

### Best of N 的收益会饱和

补充材料的 Figure 4 显示，摘要和对话的 Best of N 大约在 $N=64$ 到 $128$ 后进入平台期。它说明强奖励模型加重采样是有竞争力的基线，也说明把大量成本搬到推理阶段并不会无限带来收益。

## 失败案例与局限

论文补充材料没有只展示成功样例。Table 9 中，DPO 回答“美国为何加入二战”时生成了一段流畅但包含错误概念的长文，把与该历史语境无关的说法混入答案；参考回答更直接地指出珍珠港事件。DPO 学到偏好不等于学到事实真值。

Table 10 又暴露了评估器问题：面对“7 加 2”，DPO 虽然冗长但给出了 9，数据集参考回答却是 11；GPT-4 裁判反而判参考回答正确。自动胜率可能同时受到候选模型错误、参考答案错误和裁判错误影响。

论文自己列出的边界包括：

1. 只评估到 6B 参数量级，没有验证当时最强模型的规模；
2. 分布外泛化只做了初步实验，缺少更广泛任务和偏好分布验证；
3. DPO 是否以及如何出现 reward over-optimization 仍不清楚；
4. GPT-4 胜率明显受评估提示影响；
5. 论文主要处理固定离线偏好数据，如何有效利用无标签提示和策略自生成标签仍待研究。

还要补充三个工程边界：

- DPO 省掉了独立奖励模型训练和 PPO rollout，但训练时通常仍要同时访问策略与参考策略；
- 偏好数据若有系统性偏差、位置偏差或错误标签，DPO 会直接学习这些偏差；
- 成对偏好只表达相对次序，不天然提供事实性、安全性或校准保证。

## 可复现资源与实现检查表

### 官方资源

- [NeurIPS 正式论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf)
- [NeurIPS 官方补充材料](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Supplemental-Conference.pdf)
- [作者官方实现](https://github.com/eric-mitchell/direct-preference-optimization)
- [Anthropic Helpful-Harmless 偏好数据](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [TL;DR SFT 参考模型](https://huggingface.co/CarperAI/openai_summarize_tldr_sft)

官方仓库给出的完整 Pythia-2.8B 示例使用 4 张 80 GB A100：SFT 约 1 小时 30 分，DPO 约 2 小时 45 分。这是特定硬件、代码版本和配置的记录，不是 DPO 的普遍成本。

### 实现检查表

1. 用相同 tokenizer 和截断规则计算策略、参考策略的回答 token log probability。
2. prompt token 不应混进回答损失；padding token 必须 mask。
3. $y_w$ 与 $y_l$ 要来自同一提示，数据管道不能打乱配对。
4. 冻结参考模型并关闭其梯度；显存紧张时可预计算参考 log probability。
5. 监控 chosen/rejected 的隐式奖励、reward margin、loss、梯度范数和验证集偏好准确率。
6. 同时做生成质量、事实性、安全性和人工抽检，不能只看训练 loss 或单一自动裁判胜率。
7. 记录 $\beta$、学习率、序列长度、batch size、训练步数和采样温度；这些量共同决定最终行为。

## 个人判断

我认为 DPO 最成功的地方，是找到一个足够准确又足够便宜的抽象：把“奖励最大化”变成“相对参考策略重新排序偏好对”。它没有要求工程团队先搭好一套稳定的在线 RL 系统，因此能让更多项目真正开始做偏好优化。

但“没有 PPO”不等于“没有强化学习中的难题”。参考策略决定零点，偏好数据决定目标，$\beta$ 决定保守程度，自动裁判决定我们看到的结果。数据支持集、奖励投机、分布外泛化和评估偏差只是从 PPO 管线转移到了更容易观察的位置，并没有消失。

如果目标是复现论文，先从小模型、固定公开数据和人工可检查任务开始，确认概率 mask、参考模型和 pair 顺序完全正确，再扩大规模。如果目标是产品对齐，优先投资偏好数据质量和多维评估，而不是把所有注意力放在换一种 DPO loss 变体上。

## 一句话总结

> DPO 利用 KL 约束最优策略与奖励之间的闭式关系，把成对偏好学习改写为策略相对参考策略的二分类损失；它显著简化了 RLHF 训练，但数据、参考模型、事实性和评估可靠性仍决定最终上限。

## 参考资料

1. Rafailov et al. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html). NeurIPS 2023.
2. Rafailov et al. [DPO Supplemental Material](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Supplemental-Conference.pdf). NeurIPS 2023.
3. Christiano et al. [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper_files/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf). NeurIPS 2017.
4. Stiennon et al. [Learning to Summarize from Human Feedback](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html). NeurIPS 2020.
5. Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). 2017.
6. Bai et al. [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862). 2022.
