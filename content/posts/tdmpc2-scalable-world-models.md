---
title: "TD-MPC2 精读：用隐式世界模型统一 104 个连续控制任务"
date: 2026-07-22 09:15:00
description: 从隐式潜在动力学、SimNorm、离散回归、策略先验与 MPPI 规划出发，拆解 ICLR 2024 Spotlight 论文 TD-MPC2 如何用统一超参数稳定训练并扩展到 317M 参数、80 个任务。
series: 三大会论文精读
seriesOrder: 7
categories:
  - AI
tags:
  - 强化学习
  - 世界模型
  - 模型预测控制
  - 连续控制
  - 多任务学习
  - ICLR
hidden: true
haloPublished: true
---

强化学习论文常在一组熟悉任务上为每个环境单独调参，最后得到一串漂亮曲线。真正困难的问题是：当奖励尺度、动作维度、任务长度、动力学和探索难度都变化时，能否仍用同一套算法与超参数稳定工作？

TD-MPC2 研究的正是这个工程问题。它不尝试重建未来图像，而是在潜在空间中只预测控制所需的下一状态表示、奖励和价值，再用模型预测控制在线搜索动作。论文通过 SimNorm、离散回归、Q 函数集成、最大熵策略先验、任务嵌入和动作掩码，把原 TD-MPC 扩展成可跨 104 个连续控制任务复用、可训练到 317M 参数的世界模型。

## 论文信息卡

| 项目 | 信息 |
| --- | --- |
| 论文 | **TD-MPC2: Scalable, Robust World Models for Continuous Control** |
| 作者 | Nicklas Hansen、Hao Su、Xiaolong Wang |
| 会议 | ICLR 2024，Spotlight |
| 专题子方向 | 模型式强化学习、隐式世界模型、多任务连续控制 |
| 正式评审与最终版 | [OpenReview: Oxh5CstDJU](https://openreview.net/forum?id=Oxh5CstDJU) |
| 作者版本与许可 | [arXiv:2310.16828 v2](https://arxiv.org/abs/2310.16828)，CC BY 4.0 |
| 项目主页 | [tdmpc2.com](https://tdmpc2.com) |
| 官方实现 | [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2)，MIT License |

**选择理由**：此前专题连续覆盖了多模态与推理系统，本篇切换到强化学习。TD-MPC2 的贡献不是只在单项基准上提分，而是把稳定训练、统一超参数、多任务数据和模型扩展放在同一个设计中验证。31 页作者最终版包含完整公式、实现细节、104 项任务结果、消融、训练成本和风险讨论；论文、源码、代码、模型与数据均公开，关键事实可以交叉核验。

## 问题背景：世界模型为什么仍然难以复用

连续控制可以写成马尔可夫决策过程 $(\mathcal S,\mathcal A,\mathcal T,R,\gamma)$。策略希望最大化折扣回报：

$$
\mathbb E_{\pi}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right].
$$

模型式强化学习先学习环境动力学，再用模型预测候选动作的后果。模型预测控制（MPC）在每个时刻优化长度为 $H$ 的动作序列：

$$
\pi(\mathbf s_t)=\arg\max_{\mathbf a_{t:t+H}}
\mathbb E\left[\sum_{i=0}^{H}\gamma^{t+i}
R(\mathbf s_{t+i},\mathbf a_{t+i})\right].
$$

这条路线有两个现实矛盾：

1. 重建未来观测提供了密集监督，但像素和高维状态中有大量与控制无关的信息，长期预测又容易累积误差。
2. 只预测短期奖励会让规划视野过窄；扩大规划深度则会快速增加采样成本和模型误差。

TD-MPC 的思路是学习一个不带解码器的控制中心潜在模型，并在短规划末端用价值函数补上长期回报。TD-MPC2 沿用这条主线，但重点修复原算法在奖励尺度、高维动作、梯度稳定性和多任务扩展上的脆弱环节。

## 核心贡献

论文的贡献可以分成四层：

1. **更稳健的隐式世界模型**：用 SimNorm 限制潜在表示，用 LayerNorm + Mish 改造网络，以离散回归预测奖励和价值，并用 5 个 Q 函数降低目标偏差。
2. **规划与策略先验协作**：MPPI 负责在线局部搜索，最大熵策略先验提供高质量候选并学习可直接执行的动作分布。
3. **跨任务统一接口**：所有模块都条件化于可学习任务嵌入；不同观测和动作维度通过补零与动作掩码统一，不依赖手写域知识。
4. **规模化实证**：单任务覆盖 104 个任务；离线多任务训练覆盖 80 个任务、545M 条转移，模型从 1M 扩展到 317M 参数。

**论文结论**：在论文选择的连续控制任务、数据预算和基线实现下，TD-MPC2 用同一套超参数取得更高的整体数据效率和最终性能；多任务能力随模型规模增长，并可通过预训练提高新任务低数据阶段的学习速度。

**我的判断**：这篇论文最重要的结果不是“317M”这个数字，而是它把强化学习扩展失败的原因拆成一组可测试的数值与接口问题。它证明了世界模型可以吸收混合质量的多任务回放，但尚未证明同一方法能跨到离散动作、真实机器人或无任务 ID 的开放环境。

## 方法总览：只预测控制真正需要的量

![TD-MPC2 隐式世界模型架构](/images/posts/tdmpc2-scalable-world-models/figure3-architecture.png)

*图源：Hansen et al., [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://openreview.net/forum?id=Oxh5CstDJU), Figure 3, ICLR 2024；从作者 [CC BY 4.0 arXiv v2 源码](https://arxiv.org/abs/2310.16828)中的原始矢量图直接栅格化，未修改结构与标签。原图用于论文解读。*

TD-MPC2 包含五个模块：

$$
\begin{aligned}
\mathbf z &= h(\mathbf s,\mathbf e) && \text{编码器} \\
\mathbf z' &= d(\mathbf z,\mathbf a,\mathbf e) && \text{潜在动力学} \\
\hat r &= R(\mathbf z,\mathbf a,\mathbf e) && \text{奖励预测} \\
\hat q &= Q(\mathbf z,\mathbf a,\mathbf e) && \text{终端价值} \\
\hat{\mathbf a} &= p(\mathbf z,\mathbf e) && \text{策略先验}
\end{aligned}
$$

$\mathbf e$ 是多任务场景中的可学习任务嵌入。模型不会把 $\mathbf z'$ 解码回原始观测，而是让它接近下一时刻真实观测的编码，并同时学会预测奖励与长期价值。这样做主动放弃“生成得像”，换取“对决策有用”。

### 世界模型目标

对回放缓冲区中长度为 $H$ 的片段，编码器、动力学、奖励头和价值头共同最小化：

$$
\mathcal L(\theta)=
\mathbb E\left[
\sum_{t=0}^{H}\lambda^t
\left(
\|\mathbf z'_t-\operatorname{sg}(h(\mathbf s'_t))\|_2^2
+\operatorname{CE}(\hat r_t,r_t)
+\operatorname{CE}(\hat q_t,q_t)
\right)
\right],
$$

其中 TD 目标为：

$$
q_t=r_t+\gamma\bar Q(\mathbf z'_t,p(\mathbf z'_t)).
$$

$\bar Q$ 是 Q 网络的指数移动平均。联合嵌入项通过 `stop-grad` 提供下一状态表征目标；奖励与价值不再直接做标量均方误差，而是先映射到对数空间的 101 个桶，再以软标签交叉熵训练。

这一步对多任务非常关键。若一个环境奖励约为 $10^{-2}$，另一个约为 $10^3$，普通回归损失会被大尺度任务主导；离散回归把不同奖励尺度映射到统一预测接口，降低了损失量级对训练的影响。

### SimNorm：让潜在状态保持可控

TD-MPC2 把潜在向量切成 $L$ 个长度为 $V$ 的分组，每组单独做 softmax：

$$
\mathbf z^{\circ}=[\mathbf g_1,\ldots,\mathbf g_L],\qquad
\mathbf g_i=
\frac{\exp(\mathbf z_{i:i+V}/\tau)}
{\sum_{j=1}^{V}\exp(\mathbf z_{i:i+V,j}/\tau)}.
$$

论文默认 $V=8,\tau=1$。每组元素和为 1，使潜在状态的范数和尺度受到结构性约束，同时保留连续梯度。作者把它解释为“软的 vector-of-categoricals”：比完全离散编码平滑，又比无约束连续向量更不容易出现梯度爆炸。

### Q 集成与策略先验

论文默认训练 5 个 Q 函数，并对每个 Q 网络使用 1% dropout。TD 目标每次随机抽取两个 Q 函数取最小值，以减轻过估计；用于目标的网络仍采用 EMA 更新。

策略先验 $p$ 使用最大熵目标：

$$
\mathcal L_p=
\mathbb E\left[
\sum_{t=0}^{H}\lambda^t
\left(\alpha Q(\mathbf z_t,p(\mathbf z_t))
-\beta\mathcal H(p(\cdot|\mathbf z_t))\right)
\right].
$$

它不是最终决策的唯一来源。规划器会混合策略先验给出的候选和高斯采样候选，策略先验负责缩小搜索范围，规划器负责在当前状态下进一步优化。

## 规划流程：短视野搜索加长期价值

TD-MPC2 使用 MPPI 在潜在空间中优化时变高斯分布的均值与方差：

$$
\mu^*,\sigma^*=\arg\max_{\mu,\sigma}
\mathbb E_{\mathbf a_{t:t+H}\sim\mathcal N(\mu,\sigma^2)}
\left[
\sum_{h=t}^{H-1}\gamma^h R(\mathbf z_h,\mathbf a_h)
+\gamma^H Q(\mathbf z_{t+H},\mathbf a_{t+H})
\right].
$$

默认配置看起来很短：规划视野 $H=3$、6 次迭代、每次 512 条候选、64 个精英样本，并加入 24 条策略先验轨迹。但终端 Q 值估计了 $H$ 之后的长期回报，因此这不是只看 3 步的贪心控制。

每个环境步的执行顺序是：

1. 编码当前观测和任务嵌入。
2. 用上一时刻规划结果平移后初始化动作分布。
3. 从高斯分布和策略先验采样候选动作序列。
4. 在潜在动力学中展开候选，累计奖励并加终端 Q 值。
5. 用高回报样本更新 $\mu,\sigma$，重复若干轮。
6. 只执行第一步动作，把新转移写入回放缓冲区。
7. 从回放中均匀采样片段，更新世界模型与策略先验。

论文移除了原 TD-MPC 的优先经验回放和 MPPI 动量，并报告通过 Q 集成向量化等实现优化把规划吞吐提高约 2 倍。这些变化让 5M 参数的默认模型在墙钟时间上仍与约 1M 参数的原版接近。

## 多任务接口：统一维度，但保留任务身份

不同控制域的观测和动作维度差异很大，论文中的动作空间最高达到 $\mathbb R^{39}$。TD-MPC2 把输入与输出补零到批次中的最大维度，并在训练、策略熵计算和规划时屏蔽无效动作维度。否则，小动作空间任务会通过无效维度虚增策略熵，也会把无意义的预测误差传入 TD 目标。

所有五个模块都接收一个 96 维可学习任务嵌入 $\mathbf e$，并把其 $\ell_2$ 范数限制在 1 以内。这个设计能学习任务关系，但它也意味着测试时必须提供任务 ID；论文并没有从图像或自然语言自动推断“当前要做什么”。

## 实验设置

单任务在线强化学习覆盖 104 个任务和 4 个模拟域：

| 任务域 | 数量 | 主要指标 | 特点 |
| --- | ---: | --- | --- |
| DMControl | 39 | episode return | 含 Humanoid、Dog 等高维运动任务 |
| Meta-World | 50 | 最后一步 success | 机械臂多任务操作 |
| ManiSkill2 | 5 | 最后一步 success | 高随机化物体操作，Pick YCB 含 74 个物体 |
| MyoSuite | 10 | 最后一步 success | 39 维肌骨手部控制 |

此外，论文在 10 个图像输入的 DMControl 任务上验证视觉强化学习。主实验通常使用 5M 参数 TD-MPC2、batch size 256、update-to-data ratio 1；对比 SAC、DreamerV3、原 TD-MPC，以及视觉任务上的 CURL 和 DrQ-v2。曲线均报告 3 个随机种子的均值和 95% 置信区间。

多任务实验使用 545M 条 80 任务转移，以及 345M 条 30 任务转移。数据来自 240 个单任务 TD-MPC2 智能体的回放缓冲区，覆盖从随机到专家的多种行为。多任务模型使用 batch size 1024，其他核心超参数保持不变。

## 主要结果：统一超参数比单点最优更重要

![TD-MPC2 在六组连续控制基准上的汇总结果](/images/posts/tdmpc2-scalable-world-models/figure1-single-task-summary.png)

*图源：Hansen et al., [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://openreview.net/forum?id=Oxh5CstDJU), Figure 1 (right), ICLR 2024；从作者 [CC BY 4.0 arXiv v2 源码](https://arxiv.org/abs/2310.16828)中的原始矢量子图直接栅格化，坐标轴、任务数量和方法标签均保留。原图用于论文解读。*

Figure 1 汇总了 104 项单任务结果。TD-MPC2 在 DMControl、Meta-World、ManiSkill2、Locomotion、MyoSuite 和 Pick YCB 六组汇总中都高于论文实现的 SAC、DreamerV3 与 TD-MPC。最明显的差距出现在高维连续动作和细粒度物体操作：Pick YCB 中其他方法在给定预算内几乎没有学会，而 TD-MPC2 的最终成功率约为 70%；附录给出的严格表述是“超过 60%”。

这里需要避免两个过度结论：

- 结果证明的是论文所选预算、实现和任务上的连续控制鲁棒性，不代表 TD-MPC2 普遍优于 DreamerV3；作者明确承认 DreamerV3 在 Atari、Minecraft 等离散动作任务上更强。
- “同一套超参数”不等于完全没有规则。折扣因子和随机探索步数会根据预计 episode 长度自动计算；高维动作任务的规划迭代也从 6 增加到 8。

## 扩展结果：参数变大后能力是否增长

![TD-MPC2 在 80 个任务上的模型规模曲线](/images/posts/tdmpc2-scalable-world-models/figure7-scaling.png)

*图源：Hansen et al., [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://openreview.net/forum?id=Oxh5CstDJU), Figure 7 (left, 80-task panel), ICLR 2024；从作者 [CC BY 4.0 arXiv v2 源码](https://arxiv.org/abs/2310.16828)中的原始矢量子图直接栅格化，横轴、纵轴、图例和数据标注均保留。原图用于论文解读。*

在 80 任务数据上，TD-MPC2 的归一化分数随模型规模变化如下：

| 参数量 | 归一化分数 | 单张 RTX 3090 近似训练成本 |
| ---: | ---: | ---: |
| 1M | 16.0 | 3.7 GPU days |
| 5M | 49.5 | 4.2 GPU days |
| 19M | 57.1 | 5.3 GPU days |
| 48M | 68.0 | 12 GPU days |
| 317M | **70.6** | 33 GPU days |

原 TD-MPC 随模型增大反而从约 20 分降到接近 6 分；TD-MPC2 则持续上升。作者观察到分数与参数量对数近似线性，但明确没有据此拟合 scaling law。48M 到 317M 只增加 2.6 分，而训练成本从 12 增至 33 GPU days，也说明“继续做大”已经出现明显边际收益下降。

任务嵌入的 t-SNE 可视化显示，Door Open/Close 等动力学相近任务会靠近。论文认为嵌入更偏向聚类共同动力学，而不只是共同目标。这个结果是定性证据，不足以证明嵌入空间能可靠支持组合泛化或零样本任务识别。

## 少样本迁移

作者先在 70 个任务上训练 19M 参数模型，再对 10 个留出任务做在线微调。微调时回放缓冲区从空开始，完整模型参与更新，任务嵌入由一个语义相似的已见任务初始化。

在 20k 环境步时，预训练模型的平均归一化分数为 47.0，从头训练为 24.0，约提升 2 倍。该结果说明多任务世界模型能加速低数据阶段，但附录也指出收益强烈依赖目标任务；作者没有比较随机任务嵌入、冻结主干或更系统的相似任务选择，因此还不能确定增益分别来自共享动力学、初始化策略还是人工任务配对。

## 消融分析：稳定性来自组合，而非单个技巧

![TD-MPC2 核心设计消融](/images/posts/tdmpc2-scalable-world-models/figure9-ablations.png)

*图源：Hansen et al., [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://openreview.net/forum?id=Oxh5CstDJU), Figure 9, ICLR 2024；从作者 [CC BY 4.0 arXiv v2 PDF](https://arxiv.org/abs/2310.16828)高分辨率裁取完整八个子图，保留坐标轴、置信区间、图例和数值标签，未改动实验数据。原图用于论文解读。*

论文在 Dog Run、Humanoid Walk、Pick YCB 三个困难任务，以及 80 任务多任务训练上做了对应消融：

| 设计轴 | 多任务消融结果 | 论文支持的结论 |
| --- | ---: | --- |
| 只用策略 / 只规划 / 规划 + 策略先验 | 42.2 / 53.7 / **54.2** | 规划贡献最大，策略先验带来小幅额外收益 |
| 无归一化 / SimNorm / LayerNorm + SimNorm | 46.8 / 51.0 / **54.2** | 两种归一化都重要，组合最好 |
| 连续回归 / 离散回归 | 49.6 / **54.2** | 离散回归提高跨奖励尺度鲁棒性 |
| 2 / 5 / 10 个 Q 函数 | 53.5 / 54.2 / **57.0** | 更大集成有收益，但默认 5 个折中计算成本 |

附录的梯度范数图进一步显示，原 TD-MPC 在 Walker 等任务上会出现梯度爆炸，而 TD-MPC2 在同一训练阶段保持稳定。Mish 与 ELU 的最终性能接近，但 Mish 的梯度更平滑；这说明不能把提升简单归因于更换激活函数。

## 失败案例与局限

论文没有用单独章节展示逐轨迹失败截图，但正文、附录和风险讨论给出了清楚的能力边界：

1. **只支持连续动作**：MPPI 的高斯动作搜索针对连续空间设计。作者建议未来研究 MCTS 等离散规划器，但论文没有给出可用方案。
2. **仍依赖已知任务身份**：多任务模型需要任务 ID 查找可学习嵌入，不能从自然语言目标或环境状态自动识别任务。
3. **全部是模拟环境**：104 个任务都来自 DMControl、Meta-World、ManiSkill2、MyoSuite，没有真实机器人部署与 sim-to-real 结果。
4. **数据不是无成本产生**：545M/345M 转移来自 240 个已训练单任务智能体。论文展示了如何利用混合质量回放，但没有消除前期交互与训练成本。
5. **离线分布外动作仍有风险**：主多任务实验没有训练期保守正则。附录的测试时不确定性惩罚把 19M 模型从 56.54 提到 62.01（$c=0.01$），但 $c=0.1$ 又降到 44.13，说明保守强度仍敏感。
6. **扩展证据范围有限**：只测试到 317M 参数、两个固定多任务数据集，且 48M 到 317M 的收益趋缓，不能据此外推更大规模。
7. **奖励指定与物理安全**：作者明确指出错误奖励可能诱导意外行为；真实机器人若没有额外安全检查，规划错误可能造成灾难性后果。
8. **资源集中风险**：大规模交互数据对小团队可能过于昂贵，可能进一步把通用具身模型能力集中到少数机构。

**我的补充判断**：TD-MPC2 的“统一超参数”是一项扎实的工程成果，但任务域仍高度结构化，观测与动作都能预先对齐，奖励也已定义。开放世界机器人还要解决任务发现、语言条件、约束规划、传感器缺失和安全恢复，这些并不由稳定的世界模型训练自动解决。

## 可复现资源

- [OpenReview 正式评审与 ICLR 2024 最终版](https://openreview.net/forum?id=Oxh5CstDJU)
- [arXiv v2 全文、源码与 CC BY 4.0 许可](https://arxiv.org/abs/2310.16828)
- [官方项目主页、视频与结果](https://tdmpc2.com)
- [MIT 许可官方代码](https://github.com/nicklashansen/tdmpc2)
- [324 个模型 checkpoint](https://www.tdmpc2.com/models)
- [545M/345M 转移的多任务数据集](https://www.tdmpc2.com/dataset)

官方仓库提供 Conda 环境、训练和评估入口。默认单任务示例围绕 `train.py`，多任务训练使用项目提供的数据配置；项目页建议复现多任务时先从 48M checkpoint 入手，它在性能和计算成本之间比 317M 更均衡。

复现时至少应核对以下细节：

1. 对奖励和价值使用论文定义的对数变换、101 桶软标签离散回归。
2. SimNorm 按长度 8 分组，不能对整个潜在向量只做一次 softmax。
3. Q 目标从集成中随机抽两个取最小值，目标网络使用 EMA。
4. 无效动作维度在策略输出、熵、训练损失和 MPPI 采样中都必须屏蔽。
5. 成功率按 episode 最后一步判断，这比“任意时刻成功过”更严格。
6. 报告环境步数、action repeat、UTD、规划候选数和随机种子，不能只给墙钟时间。

## 个人判断

TD-MPC2 展示了一种很有工程吸引力的世界模型路线：不承担像素重建的全部复杂度，只学习控制所需的潜在动力学、奖励与价值；在线决策也不把所有希望寄托在一个策略网络上，而是用策略提供先验、用短视野规划做状态相关修正。

它的成功更像一次“数值系统工程”胜利。SimNorm 控制表征尺度，离散回归吸收奖励量级差异，Q 集成处理目标偏差，动作掩码消除跨域接口噪声，自动启发式减少手工调参。每个改动单看都不神秘，组合后才让模型规模和任务数量真正可以上升。

对工程团队而言，最值得复用的不是直接把模型堆到 317M，而是先建立稳定性检查表：潜在状态范数、梯度范数、各任务损失占比、Q 集成分歧、规划候选回报、无效动作泄漏和不同奖励尺度下的校准。只有这些量保持可控，增加参数和数据才可能带来能力，而不是更昂贵的发散。

一句话概括：TD-MPC2 证明了一个不重建观测的隐式世界模型，可以通过规划、价值学习与一组针对数值稳定性的设计，在 104 个连续控制任务上共享超参数，并扩展到 80 任务、317M 参数；但它距离开放世界、离散动作和真实机器人通用控制仍有明显距离。

## 参考资料

1. Hansen et al. [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://openreview.net/forum?id=Oxh5CstDJU). ICLR 2024 Spotlight.
2. Hansen et al. [TD-MPC2, arXiv:2310.16828 v2](https://arxiv.org/abs/2310.16828). CC BY 4.0.
3. Hansen et al. [Temporal Difference Learning for Model Predictive Control](https://proceedings.mlr.press/v162/hansen22a.html). ICML 2022.
4. Hafner et al. [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104). 2023.
5. Haarnoja et al. [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905). 2018.
6. Williams et al. [Information-Theoretic Model Predictive Control](https://ieeexplore.ieee.org/document/7487277). 2017.
