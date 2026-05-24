---
title: MedDocOCR-Interpreter 源码导读：医疗文档 OCR、结构化抽取与报告解读原型
date: 2026-05-22 10:30:00
tags:
  - OCR
  - 多模态
  - 医疗 AI
  - RAG
  - Python
  - 源码分析
categories:
  - 技术
---

MedDocOCR-Interpreter 是一个面向医疗单据的 OCR 与报告解读原型项目。它的目标不是直接训练一个完整医疗大模型，而是先把“医疗文档从图片/文本到结构化结果，再到可解释解读”的工程链路搭起来：输入可以是检验报告、处方、发票、住院清单等文档，系统会经过 OCR、版面检测、表格结构恢复、字段归一化、规则知识库检索和安全解读，最后输出异常项、风险说明、复查建议、证据和安全提示。

这篇文章按源码结构拆解这个项目：它解决什么问题、流水线怎么串起来、各个模块分别负责什么、为什么适合作为医疗多模态项目的工程原型，以及后续如果要从原型升级到真实系统，应该补哪些能力。



## 1. 项目定位

从 README 看，这个项目的完整定位是：

> 医疗文档 OCR 与报告解读多模态系统原型。

它覆盖的链路可以概括为：

```text
medical document image/text
  -> PaddleOCR / OCR fallback
  -> LayoutDetector
  -> TableStructureRecognizer
  -> FieldNormalizer
  -> MedicalKnowledgeBase + ReportInterpreter
  -> abnormalities / risks / suggestions / evidence / safety notes
```

也就是说，它不是单点 OCR demo，而是一个“文档智能 + 医学规则解读”的端到端原型。核心价值在于把复杂医疗单据处理拆成了几个可替换的工程边界：OCR 可以从文本 fallback 换成 PaddleOCR，版面识别可以从规则启发式换成 LayoutLMv3，报告解读可以从 JSON 规则库升级成 RAG，生成式部分也预留了 Qwen2.5-VL、LLaVA、SFT、DPO、PPO 等训练路线。

如果用一句话概括源码架构：

> MedDocOCR-Interpreter 是一个 Python 实现的医疗文档结构化流水线：用 Pydantic 统一输入输出 schema，用可替换模块串起 OCR、版面、表格、字段和医学解读，再通过 CLI、FastAPI 和 Pytest 保证可运行、可集成、可验证。

## 2. 仓库目录总览

项目目录比较清晰，主代码都在 `meddococr_interpreter` 下：

```text
meddococr_interpreter/
  api/                 FastAPI 服务入口
  interpret/           医学知识库检索与报告解读
  layout/              版面识别模块
  normalize/           字段归一化模块
  ocr/                 OCR 适配器
  synthetic/           合成报告数据生成
  table/               表格结构恢复
  training/            训练 recipe 设计
  cli.py               命令行入口
  config.py            配置加载
  pipeline.py          主流水线编排
  schemas.py           统一数据结构
configs/default.yaml   默认配置
data/kb/medical_rules.json 医学规则知识库
docs/                  项目说明和面试材料
examples/              示例报告文本
tests/                 端到端测试
```

这个结构有两个特点：

1. **模块边界很清楚**：OCR、layout、table、normalize、interpret 都是独立目录，后续替换真实模型时不需要推翻整体工程。
2. **原型可直接运行**：即使没有安装 PaddleOCR，也可以用文本 fallback 跑通 pipeline 和测试，这对演示、面试和迭代非常友好。

## 3. 数据结构设计

`schemas.py` 是整个项目的类型中心，里面定义了文档类型、OCR token、版面块、表格 cell、关键字段、异常项、流水线输入输出等模型。

主要 schema 如下：

```text
DocumentType    文档类型：lab_report / prescription / invoice / inpatient_list / unknown
OCRToken        OCR 文本、bbox、置信度、行号
LayoutBlock     版面块标签、bbox、文本、置信度
TableCell       表格行列、文本、row_span、col_span
KeyField        字段名、值、单位、参考范围、状态、置信度
AbnormalItem    异常项、风险、建议、证据
PipelineInput   输入路径/原始文本/文档类型/上下文
PipelineOutput  OCR、版面、表格、字段、解读、安全提示
```

这里最重要的是 `PipelineOutput`。它不是只返回最终一句医学建议，而是保留了中间结果：OCR tokens、layout blocks、table cells、key fields 都会出现在输出里。这种设计很适合文档智能系统，因为医疗场景尤其需要可追溯性：如果最终解读有问题，可以回看是 OCR 错了、表格恢复错了、字段归一化错了，还是医学规则匹配错了。

## 4. 主流水线：pipeline.py

主入口是 `MedDocOCRPipeline`。它在初始化时组装五个模块：

```text
PaddleOCREngine
LayoutDetector
TableStructureRecognizer
FieldNormalizer
ReportInterpreter
```

运行逻辑非常直观：

```text
payload
  -> ocr.recognize()
  -> layout.detect()
  -> table.recover()
  -> normalizer.normalize()
  -> interpreter.interpret()
  -> PipelineOutput
```

这段代码虽然不复杂，但体现了一个很好的工程习惯：主流水线只做编排，不把具体 OCR、表格解析、医学规则写在一起。这样后续升级时可以逐步替换模块，例如：

- 把 `PaddleOCREngine` 换成云 OCR、PaddleOCR PP-Structure 或自研 VLM OCR。
- 把 `LayoutDetector` 换成 LayoutLMv3、DocLayout-YOLO 或视觉语言模型。
- 把 `MedicalKnowledgeBase` 从 JSON 规则升级为向量检索 + 医学知识库。
- 把 `ReportInterpreter` 从规则模板升级为带引用约束的 LLM 解读器。

## 5. OCR 模块：PaddleOCR + 文本 fallback

`ocr/paddle_engine.py` 实现了一个 PaddleOCR adapter，并提供了确定性的文本 fallback。

它支持两类输入：

1. 如果传入 `text`，直接按行切分成 `OCRToken`。
2. 如果传入 `.txt`、`.md`、`.csv` 文件，也按文本文件读取。
3. 如果传入图片，则尝试加载 `paddleocr.PaddleOCR`，输出文本、bbox 和 score。

这种 fallback 很实用。医疗 OCR 项目经常依赖较重的视觉库，如果每次 demo、单测都必须安装完整 OCR 环境，开发效率会很低。这里通过文本 fallback，可以把后续版面、表格、字段和解读模块先跑通；真正接图片时，再安装 `[ocr]` optional dependencies。

项目默认会过滤低置信度 OCR 结果，阈值来自配置中的 `min_ocr_score`，默认是 `0.3`。图片路径的 OCR 输出会统一封装成 `OCRToken`，从而保证下游模块不依赖 PaddleOCR 的原始返回格式。

## 6. 版面识别：规则版 LayoutDetector

`layout/layoutlmv3_detector.py` 当前是启发式实现，但命名上已经预留了 LayoutLMv3 替换边界。

它按关键词和分隔符把每一行分成几类：

```text
header        包含姓名、性别、年龄、报告、医院等
 table_header  包含项目、结果、参考、单位等
 table_row     包含 |、tab、逗号等表格分隔符
 footer        包含医师、审核、日期等
 text          其他普通文本
```

这种规则当然不能覆盖复杂扫描件，但对原型很有价值：它让系统具备了“版面块”这一层抽象。后续如果换成真正的 LayoutLMv3、Donut、DocFormer 或 VLM，只需要保证输出还是 `LayoutBlock`，后续表格和字段模块就能继续复用。

## 7. 表格结构恢复

`table/structure.py` 的 `TableStructureRecognizer` 负责把 OCR 行恢复成表格单元格。

它通过正则识别三类分隔符：

```text
|
tab
连续逗号
```

每一行被拆成若干列后，输出为：

```text
TableCell(row=行号, col=列号, text=单元格文本)
```

例如：

```text
项目 | 结果 | 单位 | 参考范围
白细胞 | 12.8 | 10^9/L | 3.5-9.5
```

会恢复出表头和数据行，每个 cell 都带有 row / col 信息。这一步虽然仍是规则实现，但已经抽出了表格结构层，为后续计算 TEDS、接入表格检测模型或输出 HTML/Markdown 表格打好了接口。

## 8. 字段归一化

`normalize/fields.py` 是项目里非常关键的一层。它把不同格式的 OCR 文本和表格 cell 统一变成 `KeyField`：

```text
name       指标名
value      指标值
unit       单位
reference  参考范围
status     normal / low / high / abnormal
confidence 置信度
```

字段抽取有两条路径：

1. **优先从表格抽取**：如果一行有“项目、结果、单位、参考范围”这样的结构，就把它转成字段。
2. **再用正则补充**：对普通文本行匹配“字段名 + 数值/阴性/阳性/正常/异常 + 单位 + 参考范围”。

异常判断逻辑也在这里完成：

- `阳性`、`异常` 会被标为 `abnormal`。
- `阴性`、`正常` 会被标为 `normal`。
- 如果有数值和参考范围，会比较上下界，得到 `low`、`high` 或 `normal`。

这层的意义是把格式各异的医疗报告变成统一字段。只要 `KeyField` 稳定，后面的医学解读就不需要关心原始报告到底是表格、冒号文本，还是 OCR 分行结果。

## 9. 医学知识库与安全解读

医学解读分为两个文件：

```text
interpret/rag.py       MedicalKnowledgeBase
interpret/report.py    ReportInterpreter
```

`MedicalKnowledgeBase` 当前读取的是 `data/kb/medical_rules.json`。规则库中包含白细胞、血红蛋白、C 反应蛋白、葡萄糖等指标，每个指标有三类内容：

```text
risk        可能风险
suggestion  复查/就诊建议
evidence    规则证据
```

检索逻辑是基于字段名的简单包含匹配。如果没有命中具体规则，就返回通用安全规则。

`ReportInterpreter` 只处理异常字段，也就是 `status` 为 `abnormal`、`low` 或 `high` 的字段。它会把字段值、单位和异常方向组合起来，再从知识库中取出风险说明、建议和证据。

另外它有一个很重要的安全保护：

```text
本系统仅用于报告整理和健康教育，不替代医生诊断。
涉及危急值、明显不适、孕产妇、儿童或慢病患者时应及时就医。
```

代码里还会过滤一些高风险表达，比如“确诊”“一定是”“无需就医”。如果建议里没有“医生”或“复查”，也会追加“建议结合临床情况咨询医生”。这体现了医疗 AI 项目最基本的安全边界：可以做信息整理和健康教育，但不能越界替代诊断。

## 10. CLI 与 API

项目提供了两个集成入口。

### 10.1 CLI

命令行入口在 `cli.py`，可以直接运行示例报告：

```bash
python -m meddococr_interpreter.cli examples/lab_report.txt --document-type lab_report --age 45 --sex 男
```

也可以把结果写到 JSON 文件：

```bash
python -m meddococr_interpreter.cli examples/lab_report.txt --output outputs/sample.json
```

CLI 支持 `--raw-text`，因此可以直接把一段 OCR 文本作为输入，这对调试字段抽取和解读规则很方便。

### 10.2 FastAPI

API 入口在 `api/server.py`，提供两个接口：

```text
GET  /health
POST /interpret
```

启动方式：

```bash
uvicorn meddococr_interpreter.api.server:app --reload
```

`POST /interpret` 的请求体就是 `PipelineInput`，响应体就是 `PipelineOutput`。这意味着 CLI、测试和 Web API 共用同一套核心流水线，不会出现多套逻辑不一致的问题。

## 11. 配置与依赖设计

`pyproject.toml` 把依赖分成了几组：

```text
基础依赖：pydantic, pyyaml
api：fastapi, uvicorn, python-multipart
ocr：paddleocr, pillow
training：torch, transformers, datasets, trl
dev：pytest, ruff
```

这种 optional dependencies 的设计比较合理：如果只是跑文本 demo 和测试，不需要安装庞大的 OCR、训练和服务依赖；如果要部署 API，再安装 `[api]`；如果要接图片 OCR，再安装 `[ocr]`；如果要做训练实验，再安装 `[training]`。

默认配置在 `configs/default.yaml`：

```yaml
ocr_engine: auto
layout_engine: heuristic
vlm_model: Qwen/Qwen2.5-VL-7B-Instruct
rag_kb_path: data/kb/medical_rules.json
min_ocr_score: 0.3
enable_safety_guard: true
```

配置项里已经出现了 `vlm_model` 和 `layout_engine`，说明这个项目当前虽然是轻量规则原型，但设计目标是能逐步迁移到真实多模态模型。

## 12. 合成数据与训练路线

`synthetic/generator.py` 可以随机生成检验报告文本，包括医院、姓名、性别、年龄、白细胞、血红蛋白、C 反应蛋白等字段，并转成 VLM SFT messages 格式。

这说明项目并不只考虑推理链路，也在为训练数据准备做铺垫。一个更完整的扩展方向是：

```text
模板渲染
  -> 表格线、字体、印章、噪声、模糊、透视变换
  -> 自动生成 OCR bbox / 表格 cell / key field 标签
  -> 构造 VLM SFT JSONL
```

`training/recipes.py` 给出了四阶段训练设想：

| 阶段 | 目标 | 指标 |
| --- | --- | --- |
| OCR continual pretrain | 在医疗文档上继续训练 OCR/VLM 能力 | CER、layout block F1 |
| Document QA SFT | 做字段抽取、表格问答、报告问答监督微调 | field F1、TEDS、QA accuracy |
| Preference DPO | 偏好更安全、更可追溯的解读 | safety win rate、refusal correctness |
| Rule reward RL | 优化格式合法性、证据覆盖和医学安全 | format pass rate、medical advice safety |

这个路线很适合作为项目答辩或面试时的技术叙事：当前仓库是可运行原型，真实训练需要数据、算力和标注闭环，但工程接口已经预留好了。

## 13. 测试覆盖

`tests/test_pipeline.py` 是一个端到端测试。它构造了一段检验报告文本：

```text
项目 | 结果 | 单位 | 参考范围
白细胞 | 12.8 | 10^9/L | 3.5-9.5
血红蛋白 | 118 | g/L | 130-175
C反应蛋白 | 35.6 | mg/L | 0-10
```

测试会验证三件事：

1. OCR token 数量足够。
2. 能抽取出白细胞、血红蛋白、C 反应蛋白三个关键字段。
3. 能对这三个异常字段产生解读，并返回安全提示。

虽然测试数量不多，但它覆盖了项目最核心的主链路：文本输入、表格恢复、字段抽取、异常判断、知识库解释和安全提示。

## 14. 项目亮点

我认为这个项目比较好的地方有四点。

第一，**工程边界清晰**。每个模块都有明确职责，主流水线只做编排，后续接真实模型时替换成本较低。

第二，**类型结构完整**。通过 Pydantic 把中间结果和最终结果都结构化，方便 API 返回、测试断言、前端展示和错误排查。

第三，**安全意识明确**。医疗 AI 最怕直接给诊断结论，这个项目在规则层和输出层都强调“健康教育，不替代医生诊断”。

第四，**原型可运行**。文本 fallback、CLI、FastAPI、Pytest 都已经具备，不是只停留在架构图，而是能跑出结构化结果。

## 15. 当前局限

作为原型，它也有明显限制：

1. **OCR 和版面识别仍偏 demo**：真实扫描件会有旋转、阴影、低清晰度、表格线断裂、多栏布局等问题，当前规则很难覆盖。
2. **表格恢复比较简单**：只处理分隔符文本，无法处理复杂合并单元格、跨页表格、无框线表格和错行 OCR。
3. **医学知识库很小**：当前规则只覆盖少量常见指标，且没有分年龄、性别、孕产、儿童等医学参考范围差异。
4. **缺少证据引用闭环**：输出中有 evidence 字段，但还不是严格的知识库引用、指南引用或原文 span 引用。
5. **没有真实评测集**：缺少真实医疗票据/报告上的 OCR、字段抽取、表格恢复和安全解读 benchmark。

这些限制不影响它作为原型的价值，但如果要做生产级系统，必须重点补齐。

## 16. 后续优化方向

如果继续完善，我会按下面顺序迭代：

1. **增强输入层**：支持 PDF、多页图片、拍照件自动纠偏、图像增强和方向检测。
2. **升级版面模型**：接入 LayoutLMv3、DocLayout-YOLO 或 VLM，把规则版 layout detector 替换成模型推理。
3. **改进表格恢复**：增加 cell bbox、reading order、跨行跨列合并和 HTML 表格输出，并用 TEDS 评测。
4. **扩展医学知识库**：引入更系统的指标规则、参考范围、危急值规则和人群差异规则。
5. **加强可追溯性**：让每个字段和解读都能回链到 OCR token、表格 cell、原始 bbox 和知识库来源。
6. **建立评测闭环**：分别评测 OCR CER、字段 F1、表格 TEDS、解读安全率和 JSON schema 通过率。

## 17. 总结

MedDocOCR-Interpreter 是一个很适合作为医疗多模态工程项目的原型：它没有试图一上来就“端到端大模型解决一切”，而是把医疗文档处理拆成 OCR、版面、表格、字段、知识库和安全解读几个阶段，每个阶段都有清晰输入输出，也都预留了从规则实现升级到模型实现的空间。

从学习和展示角度看，这个项目可以体现三种能力：

1. **文档智能工程能力**：知道 OCR、layout、table、field extraction 怎么串成可运行系统。
2. **医疗 AI 安全意识**：知道报告解读必须有边界、证据和安全提示。
3. **多模态训练规划能力**：知道如何从可运行原型进一步走向 VLM SFT、偏好优化和规则奖励训练。

如果要在面试中介绍这个项目，可以这样总结：

> 我实现了一个医疗文档 OCR 到报告解读的端到端原型。系统支持图片或文本输入，先通过 PaddleOCR adapter 得到 OCR token，再进行版面识别、表格恢复和字段归一化，把不同格式的检验指标统一成结构化字段。对于异常字段，系统会检索医学规则知识库，输出风险说明、复查建议、证据和安全提示，并通过规则避免过度诊断。工程上使用 Pydantic 保证 schema 一致，用 FastAPI 和 CLI 提供集成入口，用 Pytest 覆盖主链路，同时预留了 Qwen2.5-VL、LayoutLMv3、SFT、DPO 和规则奖励 RL 的升级路线。
