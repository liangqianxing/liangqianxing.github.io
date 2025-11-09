---
title: GGML study note
date: 2025-11-09 19:35:16
categories:
  - 技术栈
tags:
  - GGML
mathjax: true
---

# GGML 学习笔记大纲

## 1. 矩阵乘法基础

矩阵乘法通常写作 $\mathbf{C} = \mathbf{A}\mathbf{B}$。只有当左矩阵 $\mathbf{A}$ 的列数与右矩阵 $\mathbf{B}$ 的行数相同，乘积矩阵 $\mathbf{C}$ 才有定义。其元素由下式确定：

$$
c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}.
$$

### 1.1 $2 \times 2$ 形式

$$
\begin{aligned}
\begin{bmatrix}
a_{11} & a_{12} \\[4pt]
a_{21} & a_{22}
\end{bmatrix}
\begin{bmatrix}
b_{11} & b_{12} \\[4pt]
b_{21} & b_{22}
\end{bmatrix}
&=
\begin{bmatrix}
a_{11}b_{11}+a_{12}b_{21} & a_{11}b_{12}+a_{12}b_{22} \\[4pt]
a_{21}b_{11}+a_{22}b_{21} & a_{21}b_{12}+a_{22}b_{22}
\end{bmatrix}.
\end{aligned}
$$

### 1.2 一般公式

$$
\begin{aligned}
\mathbf{C}_{m \times p} &= \mathbf{A}_{m \times n}\mathbf{B}_{n \times p}, \\
c_{ij} &= \sum_{k=1}^{n} a_{ik} b_{kj}.
\end{aligned}
$$

### 1.3 数值示例

$$
\begin{aligned}
\begin{bmatrix}
1 & 0 & 2 \\
-1 & 3 & 1
\end{bmatrix}_{2 \times 3}
\begin{bmatrix}
3 & 1 \\
2 & 1 \\
1 & 0
\end{bmatrix}_{3 \times 2}
&=
\begin{bmatrix}
1\cdot3+0\cdot2+2\cdot1 & 1\cdot1+0\cdot1+2\cdot0 \\
(-1)\cdot3+3\cdot2+1\cdot1 & (-1)\cdot1+3\cdot1+1\cdot0
\end{bmatrix}_{2 \times 2} \\
&=
\begin{bmatrix}
5 & 1 \\
4 & 2
\end{bmatrix}.
\end{aligned}
$$

## 2. ggml_tensor 结构

```C++
struct ggml_tensor {
        enum ggml_type type;
        struct ggml_backend_buffer * buffer;
        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_op op;
        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
        int32_t flags;
        struct ggml_tensor * src[GGML_MAX_SRC];
        // source tensor and offset for views
        struct ggml_tensor * view_src;
        size_t               view_offs;
        void * data;
        char name[GGML_MAX_NAME];
        void * extra; // extra things e.g. for ggml-cuda.cu
        char padding[8];
    };
```

要点摘要：
- `ne`（number of elements）与 `nb`（number of bytes）分别描述各维度的元素数量及字节跨度。
- `op` 和 `op_params` 指明该张量对应的运算节点及其参数，用于构建计算图。
- `view_src` 与 `view_offs` 允许视图张量共享底层数据，常用于切片、reshape 等操作。

## 3. 常用命令

```bash
# GPT-2 单路推理
.\build\bin\Release\gpt-2.exe -m .\models\gpt-2-117M\ggml-model.bin -p "This is an example" -n 128 -t 8 --top_k 40 --top_p 0.9 --temp 0.8

# GPT-2 带批次生成
.\build\bin\Release\gpt-2-batched.exe -np 4 -m .\models\gpt-2-117M\ggml-model.bin -p "Hello my name is" -n 64

# GPT-2 内存分配（alloc 版本）
.\build\bin\Release\gpt-2-alloc.exe -m .\models\gpt-2-117M\ggml-model.bin -p "Sample prompt" -n 80

# GPT-J 推理
.\build\bin\Release\gpt-j.exe -m .\models\gpt-j-6B\ggml-model.bin -p "int main(int argc, char ** argv) {" -n 200 -t 8

# 模型量化示例（F16 -> Q4_0）
.\build\bin\Release\gpt-2-quantize.exe .\models\gpt-2-1558M\ggml-model-f16.bin .\models\gpt-2-1558M\ggml-model-q4_0.bin 2

# SAM 图像分割
.\build\bin\Release\sam.exe -i .\examples\sam\example.jpg -m .\examples\sam\ggml-model-f16.bin -t 8

# YOLOv3-tiny 目标检测
.\build\bin\Release\yolov3-tiny.exe -m .\examples\yolo\yolov3-tiny.gguf -i .\examples\yolo\dog.jpg
```

## 4. metadata 速查

| 字段名 | 含义示例 |
| ------ | -------- |
| `shape` | 矩阵维度，如 `(3, 4)` 表示 3 行 4 列 |
| `dtype` | 元素类型，例如 `float64`、`int32` |
| `nnz` | 稀疏矩阵中非零元素（number of non-zero entries） |
