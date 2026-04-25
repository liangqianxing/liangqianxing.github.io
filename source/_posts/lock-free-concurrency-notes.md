---
title: 无锁并发入门：从 CAS 到 Atomic Ring Buffer
date: 2026-04-25
categories: 技术
tags:
  - C++
  - 并发
  - 无锁编程
  - 性能优化
  - 量化开发
---

这篇文章整理几个无锁并发里经常一起出现的概念：CAS、busy polling、atomic ring buffer、release/acquire、store buffer 和 CPU cache warmup。它们看起来分散，其实是一条完整链路：**硬件提供原子操作，程序用原子变量做同步，线程用轮询降低延迟，最后还要理解 cache 和内存模型带来的性能与可见性问题**。

<!-- more -->

## CAS：无锁编程的基本原子操作

CAS 是 Compare-And-Swap 的缩写，可以理解成一个硬件保证原子的「比较并交换」操作。

它有三个参数：

- `V`：要修改的内存地址
- `A`：期望旧值
- `B`：准备写入的新值

伪代码如下：

```cpp
bool CAS(addr V, value A, value B) {
    if (*V == A) {
        *V = B;
        return true;
    }
    return false;
}
```

关键点是：**读旧值、比较、写新值这三个动作整体不可分割**。如果多个线程同时执行，硬件保证同一时刻只有一个 CAS 能成功。

在 x86 上，CAS 通常对应 `LOCK CMPXCHG` 指令；在 ARM 上，常见实现是 LL/SC 语义，也就是 Load-Linked / Store-Conditional。

一个典型例子是无锁计数器：

```cpp
std::atomic<int> counter{0};

void increment() {
    int old_value;
    int new_value;

    do {
        old_value = counter.load(std::memory_order_relaxed);
        new_value = old_value + 1;
    } while (!counter.compare_exchange_weak(old_value, new_value));
}
```

这就是乐观锁的思路：先假设没有冲突，直接尝试修改；如果失败，说明别人抢先改了，那就重新读取、重新计算、再次尝试。

## ABA 问题

CAS 只关心「当前值是不是等于期望值」，但它不知道这个值中间有没有被改过。

比如：

```text
T1: 线程 1 读到值 A
T2: 线程 2 把 A 改成 B，又把 B 改回 A
T3: 线程 1 执行 CAS(A, C)，成功
```

从线程 1 的视角看，值仍然是 `A`，CAS 可以成功；但实际上这个位置已经经历过 `A -> B -> A` 的变化。这就是 ABA 问题。

常见解决方式是加版本号，把比较对象从单个值变成二元组：

```text
(value, version)
```

每次修改时版本号递增。这样即使值又变回 `A`，版本号也已经不同，CAS 就能发现中间发生过变化。Java 里的 `AtomicStampedReference` 就是这个思路。

## Busy Polling：用 CPU 换延迟

Busy polling，也叫忙轮询，指线程不睡眠、不阻塞，而是在一个循环里不断检查条件是否满足。

```cpp
while (!condition_ready()) {
    // busy polling
}

do_work();
```

它和阻塞等待的差异很直接：

| 等待方式 | 行为 | CPU 消耗 | 延迟 |
| --- | --- | --- | --- |
| Blocking | 睡眠，让出 CPU | 低 | 较高，有调度唤醒成本 |
| Busy polling | 一直检查条件 | 高 | 极低，条件满足后立刻响应 |

busy polling 的核心 trade-off 是：**牺牲一个 CPU 核的利用率，换更低的响应延迟**。

所以它常见于这些场景：

- 高频交易：绑定专用核心轮询网卡队列，尽量避免调度和中断延迟
- 网络驱动：Linux NAPI 会在高流量场景下从中断模式切到轮询模式
- 无锁队列消费者：消费者线程不断检查队列是否有新数据
- 自旋锁：本质上就是 busy polling 一个锁变量

实际工程里通常不会无限空转，而是采用混合策略：

```text
先 spin 一小段时间
  -> 还没等到，就 yield 让出时间片
  -> 再等不到，就 futex/condition_variable 真正睡眠
```

这样既能优化短等待场景，又不会在长等待时持续烧 CPU。

## Atomic Ring Buffer：低延迟队列的常见形态

ring buffer 是一个固定大小的循环数组。生产者往里写，消费者从里读，读写位置走到数组尾部后再绕回开头。

```text
slots: [ _ | A | B | C | _ | _ ]
             ^           ^
          read_idx    write_idx
```

基础结构大概是：

```cpp
template <typename T, size_t N>
struct RingBuffer {
    T slots[N];
    std::atomic<size_t> write_idx{0};
    std::atomic<size_t> read_idx{0};
};
```

一般会让 `read_idx` 和 `write_idx` 永远递增，然后通过取模定位数组下标：

```cpp
size_t index = pos % N;
```

这样可以更容易判断空和满：

```text
read_idx == write_idx       -> 空
write_idx - read_idx == N   -> 满
```

## SPSC Ring Buffer：单生产者单消费者

SPSC 是 Single Producer Single Consumer。因为只有一个生产者写 `write_idx`，只有一个消费者写 `read_idx`，所以这个版本甚至不需要 CAS。

生产者：

```cpp
bool push(const T& value) {
    size_t write_pos = write_idx.load(std::memory_order_relaxed);
    size_t read_pos = read_idx.load(std::memory_order_acquire);

    if (write_pos - read_pos == N) {
        return false;
    }

    slots[write_pos % N] = value;
    write_idx.store(write_pos + 1, std::memory_order_release);
    return true;
}
```

消费者：

```cpp
bool pop(T& value) {
    size_t read_pos = read_idx.load(std::memory_order_relaxed);
    size_t write_pos = write_idx.load(std::memory_order_acquire);

    if (read_pos == write_pos) {
        return false;
    }

    value = slots[read_pos % N];
    read_idx.store(read_pos + 1, std::memory_order_release);
    return true;
}
```

这里的重点是：

- `slots[...] = value` 必须发生在 `write_idx` 对消费者可见之前
- 消费者看到新的 `write_idx` 后，必须能看到对应 slot 里的数据

这就是 `release store` 和 `acquire load` 要解决的问题。

## MPMC Ring Buffer：多生产者多消费者

MPMC 是 Multi Producer Multi Consumer。多个生产者会同时抢写入位置，多个消费者会同时抢读取位置，这时就需要 CAS。

工业级 MPMC ring buffer 常见设计是给每个 slot 加一个 `sequence` 字段：

```cpp
template <typename T>
struct Slot {
    T data;
    std::atomic<size_t> sequence;
};
```

生产者大致流程是：

```text
1. 读取 write_idx
2. 找到对应 slot
3. 检查 slot.sequence，判断这个槽位是否可写
4. 用 CAS 抢占 write_idx
5. 写入 data
6. release-store sequence，通知消费者可读
```

消费者则反过来：

```text
1. 读取 read_idx
2. 找到对应 slot
3. 检查 slot.sequence，判断这个槽位是否可读
4. 用 CAS 抢占 read_idx
5. 读取 data
6. release-store sequence，通知生产者可复用
```

这种结构比 mutex 队列复杂，但在低延迟、高吞吐场景下非常常见。LMAX Disruptor、很多交易系统和消息队列内部都能看到类似思路。

## Release Store 与 Acquire Load

先看一个经典问题：

```cpp
// 线程 1
data = 42;
flag = true;

// 线程 2
while (!flag) {}
std::cout << data << std::endl;
```

直觉上，线程 2 看到 `flag == true` 后，应该一定能看到 `data == 42`。但在多核 CPU 和编译器优化下，如果没有同步语义，这个保证并不成立。

正确写法是：

```cpp
// 线程 1
data = 42;
flag.store(true, std::memory_order_release);

// 线程 2
while (!flag.load(std::memory_order_acquire)) {}
std::cout << data << std::endl;
```

可以这样理解：

- `release store`：这条 store 之前的读写，不能被重排到它之后；并且要对看到它的线程可见
- `acquire load`：这条 load 之后的读写，不能被重排到它之前；如果它看到了 release-store 写入的值，也能看到 release 之前的写入

当一个 `acquire load` 读到了另一个线程 `release store` 写入的值，它们之间就建立了 happens-before 关系。

在 ring buffer 里，这个语义非常关键：

```cpp
// producer
slot->data = value;
slot->sequence.store(pos + 1, std::memory_order_release);

// consumer
size_t sequence = slot->sequence.load(std::memory_order_acquire);
// 如果 sequence 表示可读，那么这里一定能看到 producer 写入的 data
```

如果把这里全部换成 `memory_order_relaxed`，消费者可能先看到 `sequence` 更新，却还看不到对应的 `data` 写入。

## Store Buffer：为什么写入不会立刻被别的核心看到

现代 CPU 不会每次写内存都停下来等缓存一致性协议完成。为了提高性能，核心通常会先把写入放进 store buffer，然后继续执行后续指令。

可以把它想成这样：

```text
CPU Core
  -> 执行 store x = 1
  -> 写入先进入 store buffer
  -> CPU 继续往后执行
  -> store buffer 异步把写入刷到 cache，并通过一致性协议让其他核心可见
```

这带来一个重要现象：**当前核心能通过 store-to-load forwarding 看到自己刚写的值，但其他核心可能暂时看不到**。

例如：

```cpp
// 初始 x = 0, y = 0

// Core 0                 // Core 1
x = 1;                    y = 1;
r1 = y;                   r2 = x;
```

在弱内存模型下，可能出现：

```text
r1 == 0 && r2 == 0
```

因为两个核心的写入都还停留在各自的 store buffer 里，对方暂时看不到。

memory barrier、release/acquire 等机制，本质上就是在约束这些乱序和可见性问题：什么时候允许写入继续留在 buffer 里，什么时候必须让之前的写入对其他核心可见。

## False Sharing：无锁结构里的隐形性能杀手

CPU cache 不是按单个变量加载的，而是按 cache line 加载。常见 cache line 大小是 64 字节。

如果两个频繁更新的原子变量刚好落在同一个 cache line 上，即使它们逻辑上毫无关系，也会互相拖慢。

比如：

```cpp
struct BadLayout {
    std::atomic<size_t> write_idx;
    std::atomic<size_t> read_idx;
};
```

生产者不断写 `write_idx`，消费者不断写 `read_idx`。如果两个变量在同一条 cache line 上，每次一个核心写入，都会导致另一个核心对应 cache line 失效。

更好的做法是让它们分开：

```cpp
struct alignas(64) PaddedAtomicSize {
    std::atomic<size_t> value;
};

struct BetterLayout {
    PaddedAtomicSize write_idx;
    PaddedAtomicSize read_idx;
};
```

在低延迟队列、线程池计数器、统计指标里，false sharing 经常是性能抖动的来源。

## CPU Cache Warmup：为什么第一次跑总是慢

CPU cache warmup 指让数据和指令逐渐进入 cache 的过程。

程序刚启动时，相关数据往往不在 cache 里，第一次访问会经历多级 cache miss：

```text
L1 miss -> L2 miss -> L3 miss -> DRAM
```

访问延迟大概可以这样理解：

```text
L1 cache: 几个 cycle
L2 cache: 十几个 cycle
L3 cache: 几十个 cycle
DRAM:     上百个 cycle
```

所以同一段代码，第一次跑和预热后再跑，耗时可能差很多。

这也是 benchmark 需要 warmup 的原因：

```cpp
for (int i = 0; i < warmup_iters; ++i) {
    run_once(); // 丢弃结果，只为预热 cache、分支预测、JIT 等
}

for (int i = 0; i < benchmark_iters; ++i) {
    measure(run_once);
}
```

在 ring buffer 场景下，如果 slots 数组大小能放进 cache，跑过几轮后访问会稳定很多；如果 ring buffer 远大于 LLC，warmup 的收益就会明显下降，因为数据不断被换出 cache。

## 这些概念怎么串起来

可以用一个低延迟消息队列来串联：

```text
1. 生产者用 CAS 抢占 ring buffer 的写入位置
2. 写入 slot.data
3. 用 release store 发布 sequence
4. 消费者 busy polling sequence
5. 用 acquire load 看到 sequence 更新
6. 安全读取 slot.data
7. 通过 padding 避免 false sharing
8. 通过 warmup 减少 cold cache 带来的尾延迟
```

它们不是孤立知识点，而是同一套低延迟并发系统里的不同层次：

- CAS 解决「多个线程怎么无锁抢同一个位置」
- release/acquire 解决「写入顺序和可见性怎么保证」
- store buffer 解释「为什么可见性不是天然成立的」
- busy polling 解决「如何避免阻塞唤醒延迟」
- ring buffer 提供「固定内存、cache-friendly 的队列结构」
- false sharing 和 cache warmup 处理「真实性能为什么和代码看起来不一样」

## 总结

无锁并发不是简单地把 mutex 换成 atomic。它真正难的地方在于：

1. 正确性依赖原子操作和内存序。
2. 性能依赖 cache line、store buffer、预取和调度行为。
3. 低延迟通常不是免费得到的，而是用 CPU、复杂度和可维护性换来的。

所以工程上要先问清楚：

- 是否真的需要无锁？
- 竞争是低还是高？
- 等待时间是短还是长？
- 是追求吞吐，还是追求尾延迟？
- 数据结构能不能放进 cache？

如果只是普通业务并发，mutex、condition_variable 和线程池往往已经足够；如果是交易系统、网络包处理、实时音视频这类低延迟场景，CAS、busy polling、atomic ring buffer 和 cache-aware 优化才真正值得投入。
