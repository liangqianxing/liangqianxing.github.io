---
title: 量化开发面试准备：从 ACM 背景切入的视角
date: 2026-04-08
categories: 技术
tags:
  - 量化开发
  - C++
  - 面试
---

有 ACM 背景去面量化开发，和普通后端有几个关键的不同。这篇整理一下我的理解，主要针对百亿私募和头部券商自营方向。

<!-- more -->

## 量化开发和后端开发的本质区别

后端开发怕的是 bug，量化开发怕的是**延迟和错误同时发生**。

交易链路上，一个下单请求从信号触发到订单入场通常要求在 100 微秒以内完成。这不是 CRUD，代码里任何一处动态分配、锁竞争、cache miss 都是看得见的延迟代价。所以量化开发面试考的 C++，不是"会不会写"，而是"知不知道这段代码会慢在哪"。

## C++：考察重点在底层行为

有 ACM 底子的话，语法和算法不是瓶颈，反而要补的是**语言底层和系统层的认知**。

**内存和对象生命周期：**

```cpp
// 高频问：shared_ptr 的线程安全边界在哪？
// 引用计数本身是原子操作，安全
// 但它管理的对象不是，多线程同时写需要加锁

// 高频问：std::move 之后原对象处于什么状态？
std::vector<int> a = {1, 2, 3};
std::vector<int> b = std::move(a);
// a 处于 valid but unspecified state，不能假设 a 是空的（虽然实际上是）
```

**多线程和内存模型：**

```cpp
// memory_order 是量化开发高频考点
std::atomic<bool> ready{false};
int data = 0;

// 生产者
data = 42;
ready.store(true, std::memory_order_release);  // 保证 data=42 不被重排到 store 之后

// 消费者
while (!ready.load(std::memory_order_acquire));  // 保证后续读 data 不被重排到 load 之前
std::cout << data;  // 安全，一定看到 42
```

**低延迟的核心原则：**

- 核心路径不 new/delete（提前分配内存池）
- 不用虚函数（vtable 查找多一次间接寻址，~1-3ns）
- 结构体按 cache line 对齐，避免 false sharing
- 减少系统调用，用 mmap 替代 read/write

## 算法：类型和竞赛有差异

量化开发的算法题和 ICPC 风格不一样：**更在意工程实现质量，不在意 trick**。

几类高频题型：

**1. 并发设计题**

```cpp
// 线程安全的无界队列（生产者消费者模型）
template<typename T>
class SafeQueue {
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cv;
public:
    void push(T val) {
        std::lock_guard<std::mutex> lock(mtx);
        q.push(std::move(val));
        cv.notify_one();
    }
    T pop() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]{ return !q.empty(); });
        T val = std::move(q.front());
        q.pop();
        return val;
    }
};
```

**2. 随机采样**

```cpp
// 水塘抽样：数据流等概率取 k 个，只遍历一次
// 面试常考，竞赛几乎不考
std::vector<int> sample(std::istream& stream, int k) {
    std::vector<int> res;
    int count = 0, x;
    while (stream >> x) {
        count++;
        if (count <= k) {
            res.push_back(x);
        } else {
            int j = rand() % count;
            if (j < k) res[j] = x;  // 以 k/count 概率替换
        }
    }
    return res;
}
```

**3. 订单簿实现**

```cpp
// 限价订单簿：买卖盘分别用 map 维护
// 价格优先 + 时间优先（FIFO）
struct OrderBook {
    // 买单：价格降序（用 greater）
    std::map<int, std::queue<Order>, std::greater<int>> bids;
    // 卖单：价格升序
    std::map<int, std::queue<Order>> asks;

    void match() {
        while (!bids.empty() && !asks.empty()) {
            auto [bid_price, bid_q] = *bids.begin();
            auto [ask_price, ask_q] = *asks.begin();
            if (bid_price < ask_price) break;  // 无法成交
            // 成交逻辑...
        }
    }
};
```

## 数学：够用就行

量化开发不是量化研究，不需要推导期权定价。但以下得会：

| 概念 | 真正要理解的 |
|------|------------|
| Sharpe 比率 | 不是越高越好，要结合回撤一起看 |
| 最大回撤 | O(n) 算法：维护历史最高点，线性扫描 |
| 滑点和手续费 | 回测不算这两项，上线就是亏钱 |
| 期望计算 | 能推导"连续抛正面 2 次平均要几次"就够了 |

```python
# 最大回撤 O(n) 实现
def max_drawdown(prices):
    peak = prices[0]
    max_dd = 0
    for p in prices:
        peak = max(peak, p)
        max_dd = max(max_dd, (peak - p) / peak)
    return max_dd
```

## 面试会真正拉开差距的地方

同样写出了订单簿，有人能继续讲：

- 为什么用 `std::map` 而不是 `unordered_map`（订单簿需要价格有序）
- 如果撤单很频繁，`queue` 里存已撤单 id 更好还是直接删（lazy deletion vs 实时删）
- 多线程下怎么加锁，锁粒度在价格档位级别还是整个订单簿
- 测过的数据是什么，吞吐量大概多少

能把这几个问题讲清楚，就够了。

## ACM 背景可以重点突出的地方

- C++ 熟练度不用证明，但要把方向从"正确性"转到"性能"
- 算法题通常不是瓶颈，把精力集中在多线程和系统设计上
- 竞赛里学的很多是 offline 算法，面试里更多是 online 和流式数据场景，注意转换

推荐补一个实际能跑的 C++ 项目：异步日志库或者简化版订单簿，比任何八股都有说服力。
