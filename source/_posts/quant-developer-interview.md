---
title: 量化开发面试准备：从 ACM 背景切入的完整指南
date: 2026-04-08
categories: 技术
tags:
  - 量化开发
  - C++
  - 面试
  - 低延迟
  - 系统设计
---

有 ACM 背景去面量化开发，和普通后端有几个关键的不同。这篇是完整版，覆盖公司类型、C++ 深度、系统设计、网络/OS、数学概率、面试流程和资源，针对百亿私募和头部券商自营方向。

<!-- more -->

## 一、先搞清楚你要面哪类公司

量化开发不是一个岗位，是一类岗位。不同公司考察侧重差异巨大。

### 公司类型

| 类型 | 代表公司 | 延迟要求 | 技术侧重 |
|------|---------|---------|---------|
| HFT/超高频 | Jane Street, Citadel Securities, 奇点金融 | 纳秒级 | 无锁编程、FPGA、内核旁路 |
| 百亿私募（中高频） | 幻方、明汯、九坤、灵均 | 微秒~毫秒 | C++ 性能优化、交易系统设计 |
| 券商自营/做市 | 中信、国泰君安、华泰 | 毫秒级 | 系统稳定性、风控、行情处理 |
| 量化基金（低频） | 景林、富善 | 秒级以上 | Python/C++ 回测框架、因子研究 |

**ACM 背景最匹配的是百亿私募**。HFT 太依赖硬件和专有知识，低频基金 C++ 用得少。私募的核心需求恰好是：C++ 熟练 + 算法扎实 + 能设计交易相关系统。

### 量化开发 vs 量化研究

面试前要想清楚：
- **量化开发（Quant Dev）**：写交易系统、行情处理、回测引擎，代码是核心产出
- **量化研究（Quant Researcher）**：挖因子、建模、研究策略，Python + 统计是主要工具

本文针对量化开发。

---

## 二、C++：考察的是系统层认知，不是语法

ACM 选手 C++ 语法不是问题，要补的是**语言底层行为和性能工程**。

### 2.1 内存模型和对象生命周期

这是高频考点，每轮面试几乎都会涉及。

**移动语义：**

```cpp
// move 之后原对象处于 valid but unspecified state
std::vector<int> a = {1, 2, 3};
std::vector<int> b = std::move(a);
// a.size() 可能是 0，也可能不是，标准不保证
// 只保证 a 可以被安全地销毁或重新赋值

// 完美转发：保留值类别
template<typename T>
void wrapper(T&& arg) {
    target(std::forward<T>(arg));  // 左值传左值，右值传右值
}
```

**shared_ptr 的线程安全边界：**

```cpp
// 引用计数本身的修改是原子的（线程安全）
// 但 shared_ptr 对象本身不是（赋值操作不是原子的）
// 被管理的对象也不是

std::shared_ptr<int> global = std::make_shared<int>(0);

// 危险：多线程同时修改同一个 shared_ptr 对象
// thread1: global = std::make_shared<int>(1);  // 非原子
// thread2: global = std::make_shared<int>(2);  // 数据竞争

// 正确做法：atomic<shared_ptr> (C++20) 或外加锁
std::atomic<std::shared_ptr<int>> atomic_ptr = std::make_shared<int>(0);
```

**内存池（核心路径禁止 new/delete）：**

```cpp
// 简单对象池：预分配，O(1) 分配和释放
template<typename T, size_t N>
class ObjectPool {
    alignas(T) char storage[N * sizeof(T)];
    std::stack<T*> free_list;
public:
    ObjectPool() {
        for (size_t i = 0; i < N; ++i)
            free_list.push(reinterpret_cast<T*>(storage + i * sizeof(T)));
    }
    T* acquire() {
        if (free_list.empty()) return nullptr;
        T* p = free_list.top(); free_list.pop();
        return new(p) T{};  // placement new，不分配内存
    }
    void release(T* p) {
        p->~T();
        free_list.push(p);
    }
};
```

### 2.2 多线程和内存模型

量化开发最高频考点之一。

**memory_order 不是背诵题，要理解为什么：**

```cpp
// 场景：无锁发布订阅（SPSC 场景极常见）
std::atomic<int*> ptr{nullptr};
int data = 0;

// 生产者线程
data = 42;                                    // (1)
ptr.store(new_ptr, std::memory_order_release); // (2)
// release 保证：(1) 不会被重排到 (2) 之后

// 消费者线程
int* p = ptr.load(std::memory_order_acquire);  // (3)
if (p) use(*p);                                // (4)
// acquire 保证：(4) 不会被重排到 (3) 之前
// acquire-release 配对：消费者一定能看到生产者 release 之前的所有写入
```

**无锁队列（SPSC，单生产者单消费者）：**

```cpp
// 量化交易里用得最多的无锁结构
// 行情线程 -> 策略线程，典型的 SPSC 场景
template<typename T, size_t N>
class SPSCQueue {
    alignas(64) std::atomic<size_t> head_{0};  // 消费者用
    alignas(64) std::atomic<size_t> tail_{0};  // 生产者用
    T data_[N];

public:
    bool push(const T& val) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next = (tail + 1) % N;
        if (next == head_.load(std::memory_order_acquire)) return false; // 满
        data_[tail] = val;
        tail_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& val) {
        size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire)) return false; // 空
        val = data_[head];
        head_.store((head + 1) % N, std::memory_order_release);
        return true;
    }
};
// 注意：head_ 和 tail_ 分别放在不同 cache line（alignas(64)）
// 避免 false sharing：两个线程分别写不同变量，但在同一 cache line 会互相失效
```

**false sharing 是什么：**

```cpp
// 错误写法：a 和 b 可能在同一个 cache line（64 字节）
struct Bad {
    int a;  // thread 1 写
    int b;  // thread 2 写
    // CPU 以 cache line 为单位同步，两个核写不同变量也会相互竞争
};

// 正确写法：强制对齐到 cache line 边界
struct Good {
    alignas(64) int a;
    alignas(64) int b;
};
```

### 2.3 模板和编译期计算

量化代码里大量用模板，不是为了炫技，是为了**零开销抽象**。

```cpp
// CRTP：编译期多态，比虚函数少一次间接跳转
template<typename Derived>
class Strategy {
public:
    void on_tick(const Tick& t) {
        static_cast<Derived*>(this)->on_tick_impl(t);  // 编译期确定，无虚表
    }
};

class MeanReversion : public Strategy<MeanReversion> {
public:
    void on_tick_impl(const Tick& t) { /* 均值回归逻辑 */ }
};

// std::variant + std::visit：替代虚函数的另一种方式
using Order = std::variant<LimitOrder, MarketOrder, StopOrder>;
std::visit([](auto& o) { o.execute(); }, order);  // 编译期分发
```

### 2.4 性能分析常用工具

面试时如果聊到性能优化，能说出工具链是加分项：

```bash
# perf：Linux 性能分析神器
perf stat ./trading_engine          # CPU 周期、cache miss、分支预测失败率
perf record -g ./trading_engine     # 采样调用栈
perf report                         # 火焰图数据

# valgrind cachegrind：cache 访问模拟
valgrind --tool=cachegrind ./program

# 编译器优化选项
g++ -O3 -march=native -flto -fprofile-use  # 生产环境常用
```

---

## 三、系统设计：交易系统架构

大厂面试必考，ACM 背景在这里容易吃亏，要专门准备。

### 3.1 完整交易链路

```
行情源（交易所/数据商）
    ↓ UDP 组播 / TCP
行情接入层（解码、归一化）
    ↓ 内部消息总线（无锁队列 / 共享内存）
策略引擎（信号计算）
    ↓
风控模块（实时检查）
    ↓
执行层（报单、撤单、改单）
    ↓ FIX 协议 / 柜台专有协议
券商柜台 → 交易所
```

**每一层的延迟预算（参考数量级）：**

| 环节 | 延迟目标 |
|------|---------|
| 行情解码 | < 1 μs |
| 策略计算 | < 5 μs |
| 风控检查 | < 1 μs |
| 报单发出 | < 10 μs |
| 全链路 | < 50~100 μs |

### 3.2 订单簿设计（面试必考）

**基础实现（Map 版本）：**

```cpp
// 价格档位用 map：有序、O(log n) 插入/查询
// 买单：价格降序；卖单：价格升序
struct Order {
    uint64_t order_id;
    int qty;
    uint64_t timestamp;
};

class OrderBook {
    std::map<int, std::list<Order>, std::greater<int>> bids;  // 买盘
    std::map<int, std::list<Order>> asks;                     // 卖盘
    std::unordered_map<uint64_t, 
        std::pair<int, std::list<Order>::iterator>> order_map;  // id -> 快速定位

public:
    void add_order(bool is_bid, int price, Order order) {
        auto& side = is_bid ? bids : asks;  // 不行，类型不同，需要重构
        // 实际用模板或者分开写
    }

    // 撤单 O(1)：通过 order_map 直接找到迭代器
    void cancel_order(uint64_t order_id) {
        auto it = order_map.find(order_id);
        if (it == order_map.end()) return;
        auto [price, list_it] = it->second;
        bids[price].erase(list_it);  // O(1)，因为有迭代器
        if (bids[price].empty()) bids.erase(price);
        order_map.erase(it);
    }
};
```

**面试追问清单（能答出来就够了）：**

1. 为什么用 `map` 不用 `unordered_map`？
   - 订单簿需要按价格遍历（找最优价、范围查询），hash map 做不到有序遍历
2. 撤单频繁时怎么优化？
   - lazy deletion：标记已撤，匹配时跳过，避免从队列中频繁删除元素
3. 多线程下如何加锁？
   - 粗粒度：整个订单簿一把锁（简单，但竞争大）
   - 细粒度：每个价格档位一把锁（复杂，但并发更好）
   - 无锁：SPSC 场景下，行情线程写，策略线程读，用原子操作
4. 如何测试吞吐量？
   - 生成随机订单流，统计 1 秒内能处理的报单/撤单数量

### 3.3 行情系统设计

```
UDP 组播接收（内核 -> 用户态）
    ↓
序列号检验（检测丢包）
    ↓
解码（二进制协议 / 专有格式）
    ↓
归一化（统一内部格式）
    ↓
分发（按证券代码路由到对应策略）
```

**丢包处理：**

```cpp
// 序列号不连续 -> 丢包 -> 触发 TCP 重传请求（行情通常支持）
class MarketDataHandler {
    uint64_t expected_seq_ = 1;

    void on_packet(const Packet& pkt) {
        if (pkt.seq > expected_seq_) {
            // 丢包：请求重传 expected_seq_ ~ pkt.seq-1
            request_retransmit(expected_seq_, pkt.seq - 1);
            // 同时缓存当前包，等重传回来按序处理
            buffer_[pkt.seq] = pkt;
            return;
        }
        process(pkt);
        expected_seq_ = pkt.seq + 1;
        // 检查缓存里是否有后续包可以处理
        while (buffer_.count(expected_seq_)) {
            process(buffer_[expected_seq_]);
            buffer_.erase(expected_seq_++);
        }
    }
};
```

### 3.4 风控系统

```cpp
// 交易前风控（pre-trade risk check）：拦截异常订单
struct RiskCheck {
    // 单笔订单限额
    bool check_single_order(const Order& o) {
        return o.qty * o.price <= max_single_order_value_;
    }

    // 当日累计持仓限额
    bool check_position(const std::string& symbol, int delta) {
        return abs(positions_[symbol] + delta) <= position_limit_[symbol];
    }

    // 频率限制：单位时间内报单数
    bool check_rate_limit() {
        auto now = Clock::now();
        // 滑动窗口计数
        while (!timestamps_.empty() && 
               now - timestamps_.front() > window_) {
            timestamps_.pop();
        }
        if (timestamps_.size() >= max_orders_per_window_) return false;
        timestamps_.push(now);
        return true;
    }
};
```

---

## 四、网络和操作系统

这是普通后端面试不考，但量化开发**必考**的部分。

### 4.1 内核旁路（Kernel Bypass）

正常网络包走的路径：

```
网卡 -> 内核网络栈 -> 系统调用 -> 用户态
                  ↑ 这里有 5~10 μs 的不可避免延迟
```

内核旁路的方案：

```
网卡 -> 用户态（直接）  <- 完全绕过内核
        ↑ 延迟降到 1~2 μs
```

常见技术：
- **DPDK**（Data Plane Development Kit）：用户态网络驱动，CPU 轮询收包（不是中断）
- **RDMA / InfiniBand**：Remote Direct Memory Access，绕过 CPU 直接读写远端内存，延迟 < 1 μs
- **Solarflare / Xilinx Onload**：商业化内核旁路方案，交易所常用

面试时知道这些概念和为什么用就够，不需要写过 DPDK。

### 4.2 CPU 亲和性和 NUMA

```bash
# 将进程绑定到特定 CPU 核心，避免上下文切换和跨核 cache 失效
taskset -c 2,3 ./trading_engine

# NUMA（非统一内存访问）：访问本地内存 vs 跨 NUMA 节点内存相差 2-3 倍延迟
numactl --cpunodebind=0 --membind=0 ./trading_engine  # 绑定到 NUMA 节点 0
```

```cpp
// C++ 代码里设置 CPU 亲和性
#include <pthread.h>
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(2, &cpuset);  // 绑定到第 2 个核
pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
```

### 4.3 大页内存（HugePage）

```cpp
// 普通内存：4KB 页，TLB（地址转换缓存）频繁 miss
// 大页：2MB / 1GB 页，减少 TLB miss，对大型数据结构性能提升显著

// mmap 分配大页
void* ptr = mmap(nullptr, size,
    PROT_READ | PROT_WRITE,
    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,  // MAP_HUGETLB 关键标志
    -1, 0);
```

### 4.4 忙轮询 vs 中断

```cpp
// 中断驱动：有包到来才唤醒，适合普通服务
// 忙轮询（Busy Polling/Spinning）：线程持续检查，适合低延迟场景

// 忙轮询牺牲 CPU 换延迟
while (true) {
    if (queue.pop(msg)) {
        process(msg);
    }
    // 不 sleep，不让出 CPU
    // 代价：一个核 100% 占用
    // 收益：响应延迟从毫秒级降到微秒级
}
```

### 4.5 常考的 OS 面试题

**上下文切换的代价是什么？**
- 保存/恢复寄存器（几十纳秒）
- TLB 刷新（几百纳秒）
- Cache 污染（影响最大，可能增加几微秒）
- 这是量化代码尽量避免上下文切换的根本原因

**虚拟内存 / 缺页中断：**
- 首次访问内存触发缺页中断，需要几微秒甚至几毫秒
- 解决方案：程序启动时 `memset` 预热所有内存，避免交易时触发缺页

**UDP vs TCP：**
- 行情接收用 UDP 组播（低延迟、一对多），自己处理丢包重传
- 报单通道用 TCP（可靠性优先），或者交易所专有协议

---

## 五、算法：面试里的真实考法

### 5.1 并发设计题（最高频）

**读写锁实现：**

```cpp
// 用两个 mutex 和计数器实现读写锁
// 多读者可同时读，写者独占
class RWLock {
    std::mutex read_mtx, write_mtx;
    int readers = 0;
public:
    void read_lock() {
        std::lock_guard<std::mutex> lock(read_mtx);
        if (++readers == 1) write_mtx.lock();  // 第一个读者锁定写锁
    }
    void read_unlock() {
        std::lock_guard<std::mutex> lock(read_mtx);
        if (--readers == 0) write_mtx.unlock();  // 最后一个读者释放写锁
    }
    void write_lock()   { write_mtx.lock(); }
    void write_unlock() { write_mtx.unlock(); }
};
// 注意：这个实现有读者饥饿问题，写者可能永远等不到
// 面试时指出这点是加分项
```

**线程池：**

```cpp
class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;

public:
    ThreadPool(size_t n) {
        for (size_t i = 0; i < n; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<typename F>
    void enqueue(F&& f) {
        { std::lock_guard<std::mutex> lock(mtx); tasks.emplace(std::forward<F>(f)); }
        cv.notify_one();
    }

    ~ThreadPool() {
        { std::lock_guard<std::mutex> lock(mtx); stop = true; }
        cv.notify_all();
        for (auto& w : workers) w.join();
    }
};
```

### 5.2 数据流 / 在线算法（第二高频）

量化里大量是 online 场景，数据流不能重复遍历。

**滑动窗口统计：**

```cpp
// 维护最近 N 个 tick 的均值和方差（均值回归策略的基础）
class RollingStats {
    std::deque<double> window;
    double sum = 0, sum_sq = 0;
    size_t max_size;
public:
    RollingStats(size_t n) : max_size(n) {}

    void push(double x) {
        window.push_back(x);
        sum += x; sum_sq += x * x;
        if (window.size() > max_size) {
            double old = window.front(); window.pop_front();
            sum -= old; sum_sq -= old * old;
        }
    }

    double mean() const { return sum / window.size(); }
    double variance() const {
        double m = mean();
        return sum_sq / window.size() - m * m;
    }
    double stddev() const { return std::sqrt(variance()); }
};
```

**水塘抽样：**

```cpp
// 数据流中等概率取 k 个样本，只遍历一次
// 不知道总数 n（典型的 online 算法）
std::vector<int> reservoir_sample(std::istream& stream, int k) {
    std::vector<int> res;
    std::mt19937 rng(42);
    int count = 0, x;
    while (stream >> x) {
        ++count;
        if ((int)res.size() < k) {
            res.push_back(x);
        } else {
            // 以 k/count 的概率替换
            std::uniform_int_distribution<int> dist(0, count - 1);
            int j = dist(rng);
            if (j < k) res[j] = x;
        }
    }
    return res;
}
```

**Top-K 问题（分钟内最活跃的 K 只股票）：**

```cpp
// 用小根堆维护 Top-K，时间 O(n log k)
std::vector<std::pair<int,std::string>> topK(
    const std::unordered_map<std::string, int>& freq, int k) {
    // min-heap：始终保留频率最高的 k 个
    using P = std::pair<int,std::string>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> heap;
    for (auto& [sym, cnt] : freq) {
        heap.push({cnt, sym});
        if ((int)heap.size() > k) heap.pop();  // 弹出最小的
    }
    std::vector<P> result;
    while (!heap.empty()) { result.push_back(heap.top()); heap.pop(); }
    return result;
}
```

### 5.3 概率和随机算法

**随机打乱（Fisher-Yates Shuffle）：**

```cpp
// 等概率随机排列，O(n)，面试常考证明正确性
void shuffle(std::vector<int>& arr) {
    std::mt19937 rng(std::random_device{}());
    for (int i = arr.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        std::swap(arr[i], arr[dist(rng)]);
        // 正确性：第 i 个位置最终放任意一个元素的概率都是 1/n
    }
}
```

---

## 六、数学和统计

量化开发不是量化研究员，不需要推导期权定价。但以下要会。

### 6.1 概率题（笔试常见）

**经典题：期望步数**

> 连续抛硬币，出现连续 2 次正面平均需要抛多少次？

设 E 为期望步数。列方程（状态机法）：

```
状态 0（初始）：
  - 抛正面（概率 1/2）-> 状态 1，耗费 1 步
  - 抛反面（概率 1/2）-> 状态 0，耗费 1 步
  E0 = 1 + (1/2)*E1 + (1/2)*E0

状态 1（已有 1 次正面）：
  - 抛正面（概率 1/2）-> 完成，耗费 1 步
  - 抛反面（概率 1/2）-> 状态 0，耗费 1 步
  E1 = 1 + (1/2)*0 + (1/2)*E0

解方程：E0 = 6
```

这类题考的是建模能力，不是记结论。

**赌徒破产问题：**

> 每次赢 1 元（概率 p）或输 1 元（概率 1-p），从 k 元出发，到达 n 元或 0 元，求最终破产概率。

```
破产概率 = (r^k - r^n) / (1 - r^n)，其中 r = (1-p)/p（p ≠ 0.5 时）
p = 0.5 时，破产概率 = (n-k)/n
```

量化里的意义：策略的边际优势（p > 0.5）有多重要，止损点（0 元）的设置如何影响生存率。

### 6.2 常用指标实现

```python
# Sharpe 比率：超额收益 / 收益波动率
# 年化 Sharpe = (均值收益 - 无风险利率) / 收益标准差 * sqrt(252)
def sharpe_ratio(returns, rf=0.0):
    excess = np.array(returns) - rf / 252
    return np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(252)

# 最大回撤 O(n)：维护历史最高点
def max_drawdown(prices):
    peak = prices[0]
    max_dd = 0
    for p in prices:
        peak = max(peak, p)
        drawdown = (peak - p) / peak
        max_dd = max(max_dd, drawdown)
    return max_dd

# Calmar 比率：年化收益 / 最大回撤（比 Sharpe 更看重下行风险）
def calmar_ratio(returns, prices):
    annualized_return = (1 + sum(returns)) ** (252 / len(returns)) - 1
    return annualized_return / max_drawdown(prices)
```

### 6.3 时间序列基础

量化开发不需要推导，但要能解释概念：

| 概念 | 一句话解释 | 量化里的场景 |
|------|-----------|------------|
| 平稳性 | 均值和方差不随时间变化 | 判断价差序列是否适合均值回归 |
| 自相关 | 序列和自身滞后项的相关 | 动量效应的统计依据 |
| 协整 | 两个非平稳序列的线性组合是平稳的 | 配对交易（统计套利） |
| ADF 检验 | 检验单位根（非平稳的一种） | 验证价差是否可以均值回归 |

---

## 七、项目推荐：比八股有说服力

推荐做 **能实际跑的 C++ 项目**，展示代码质量比背面试题更有说服力。

### 7.1 异步日志库（适合展示 C++ 并发能力）

**核心设计：**
- 前端（业务线程）：将日志消息写入无锁队列，不阻塞业务
- 后端（日志线程）：消费队列，批量写磁盘

```cpp
// 关键实现点
class AsyncLogger {
    SPSCQueue<LogMsg, 65536> queue_;  // 无锁队列
    std::thread backend_thread_;
    std::ofstream file_;

    void backend_loop() {
        LogMsg msg;
        while (running_) {
            if (queue_.pop(msg)) {
                file_ << format(msg);
                // 批量 flush，减少 fsync 次数
                if (++write_count_ % 1000 == 0) file_.flush();
            }
        }
    }
public:
    void log(Level level, const char* fmt, ...) {
        // 格式化后放入队列，不阻塞调用方
        queue_.push(make_msg(level, fmt, ...));
    }
};
```

**面试中可以聊的点：**
- 为什么用 SPSC 而不是加锁的普通队列
- 队列满了怎么办（丢弃 / 阻塞 / 降级到同步）
- 如何测量实际写入延迟（rdtsc）

### 7.2 简化版撮合引擎（最直接相关）

**功能范围：** 支持限价单、市价单、撤单，输出成交回报。

**性能目标：** 单线程 50 万笔/秒以上订单处理能力。

```cpp
// 核心数据结构
class MatchingEngine {
    // bid: 价格从高到低（买方愿意出高价优先成交）
    std::map<Price, std::list<Order>, std::greater<Price>> bids_;
    // ask: 价格从低到高（卖方愿意接受低价优先成交）
    std::map<Price, std::list<Order>> asks_;
    // 快速定位订单用于撤单
    std::unordered_map<OrderId, OrderIter> order_index_;

    std::vector<Trade> try_match() {
        std::vector<Trade> trades;
        while (!bids_.empty() && !asks_.empty()) {
            auto& [bid_px, bid_q] = *bids_.begin();
            auto& [ask_px, ask_q] = *asks_.begin();
            if (bid_px < ask_px) break;  // 买价 < 卖价，无法成交

            auto& bid = bid_q.front();
            auto& ask = ask_q.front();
            int qty = std::min(bid.qty, ask.qty);
            trades.push_back({bid.id, ask.id, ask_px, qty});

            bid.qty -= qty;
            ask.qty -= qty;
            if (bid.qty == 0) { order_index_.erase(bid.id); bid_q.pop_front(); }
            if (ask.qty == 0) { order_index_.erase(ask.id); ask_q.pop_front(); }
            if (bid_q.empty()) bids_.erase(bid_px);
            if (ask_q.empty()) asks_.erase(ask_px);
        }
        return trades;
    }
};
```

### 7.3 回测框架（展示系统设计能力）

关键设计决策：
- 事件驱动架构：Tick -> Strategy -> Signal -> Order -> Fill
- 向量化计算：因子计算用 NumPy / Eigen
- 精确的成本模型：滑点、手续费、冲击成本

---

## 八、面试流程和各公司风格

### 8.1 典型面试流程

**百亿私募（如幻方、明汯）：**
1. 简历筛选（ACM 经历很加分）
2. 笔试：C++ 题 + 算法题 + 概率数学题（2~3 小时）
3. 技术一面：代码 + 项目 + C++ 底层
4. 技术二面：系统设计 + 深挖项目
5. HR 面

**笔试常见题型：**
- C++ 输出结果题（考虑复制/移动/析构顺序）
- 多线程代码找 bug（数据竞争、死锁）
- 算法题（通常 2~3 道，注重代码质量）
- 概率题（期望、条件概率）

### 8.2 C++ 输出题示例

```cpp
// 这类题笔试常考，考察对象生命周期理解
struct A {
    int x;
    A(int x) : x(x) { std::cout << "ctor " << x << "\n"; }
    A(const A& o) : x(o.x) { std::cout << "copy " << x << "\n"; }
    A(A&& o) : x(o.x) { o.x = 0; std::cout << "move " << x << "\n"; }
    ~A() { std::cout << "dtor " << x << "\n"; }
};

A f() {
    A a(1);
    return a;  // NRVO：大概率不调用移动构造，直接构造在返回值的位置
}

int main() {
    A b = f();      // 可能输出：ctor 1  （NRVO，0 次额外拷贝）
    A c = std::move(b);  // 输出：move 1
}  // 输出：dtor 1, dtor 0（b 被 move 后 x=0）
```

### 8.3 面试中加分的表达方式

给出实现后主动分析：
- 时间/空间复杂度
- 这个实现的局限性在哪
- 如果数据量大 10 倍、并发量大 100 倍，哪里会成为瓶颈
- 生产环境里还需要考虑哪些问题（监控、日志、容错）

---

## 九、学习资源

### C++ 性能和底层

| 资源 | 内容 | 推荐程度 |
|------|------|---------|
| 《Effective Modern C++》Scott Meyers | 移动语义、智能指针、lambda，必读 | ★★★★★ |
| 《C++ Concurrency in Action》Anthony Williams | 多线程和内存模型权威资料 | ★★★★★ |
| CppCon 演讲（YouTube） | 性能优化实战，尤其 Chandler Carruth 的 talks | ★★★★☆ |
| [Abseil C++ Tips](https://abseil.io/tips/) | Google 内部 C++ 最佳实践 | ★★★★☆ |

### 系统设计和低延迟

| 资源 | 内容 |
|------|------|
| 《UNIX 网络编程》Stevens | 网络编程基础，socket 到 select/epoll |
| [High Frequency Trading Systems](https://www.amazon.com/dp/1801810540) | HFT 系统工程实践 |
| Martin Thompson 的博客（mechanical-sympathy.blogspot.com）| CPU、内存、并发性能深度文章 |

### 量化基础

| 资源 | 内容 |
|------|------|
| 《Advances in Financial Machine Learning》 | 因子挖掘方法论 |
| QuantLib 源码 | 看金融计算库的工程实现 |
| 各交易所开发者文档 | 了解真实协议格式（上交所、深交所都有公开文档） |

---

## 总结

ACM 背景面量化开发，优势在算法和 C++ 基础，需要补的是：

1. **C++ 底层行为**（内存模型、并发、性能优化）——不是语法，是"代码为什么慢"
2. **系统设计**（交易系统架构）——从没做过，需要专门学
3. **OS 和网络**（内核旁路、CPU 亲和性）——和竞赛知识体系不重叠
4. **一个能跑的 C++ 项目**——比背八股有说服力得多

准备顺序建议：C++ 并发（2 周）→ 写一个项目（2~3 周）→ 系统设计（1 周）→ 刷题复习（1 周）。
