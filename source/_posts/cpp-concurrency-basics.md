---
title: C++ 并发编程入门：从数据竞争到线程池
date: 2026-04-15
categories: 技术
tags:
  - C++
  - 并发
  - 面试
  - 量化开发
---

量化开发面试必考并发编程，但很多人第一次接触就直接看线程池代码，结果一头雾水。这篇文章从最基础的数据竞争开始，一步步推导出有界阻塞队列和线程池，每个概念都从"它解决什么问题"出发。

<!-- more -->

## 第一课：数据竞争

先看这段代码，猜一下输出是什么：

```cpp
#include <iostream>
#include <thread>

int counter = 0;

void add() {
    for (int i = 0; i < 100000; i++) {
        counter++;
    }
}

int main() {
    std::thread t1(add);
    std::thread t2(add);
    t1.join();
    t2.join();
    std::cout << counter << std::endl;
}
```

两个线程各加 10 万次，直觉上应该输出 200000。实际运行：

```
131072
118934
157823
```

每次都不一样，几乎从来不是 200000。

**为什么？**

`counter++` 看起来是一行，但 CPU 实际执行是三步：

```
1. 把 counter 从内存读到寄存器   (READ)
2. 寄存器里的值 +1              (ADD)
3. 把寄存器的值写回内存          (WRITE)
```

两个线程同时跑，可能发生这种情况：

```
时间线      线程1                线程2
────────────────────────────────────────
t1          READ  → 拿到 100
t2                               READ  → 也拿到 100  ← 还没被写回！
t3          ADD   → 算出 101
t4                               ADD   → 也算出 101
t5          WRITE → 写回 101
t6                               WRITE → 写回 101    ← 覆盖了！
```

两个线程各加了一次，但 counter 只从 100 变成 101，丢了一次加法。这就叫**数据竞争（Data Race）**。

---

## 四个核心工具

### 1. mutex

本质：一把锁，同一时刻只有一个线程能持有它。

```cpp
std::mutex mtx;

mtx.lock();    // 拿锁，拿不到就在这里阻塞等待
counter++;     // 现在只有我一个线程在这里
mtx.unlock();  // 还锁
```

问题是手动 lock/unlock 很危险：

```cpp
mtx.lock();
doSomething();  // 如果这里抛异常
mtx.unlock();   // 永远不会执行 → 死锁
```

### 2. lock_guard

本质：一个对象，构造时加锁，析构时自动解锁。

```cpp
{
    std::lock_guard<std::mutex> lock(mtx);  // 构造 → 加锁
    counter++;
}  // 离开大括号 → lock 析构 → 自动解锁
```

利用 C++ 的 RAII：对象销毁时自动执行清理。不管正常退出还是异常，析构函数一定执行，锁一定释放。

**局限：** 加锁后不能中途解锁，整个生命周期都是锁定状态。

### 3. unique_lock

本质：比 lock_guard 更灵活，可以中途解锁/重新加锁。

```cpp
std::unique_lock<std::mutex> lock(mtx);  // 构造 → 加锁

lock.unlock();   // 中途解锁
// ... 做一些不需要锁的事
lock.lock();     // 重新加锁
// 析构时如果还持有锁，自动解锁
```

只有两种情况必须用 unique_lock：
1. 需要配合 condition_variable（cv.wait 内部需要中途解锁）
2. 需要在持有锁期间手动解锁

### 4. condition_variable

mutex 只能解决"同时访问"的问题，解决不了"等待某个条件"的问题。

比如"队列空时，取数据的线程要等待"，用 mutex 的朴素想法是：

```cpp
while (queue.empty()) {
    // 空转等待？← 线程一直占着 CPU，非常浪费
}
```

这叫**忙等（busy waiting）**。condition_variable 解决这个问题：

```cpp
std::condition_variable cv;

// 等待方
cv.wait(lock, [] { return !queue.empty(); });  // 条件不满足就睡觉

// 通知方
cv.notify_one();   // 叫醒一个等待的线程
cv.notify_all();   // 叫醒所有等待的线程
```

**wait 内部到底发生了什么：**

```cpp
// cv.wait(lock, pred) 等价于：
while (!pred()) {
    mtx.unlock();   // 解锁，让别人能放数据
    // 线程进入睡眠...
    // 被 notify 唤醒...
    mtx.lock();     // 重新加锁
}
// 条件满足，继续执行
```

为什么必须先解锁再睡？如果持有锁去睡觉，生产者永远拿不到锁放数据，消费者永远不会被唤醒——死锁。

**为什么 wait 要传 lambda：**

操作系统存在**虚假唤醒**：线程没有被 notify，却自己醒了。不传 lambda 的话醒来队列可能还是空的就去取数据，直接崩。lambda 保证醒来条件一定为真。

**四者关系：**

```
mutex
  │ 太危险，需要自动管理
  ▼
lock_guard          ← 够用就用这个
  │ 需要中途解锁时
  ▼
unique_lock         ← 配合 cv 时必须用这个
  │ 需要等待条件时
  ▼
condition_variable  ← wait/notify，线程间协调
```

**一句话记忆：**

```
mutex          → 门上的锁
lock_guard     → 进门自动锁，出门自动开（钥匙绑在脚上）
unique_lock    → 同上，但可以临时把钥匙摘下来
cv.wait        → 我先把钥匙放门口，去睡觉，有人叫我再来拿
cv.notify_one  → 喂，你可以来拿钥匙了
```

---

## 实战一：有界阻塞队列

理解了四个工具，阻塞队列就是把它们组合起来。

**push 的逻辑：**
1. 加锁
2. 如果队列满了，等待（not_full 条件变量）
3. 放入数据
4. 通知等待取数据的线程（not_empty）
5. 解锁

**pop 的逻辑：**
1. 加锁
2. 如果队列空了，等待（not_empty 条件变量）
3. 取出数据
4. 通知等待放数据的线程（not_full）
5. 解锁

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class BoundedBlockingQueue {
public:
    explicit BoundedBlockingQueue(size_t capacity)
        : capacity_(capacity) {}

    void push(T val) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < capacity_; });
        queue_.push(std::move(val));
        not_empty_.notify_one();
    }

    void pop(T& val) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty(); });
        val = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
    }

    bool try_push(T val) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= capacity_) return false;
        queue_.push(std::move(val));
        not_empty_.notify_one();
        return true;
    }

    size_t size() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    size_t capacity_;
    std::queue<T> queue_;
    mutable std::mutex mutex_;       // mutable：const 方法里也能加锁
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
};
```

**几个细节：**

- 两个条件变量分开（not_full / not_empty）：每次只唤醒对应的等待方，避免惊群
- `mutable mutex_`：size() 是 const 方法，但读 queue_.size() 也需要加锁防止读到中间态
- `std::move`：push 时 move 进去，pop 时 move 出来，避免不必要的拷贝

---

## 实战二：线程池

线程池 = 有界阻塞队列 + 一组工作线程。

**类比餐厅：**
- 厨师 = 线程（提前雇好，一直在岗）
- 点单 = 提交任务
- 待处理订单队列 = 任务队列

不用线程池的话，每来一个任务就 new thread，做完就销毁——每次点菜都临时招一个厨师，代价极高。

**结构：**

```
ThreadPool
├── vector<thread> workers        // 固定数量的工作线程
├── queue<function<void()>> tasks // 待执行的任务队列
├── mutex                         // 保护任务队列
├── condition_variable            // 通知有新任务 / 要关闭了
└── bool stop                     // 析构时通知线程退出
```

**完整实现：**

```cpp
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <stdexcept>

class ThreadPool {
public:
    explicit ThreadPool(size_t n) : stop_(false) {
        for (size_t i = 0; i < n; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();  // 锁已释放，执行任务
                }
            });
        }
    }

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<decltype(f(args...))>
    {
        using ReturnType = decltype(f(args...));

        // packaged_task 不可拷贝，用 shared_ptr 包装后 lambda 可以捕获
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<ReturnType> result = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_) throw std::runtime_error("submit on stopped ThreadPool");
            tasks_.emplace([task] { (*task)(); });
        }
        cv_.notify_one();
        return result;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();  // 唤醒所有线程检查 stop_
        for (auto& w : workers_) w.join();
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};
```

**使用：**

```cpp
ThreadPool pool(4);

auto f1 = pool.submit([](int a, int b) { return a + b; }, 3, 5);
auto f2 = pool.submit([] { return 42; });

std::cout << f1.get() << "\n";  // 8
std::cout << f2.get() << "\n";  // 42
```

**几个关键问题：**

**为什么用 packaged_task 而不是直接存 function？**
因为要拿返回值。packaged_task 内部绑定了一个 promise，执行完自动把结果放进去，外面用 future.get() 取出来。

**为什么 task 要用 shared_ptr 包装？**
packaged_task 不能拷贝，但 std::function 要求内容可拷贝。shared_ptr 本身可拷贝（引用计数），lambda 捕获 shared_ptr 后存进 function<void()> 就没问题了。

**析构为什么用 notify_all 而不是 notify_one？**
要让所有线程都醒来检查 stop_，不然没被唤醒的线程会永远阻塞，join() 就会卡死。

**默写路径（按逻辑推导，不要硬背）：**

```
1. 构造：for(n个线程) { emplace_back(主循环lambda) }
2. 主循环：while(true) { 加锁→wait→取任务→解锁→执行 }
3. wait 条件：stop_ || !tasks_.empty()
4. 退出条件：stop_ && tasks_.empty()
5. submit：packaged_task→shared_ptr→lambda入队→notify_one→返回future
6. 析构：stop_=true→notify_all→join all
```

---

## 量化开发手撕路线图

有了并发基础，接下来的学习路径：

**第一阶段（并发基础，已完成）**
- ✅ 数据竞争、mutex、lock_guard、unique_lock
- ✅ condition_variable、有界阻塞队列
- ✅ 线程池

**第二阶段（量化特色）**
- 内存池：预分配 + freelist，避免 malloc 开销
- SPSC 无锁队列：ring buffer + atomic，行情线程→策略线程的标准方案
- 读写锁：shared_mutex，读多写少场景

**第三阶段（数据结构）**
- LRU Cache（unordered_map + list）
- 滑动窗口均值/方差（O(1) 更新）
- 定时器轮（TimerWheel）

量化开发和普通后端的核心差异是**低延迟意识**：每写一行代码都要问自己"这里有没有不必要的拷贝/锁/内存分配"。mutex 版本写对写熟比死磕无锁更重要，面试官更在意你理解 happens-before 和数据竞争，不在意你背了多少原子操作。
