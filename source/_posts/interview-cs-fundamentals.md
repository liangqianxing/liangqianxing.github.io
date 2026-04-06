---
title: 面试八股速通：计算机网络 / 操作系统 / C++ 语法
date: 2026-04-07
categories: 技术
tags:
  - 面试
  - 计算机网络
  - 操作系统
  - C++
---

ACM 竞赛背景去面后端/研发岗，这三块是必考的。整理成问答形式，方便快速过一遍。

<!-- more -->

## 一、计算机网络

### TCP vs UDP

| | TCP | UDP |
|---|---|---|
| 连接 | 面向连接（三次握手） | 无连接 |
| 可靠性 | 可靠（确认、重传、排序） | 不可靠 |
| 速度 | 慢（有开销） | 快 |
| 适合场景 | HTTP、文件传输、数据库 | 视频流、DNS、游戏 |

### TCP 三次握手 / 四次挥手

**三次握手（建立连接）：**

```
客户端 → SYN(seq=x)          → 服务端   [客户端: SYN_SENT]
客户端 ← SYN+ACK(seq=y,ack=x+1) ← 服务端   [服务端: SYN_RCVD]
客户端 → ACK(ack=y+1)        → 服务端   [双方: ESTABLISHED]
```

为什么是三次不是两次？两次握手服务端无法确认客户端能收到自己的消息，可能建立无效连接。

**四次挥手（断开连接）：**

```
客户端 → FIN → 服务端   [客户端: FIN_WAIT_1]
客户端 ← ACK ← 服务端   [客户端: FIN_WAIT_2，服务端还可以发数据]
客户端 ← FIN ← 服务端   [服务端: LAST_ACK]
客户端 → ACK → 服务端   [客户端: TIME_WAIT，等 2MSL 后关闭]
```

为什么四次？TCP 是全双工，两个方向需要分别关闭。服务端收到 FIN 后可能还有数据要发，所以 ACK 和 FIN 分开发。

**TIME_WAIT 为什么等 2MSL？**
确保最后一个 ACK 能到达服务端（如果丢了，服务端会重发 FIN，客户端需要能响应）。

### HTTP/1.1 vs HTTP/2 vs HTTP/3

| | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---|---|---|---|
| 传输层 | TCP | TCP | QUIC（基于 UDP）|
| 多路复用 | 不支持（队头阻塞） | 支持（同一连接多流） | 支持（流级别独立）|
| 头部压缩 | 无 | HPACK | QPACK |
| 队头阻塞 | 有 | TCP 层仍有 | 彻底解决 |

**HTTP/2 多路复用原理：** 把请求拆成帧（frame），多个请求的帧交错发送，接收端按 stream ID 重组。一个连接并发多个请求，不需要多开 TCP 连接。

### HTTPS 握手过程

```
1. 客户端 → ClientHello（支持的加密套件、随机数 C）
2. 服务端 → ServerHello（选定套件、随机数 S）+ 证书
3. 客户端验证证书（CA 链）
4. 客户端生成预主密钥，用服务端公钥加密发送
5. 双方用 C + S + 预主密钥 推导出对称密钥
6. 后续通信用对称加密（AES 等）
```

TLS 1.3 简化为 1-RTT（甚至 0-RTT 恢复连接）。

### DNS 解析过程

```
浏览器缓存 → 系统 hosts → 本地 DNS 服务器
  → 根域名服务器（返回 .com 服务器地址）
  → .com 顶级域名服务器（返回 example.com 服务器地址）
  → example.com 权威服务器（返回 IP）
  → 缓存结果（TTL 时间内有效）
```

### TCP 拥塞控制

四个阶段：
1. **慢启动**：cwnd 从 1 开始，每个 RTT 翻倍（指数增长）
2. **拥塞避免**：cwnd 超过阈值（ssthresh）后，每 RTT 加 1（线性增长）
3. **快重传**：收到 3 个重复 ACK，立即重传，不等超时
4. **快恢复**：快重传后 ssthresh = cwnd/2，cwnd = ssthresh，进入拥塞避免

### 常见状态码

| 码 | 含义 |
|---|---|
| 200 | OK |
| 301 | 永久重定向 |
| 302 | 临时重定向 |
| 304 | Not Modified（缓存有效） |
| 400 | Bad Request（客户端参数错误） |
| 401 | Unauthorized（未认证） |
| 403 | Forbidden（无权限） |
| 404 | Not Found |
| 429 | Too Many Requests（限流） |
| 500 | Internal Server Error |
| 502 | Bad Gateway（上游服务挂了） |
| 503 | Service Unavailable |

---

## 二、操作系统

### 进程 vs 线程 vs 协程

| | 进程 | 线程 | 协程 |
|---|---|---|---|
| 资源 | 独立地址空间 | 共享进程资源 | 共享线程资源 |
| 切换开销 | 大（上下文切换） | 中 | 极小（用户态切换） |
| 通信 | IPC（管道、共享内存） | 共享内存（需加锁） | 直接共享（单线程内） |
| 崩溃影响 | 不影响其他进程 | 可能导致整个进程崩溃 | 影响当前线程 |

**协程的本质：** 用户态的"假线程"，由程序自己调度（不是 OS），切换时只保存少量寄存器，开销极小。Python 的 `async/await`、Go 的 goroutine 都是协程思想。

### 死锁

**四个必要条件（Coffman 条件）：**
1. 互斥：资源同时只能被一个进程持有
2. 持有并等待：进程持有资源同时等待其他资源
3. 不可剥夺：资源不能被强制取走
4. 循环等待：进程间形成等待环

**预防死锁：** 破坏任意一个条件。最常用：资源有序分配（破坏循环等待）、一次性申请所有资源（破坏持有并等待）。

### 内存管理

**虚拟内存：** 每个进程有独立的虚拟地址空间，通过页表映射到物理内存。好处：进程隔离、内存可以超过物理内存（换页到磁盘）。

**页面置换算法：**
- **LRU（最近最少使用）**：淘汰最久没被访问的页，实际用近似算法（时钟算法）
- **FIFO**：淘汰最早进入的页，可能出现 Belady 异常
- **OPT（最优）**：淘汰未来最久不用的页，理论最优但不可实现

**内存碎片：**
- 内部碎片：分配的内存比实际需要大（固定分区）
- 外部碎片：空闲内存总量够但不连续（动态分区）
- 解决：分页（消除外部碎片）、内存紧缩

### 进程调度算法

| 算法 | 特点 | 适合场景 |
|---|---|---|
| FCFS | 先来先服务，简单但可能饥饿 | 批处理 |
| SJF | 最短作业优先，平均等待时间最短 | 批处理 |
| 时间片轮转 | 公平，响应时间有保证 | 交互式系统 |
| 优先级调度 | 高优先级先执行，可能饥饿 | 实时系统 |
| 多级反馈队列 | 综合以上，Linux 实际使用 | 通用 |

### 锁

**互斥锁（Mutex）：** 同一时刻只有一个线程持有，其他线程阻塞等待（睡眠）。

**自旋锁（Spinlock）：** 等待时不睡眠，循环检查（忙等）。适合锁持有时间极短的场景，避免线程切换开销。

**读写锁：** 多个读者可以并发，写者独占。适合读多写少场景。

**乐观锁 vs 悲观锁：**
- 悲观锁：先加锁再操作（Mutex）
- 乐观锁：不加锁，提交时检查是否有冲突（CAS、数据库版本号）

### 系统调用

用户态程序不能直接访问硬件，需要通过系统调用陷入内核态。过程：

```
用户程序调用 read() → 触发软中断（int 0x80 / syscall 指令）
→ CPU 切换到内核态 → 内核执行 sys_read → 返回用户态
```

常见系统调用：`fork`、`exec`、`open`/`read`/`write`、`mmap`、`socket`。

---

## 三、C++ 语法

### 指针 vs 引用

| | 指针 | 引用 |
|---|---|---|
| 可为空 | 可以（`nullptr`） | 不可以 |
| 可重新绑定 | 可以 | 不可以（初始化后固定） |
| 有自己的地址 | 有 | 没有（是别名） |
| 算术运算 | 支持 | 不支持 |

引用本质是编译器保证非空的指针别名，通常用于函数参数传递避免拷贝。

### 智能指针

C++11 引入，解决手动 `delete` 导致的内存泄漏和悬空指针问题。

```cpp
// unique_ptr：独占所有权，不可复制，可移动
std::unique_ptr<int> p1 = std::make_unique<int>(42);
std::unique_ptr<int> p2 = std::move(p1);  // p1 变为 nullptr

// shared_ptr：共享所有权，引用计数
std::shared_ptr<int> sp1 = std::make_shared<int>(42);
std::shared_ptr<int> sp2 = sp1;  // 引用计数 = 2
// 两个都析构后，引用计数归 0，内存释放

// weak_ptr：不增加引用计数，解决循环引用
std::weak_ptr<int> wp = sp1;
if (auto locked = wp.lock()) {  // 使用前先 lock
    // locked 是 shared_ptr
}
```

**循环引用问题：** A 持有 B 的 `shared_ptr`，B 持有 A 的 `shared_ptr`，两者引用计数永远不为 0，内存泄漏。解决：其中一方改用 `weak_ptr`。

### 移动语义 / 右值引用

```cpp
// 左值：有名字，有地址
int a = 10;

// 右值：临时对象，没有名字
int b = a + 1;  // a+1 是右值

// 右值引用
int&& r = std::move(a);  // std::move 把左值转为右值引用

// 移动构造函数：把资源"偷"过来，不拷贝
class MyVector {
    int* data;
    size_t size;
public:
    // 移动构造：直接接管指针，原对象置空
    MyVector(MyVector&& other) noexcept
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
};
```

移动语义的意义：避免深拷贝，把 O(n) 的拷贝变成 O(1) 的指针转移。

### 虚函数 / 多态

```cpp
class Animal {
public:
    virtual void speak() { std::cout << "..."; }
    virtual ~Animal() {}  // 基类析构函数必须是虚函数！
};

class Dog : public Animal {
public:
    void speak() override { std::cout << "Woof"; }
};

Animal* a = new Dog();
a->speak();  // 输出 "Woof"，运行时多态
```

**虚函数表（vtable）：** 每个有虚函数的类有一个 vtable，存放虚函数指针。对象内存里有一个 vptr 指向 vtable。调用虚函数时通过 vptr 查表，有一次间接寻址开销。

**纯虚函数 / 抽象类：**
```cpp
class Shape {
public:
    virtual double area() = 0;  // 纯虚函数，子类必须实现
};
```

### 内存布局

```
栈（Stack）：局部变量、函数参数，自动管理，LIFO，空间小（通常 8MB）
堆（Heap）：new/malloc 分配，手动管理（或智能指针），空间大
全局/静态区：全局变量、static 变量，程序生命周期内存在
代码区：存放编译后的机器码，只读
```

### 常见陷阱

**1. 未初始化变量**
```cpp
int x;
std::cout << x;  // 未定义行为，可能是任意值
```

**2. 数组越界**
```cpp
int arr[5];
arr[5] = 1;  // 未定义行为，可能覆盖其他内存
```

**3. 悬空指针**
```cpp
int* p = new int(42);
delete p;
*p = 1;  // 未定义行为，p 已经是悬空指针
p = nullptr;  // delete 后立即置空
```

**4. 对象切片**
```cpp
Dog dog;
Animal a = dog;  // 切片！只复制了 Animal 部分，Dog 的数据丢失
Animal& ref = dog;  // 正确：用引用或指针
```

### STL 容器选型

| 容器 | 底层 | 查找 | 插入/删除 | 适合场景 |
|---|---|---|---|---|
| `vector` | 动态数组 | O(n) | 尾部 O(1)，中间 O(n) | 随机访问，尾部增删 |
| `deque` | 分段数组 | O(1) | 两端 O(1) | 双端队列 |
| `list` | 双向链表 | O(n) | 任意位置 O(1) | 频繁中间插删 |
| `map` | 红黑树 | O(log n) | O(log n) | 有序键值对 |
| `unordered_map` | 哈希表 | O(1) 均摊 | O(1) 均摊 | 无序键值对，高频查找 |
| `set` | 红黑树 | O(log n) | O(log n) | 有序不重复集合 |
| `priority_queue` | 堆 | O(1) 取最值 | O(log n) | 堆/优先队列 |

### Lambda 表达式

```cpp
// [捕获列表](参数) -> 返回类型 { 函数体 }
auto add = [](int a, int b) { return a + b; };

int x = 10;
// 值捕获（拷贝）
auto f1 = [x]() { return x; };
// 引用捕获
auto f2 = [&x]() { x++; };
// 捕获所有（值）
auto f3 = [=]() { return x; };

// 配合 STL
std::vector<int> v = {3, 1, 4, 1, 5};
std::sort(v.begin(), v.end(), [](int a, int b) { return a > b; });
```

### 模板

```cpp
// 函数模板
template<typename T>
T max(T a, T b) { return a > b ? a : b; }

// 类模板
template<typename T>
class Stack {
    std::vector<T> data;
public:
    void push(T val) { data.push_back(val); }
    T pop() { T v = data.back(); data.pop_back(); return v; }
};

// 使用
Stack<int> s;
s.push(1);
```

模板在编译期展开，零运行时开销，但会增加编译时间和二进制体积。

---

## 面试高频问题汇总

**Q：TCP 为什么可靠？**
序列号保证有序、确认号 + 超时重传保证不丢、滑动窗口控制流量、拥塞控制避免网络拥塞。

**Q：进程间通信方式？**
管道（pipe）、消息队列、共享内存（最快）、信号量（同步）、Socket（跨机器）。

**Q：`malloc` 和 `new` 的区别？**
`malloc` 是 C 库函数，只分配内存；`new` 是 C++ 运算符，分配内存 + 调用构造函数。`free` vs `delete` 同理。`new[]` 对应 `delete[]`，不能混用。

**Q：`const` 的用法？**
```cpp
const int* p;      // 指向常量的指针，不能通过 p 修改值
int* const p;      // 常量指针，p 本身不能改变指向
const int* const p; // 两者都不能改
void func() const; // 成员函数，不能修改成员变量
```

**Q：`static` 的用法？**
- 局部变量：生命周期延长到程序结束，只初始化一次
- 成员变量/函数：属于类而非对象，所有实例共享
- 全局变量/函数：限制作用域在当前文件（内部链接）

**Q：`sizeof` 结构体为什么不等于成员之和？**
内存对齐。编译器为了 CPU 访问效率，会在成员之间插入 padding，使每个成员的地址是其大小的整数倍。可以用 `#pragma pack(1)` 关闭对齐（但可能降低性能）。
