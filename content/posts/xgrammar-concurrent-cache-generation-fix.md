---
title: "XGrammar 结构化解码缓存剖析：Unlimited 模式并发 Clear 字节记账竞态与代际修复（PR #1011）"
date: '2026-10-23'
description: "拆解 XGrammar GrammarCompiler 的双层缓存架构与 ThreadSafeLRUCache 锁分离实现，定位 Unlimited 模式下 Clear() 字节残留与在途编译（In-flight Computation）跨代污染根因，给出代际号 + 读写锁原子修复与零 sleep 确定性并发 GoogleTest 验证。"
cover: /images/posts/xgrammar-concurrent-cache-generation-fix/xgrammar-cache-fix-cover-20261022.webp
coverAlt: 双层磨砂玻璃流道经暖橙色代际闸门过滤后汇入晶格缓存矩阵的等距技术插画
categories:
  - AI Infra
tags:
  - AI Infra
  - C++
  - XGrammar
  - 结构化解码
  - 并发编程
  - 源码分析
hidden: false
haloPublished: true
---

在大语言模型服务（如 SGLang、vLLM）的结构化输出场景中，JSON Schema 或 EBNF 语法需要先编译为词表级掩码缓存（`CompiledGrammar`），才能在每个 Decode 步以微秒级开销生成合法 token 位图。XGrammar 通过 `GrammarCompiler` 维护编译产物的内存缓存，并提供 `ClearCache()` 与 `GetCacheSizeBytes()` 供上层推理引擎做显存/内存水位管理。

在阅读和审计 XGrammar（基线提交 `614ab01d`）的缓存实现时，我发现默认无容量上限（`max_memory_bytes = -1`，即 `kUnlimitedSize`）模式下存在两处紧密关联的缺陷：

1. **静态记账残留**：调用 `ClearCache()` 会清空底层哈希表，但未重置 `grammar_level_cache_` 的原子字节计数器 `current_size_`，导致清理后 `GetCacheSizeBytes()` 依然返回旧值，后续重编译同一 Schema 时字节数单调累加；
2. **并发在途记账竞态（Stale In-flight Charge）**：`ThreadSafeLRUCache` 为了避免耗时的语法编译阻塞并发读，将 `packaged_task` 放到锁外执行。如果在途编译尚未完成时另一线程调用了 `Clear()`，旧编译任务结束时仍会无条件执行 `current_size_ += result.size`，把已被清空驱逐的旧代际编译产物字节数错误计入新代际缓存。

本文沿着 `GrammarCompiler` 的源码路径，完整复盘双层缓存结构、锁分离并发模型、为什么 Unlimited 模式不能直接照搬 Finite 模式的阻塞清理逻辑，以及最终提交至上游的代际号（Generation Counter）+ 读写锁原子修复方案（[mlc-ai/xgrammar PR #1011](https://github.com/mlc-ai/xgrammar/pull/1011)）。

## 一、GrammarCompiler 的双层缓存与自适应掩码开销

为什么结构化解码必须依赖编译器缓存？在 XGrammar 中，给定词表大小 $|V|$（例如 32K 或 128K）和一条上下文无关文法 $G$，编译器需要对每个可扫描的有限状态机（FSM）节点预计算 `AdaptiveTokenMask`。对于状态 $s$，词表 $V$ 被划分为三个互不相交的下标集合：

$$
V = T_{\text{acc}}(s) \;\cup\; T_{\text{rej}}(s) \;\cup\; T_{\text{unc}}(s)
$$

其中 $T_{\text{acc}}(s)$ 是无论父规则上下文如何都必定合法的 token 集合，$T_{\text{rej}}(s)$ 是由首字节掩码（`first_character_mask`）或前缀树直接排除的非法集合，$T_{\text{unc}}(s)$ 则是推入字符后可能跨越当前规则末尾、需要在解码运行时结合父栈及 Lookahead 断言动态判定的不确定集合。

![XGrammar 结构化解码编译器的双层缓存与自适应掩码生成流水线](/images/posts/xgrammar-concurrent-cache-generation-fix/xgrammar-compiler-two-level-cache-20261022.webp)

图 1：`GrammarCompiler` 的双层缓存架构。输入 Schema 或 EBNF 经 FSM 优化后，在规则级（Rule-Level）与语法级（Grammar-Level）两个粒度缓存 `AdaptiveTokenMask`。<a href="/images/posts/xgrammar-concurrent-cache-generation-fix/xgrammar-compiler-two-level-cache-20261022.webp" target="_blank" rel="noopener noreferrer">查看大图</a>。

为了兼顾“完全相同的 Schema 复用”与“不同 Schema 之间共享常见子规则（如 JSON 字符串、数字、数组）”，`GrammarCompiler::Impl` 在构造时将总内存预算 `max_memory_bytes` 拆分为两层（见 `cpp/grammar_compiler.cc`）：

- **规则级缓存（`rule_level_cache_`）**：分配 $\frac{1}{3}$ 内存预算（当 `max_memory_bytes == -1` 时为无上限 `(size_t)-1`），以子规则 FSM 的结构哈希与 Lookahead 哈希为键，跨语法共享子规则状态的 `AdaptiveTokenMask`；
- **语法级缓存（`grammar_level_cache_`）**：分配 $\frac{2}{3}$ 内存预算（无上限时同样为 `kUnlimitedSize`），底层是一个 `ThreadSafeLRUCache<UnionKey, CompiledGrammar, Computer, SizeEstimator>`，直接缓存完整的 `CompiledGrammar` 实例。

上层查询缓存占用时，`GetCacheSizeBytes()` 会汇总两层缓存的字节数。一份中等复杂度的 JSON Schema 在 4096 词表下编译出的 `CompiledGrammar` 约占 `1.83 MB`（连同规则级缓存共约 `2.11 MB`），在 128K 词表下则会放大数十倍。因此，`grammar_level_cache_.MemorySize()` 的准确性直接关系到外部缓存管理器的水位判断。

## 二、问题复现与并发竞态根因分析

### 1. 静态清理残留：Clear 后不归零的计数器

我们先看修复前 `cpp/support/thread_safe_cache.h` 中 `ThreadSafeLRUCache::Clear()` 的原始实现：

```cpp
void Clear() {
  // Remove all the ready entries.
  const auto lock_map = std::lock_guard{map_mutex_};
  if (this->max_size_ == kUnlimitedSize)
    cache_.GetMap().clear();
  else
    cache_.LRUEvict(
        [] { return true; },
        [&](const std::shared_future<SizedValue>& value) {
          // always evict and block until the value is ready
          try {
            current_size_ -= value.get().size;
          } catch (...) {
          }
          return true;
        }
    );
}
```

可以看到，当 `max_size_ != kUnlimitedSize`（有限容量模式）时，`Clear()` 通过 `LRUEvict` 逐个等待表项的 `std::shared_future` 就绪，并从 `current_size_` 中减去 `value.get().size`。然而在默认开启的 `kUnlimitedSize` 分支下，代码为了跳过 LRU 双向链表维护，直接调用了 `cache_.GetMap().clear()`，**却遗漏了对 `current_size_` 的重置**。

在单线程公开接口探针（词表大小 4096，编译含 16 个字段约束的 JSON Schema）中，这一遗漏表现为：

| 阶段 | Unlimited 基线（修复前） | Finite 对照（`16 MiB`） | Unlimited 修复后 |
| --- | ---: | ---: | ---: |
| 初始状态 `cache_before_bytes` | `0 B` | `0 B` | `0 B` |
| 首次 `CompileJSONSchema` 后 | `2,109,420 B` | `2,109,420 B` | `2,109,420 B` |
| 调用 `ClearCache()` 后 | **`1,829,544 B`（异常残留）** | `0 B` | **`0 B`** |
| 再次编译同一 Schema 后 | **`3,938,964 B`（翻倍累加）** | `2,109,420 B` | **`2,109,420 B`** |

表中的 `1,829,544 B` 正好是该 `CompiledGrammar` 在 `grammar_level_cache_` 中的大小（其余 `279,876 B` 位于 `rule_level_cache_`，后者在 `ClearCache()` 中被正常清零）。每次清理后重编译，语法级缓存的记账都会凭空多出一份幽灵体积。

### 2. 为什么只加一行 `current_size_ = 0` 不够？

如果只在 `cache_.GetMap().clear()` 后面加上 `current_size_ = 0;`，单线程用例确实能通过，但在多线程并发编译场景下依然存在竞态。原因在于 `GetFutureUnlimited` 的**锁分离（Lock-split）执行模型**：

```cpp
// 修复前的 GetFutureUnlimited 核心逻辑
auto task = std::packaged_task<SizedValue()>{[this, &key] {
  auto value = computer_(key);
  auto result = SizedValue{value, size_estimator_(value)};
  current_size_ += result.size; // <--- 锁外无条件累加！
  return result;
}};

auto lock_map = std::unique_lock{map_mutex_};
auto [it, success] = map.try_emplace(key);
if (!success) return it->second.value;

auto future = task.get_future().share();
it->second.value = future;

// perform the costly computation outside all locks
lock_map.unlock();
task();
return future;
```

为了不让长达数十毫秒的 `computer_(key)` 阻塞其他线程查询缓存，线程 A 在哈希表中插入未就绪的 `std::shared_future` 后立即调用 `lock_map.unlock()`，在锁外执行 `task()`。考虑如下并发交错时序：

1. **时刻 $t_1$**：线程 A 请求编译 `key = K`，持独占锁插入 `future_0`，随后释放锁，开始在锁外执行耗时的 `computer_(K)`；
2. **时刻 $t_2$**：线程 B 调用 `Clear()`，获取 `map_mutex_` 独占锁，清空哈希表并将 `current_size_` 置为 `0` 后返回；
3. **时刻 $t_3$**：线程 C 再次请求编译同一个 `key = K`，由于哈希表已空，线程 C 插入新的 `future_1` 并完成编译，此时缓存真实占用为 $S_1$；
4. **时刻 $t_4$**：线程 A 的旧计算终于完成，在闭包末尾执行 `current_size_ += S_0`。此时哈希表里根本没有线程 A 的条目，但 `current_size_` 被污染成了 $S_1 + S_0$。

![并发缓存中未设围栏的在途任务污染与代际锁修复对比时序图](/images/posts/xgrammar-concurrent-cache-generation-fix/xgrammar-generation-race-timeline-20261022.webp)

图 2：并发交错下的记账竞态（上）与代际围栏修复（下）。旧代际的在途任务在 `Clear()` 之后完成时，通过共享锁校验代际号即可避免向新代际累加幽灵字节。<a href="/images/posts/xgrammar-concurrent-cache-generation-fix/xgrammar-generation-race-timeline-20261022.webp" target="_blank" rel="noopener noreferrer">查看大图</a>。

### 3. 为什么 Unlimited 模式不能照搬 Finite 模式阻塞等待？

一个自然的追问是：既然 Finite 模式的 `Clear()` 会通过 `value.get()` 阻塞等待所有在途 `shared_future` 算完再扣减尺寸，为什么 Unlimited 模式不这样做？

答案有两个层面的考量：

- **非阻塞清理契约与尾延迟**：Unlimited 模式不维护 LRU 链表，调用方在触发 `Clear()` 时预期的是 $O(N)$ 释放字典引用的快速操作，而不希望被后台正在编译的超长 Schema 阻塞数十乃至数百毫秒；已在等待 `old_future` 的旧调用方通过 `shared_future` 依然能正常拿到自己的返回值或异常，无需 `Clear()` 替它们同步等待。
- **持锁等待的死锁风险**：Finite 模式的 `Clear()` 是在**持有 `map_mutex_` 独占锁**期间调用 `value.get()` 的。如果某个在途编译任务内部（或自定义 `computer_` 回调中）间接触发了对同一个缓存的读取或清理，持锁等待 `future` 会立刻造成死锁。因此，保持 Unlimited 模式 `Clear()` 的非阻塞特性是更安全的系统设计。

## 三、代际号（Generation Counter）+ 读写锁原子修复

既然允许旧代际的在途编译在 `Clear()` 返回后继续跑完，那么核心不变量就应当定义为：**每一次 `Clear()` 都会开启一个新的缓存代际（Generation）；只有当编译任务所属的代际与当前缓存代际相同时，其产物的字节数才允许计入 `current_size_`。**

要让这条不变量在并发环境下严格成立，必须注意一个极易踩坑的细节：**不能只用原子变量比较 `generation == generation_`，然后再执行 `current_size_ += result.size`**。如果比较与累加不是相对 `Clear()` 原子的，线程 A 刚刚判断完 `generation == generation_` 为真，线程 B 就切进来执行完整个 `Clear()`（将 `generation_` 加一并将 `current_size_` 清零），随后线程 A 继续执行下一行 `current_size_ += result.size`，竞态依然会发生！

正确的做法是利用现有的 `std::shared_mutex map_mutex_`：

- `Clear()` 原本就持有独占锁 `std::lock_guard{map_mutex_}`，在独占锁内同时完成 `++generation_`、`cache_.GetMap().clear()` 和 `current_size_ = 0`；
- 编译任务在锁外完成耗时的 `computer_(key)` 与 `size_estimator_(value)` 后，获取轻量级的**读共享锁** `std::shared_lock{map_mutex_}`，在共享锁保护下检查 `if (generation == generation_) current_size_ += result.size;`。

这样一来，多个并发编译任务完成时的记账可以彼此并行（共享锁互不阻塞），但任何任务做代际校验与字节累加的瞬间，都与 `Clear()` 的独占锁严格互斥，彻底消除了 TOCTOU（Time-of-Check to Time-of-Use）窗口。同时，我还顺手修复了原代码在 `unique_lock` 之前就构造 `packaged_task` 的小瑕疵——改为在持有 `unique_lock` 确认第二次查找仍未命中后，才捕获当前 `generation_` 并构造 `task`。

核心补丁（`cpp/support/thread_safe_cache.h`）如下：

```cpp
  void Clear() {
    // Remove cached entries.
    const auto lock_map = std::lock_guard{map_mutex_};
    if (this->max_size_ == kUnlimitedSize) {
      // In-flight computations still return their values, but no longer belong to this cache.
      ++generation_;
      cache_.GetMap().clear();
      current_size_ = 0;
    } else
      cache_.LRUEvict(
          [] { return true; },
          [&](const std::shared_future<SizedValue>& value) {
            // always evict and block until the value is ready
            try {
              current_size_ -= value.get().size;
            } catch (...) {
              // fine, just ignore the exception, size is not updated
            }
            return true;
          }
      );
  }
```

以及 `GetFutureUnlimited` 的代际捕获与共享锁记账实现：

```cpp
  std::shared_future<SizedValue> GetFutureUnlimited(const Key& key) {
    auto& map = cache_.GetMap();

    {
      auto lock_map = std::shared_lock{map_mutex_};
      auto it = map.find(key);
      if (it != map.end()) return it->second.value;
    }

    auto lock_map = std::unique_lock{map_mutex_};
    auto it = map.find(key);
    if (it != map.end()) return it->second.value;

    const auto generation = generation_;
    auto task = std::packaged_task<SizedValue()>{[this, &key, generation] {
      auto value = computer_(key);
      auto result = SizedValue{value, size_estimator_(value)};
      // Serialize the generation check and accounting with Clear().
      auto lock_map = std::shared_lock{map_mutex_};
      if (generation == generation_) current_size_ += result.size;
      return result;
    }};

    auto future = task.get_future().share();
    map.try_emplace(key).first->second.value = future;

    // perform the costly computation outside all locks
    lock_map.unlock();
    task();
    return future;
  }
```

## 四、零 Sleep 的确定性并发单测与死锁门卫设计

并发 Bug 的回归测试最忌讳依赖 `std::this_thread::sleep_for` 碰运气。为了在 GoogleTest 中 100% 确定性地复现并验证上述跨代交错时序，我在 `tests/cpp/test_thread_safe_cache.cc` 中使用两对 `std::promise<void>`（`entered` 与 `release`）构造了精确的同步屏障，并用参数化测试 `TEST_P(UnlimitedCacheClearTest, ...)` 同时覆盖旧任务正常返回与抛出异常两条路径：

```cpp
class UnlimitedCacheClearTest : public ::testing::TestWithParam<bool> {};

TEST_P(UnlimitedCacheClearTest, InFlightCompletionDoesNotChargeANewerGeneration) {
  using namespace std::chrono_literals;
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> release;
  auto release_future = release.get_future().share();
  std::atomic<int> calls{0};
  const bool fail_first = GetParam();
  auto compute = [&](const int& key) {
    EXPECT_EQ(key, 7);
    const int call = ++calls;
    if (call == 1) {
      entered.set_value();
      release_future.wait();
      if (fail_first) throw std::runtime_error("first computation failed");
    }
    return call * 17;
  };
  auto estimate = [](int value) { return static_cast<std::size_t>(value); };
  using Cache = ThreadSafeLRUCache<int, int, decltype(compute), decltype(estimate)>;
  Cache cache(Cache::kUnlimitedSize, compute, estimate);
  std::future<int> old_get, new_get;
  std::future<void> clear;
  struct ReleaseGuard {
    std::promise<void>& promise;
    bool released{false};
    void Release() {
      if (!released) {
        promise.set_value();
        released = true;
      }
    }
    ~ReleaseGuard() { Release(); }
  } release_guard{release};
  // Destroy the guard before the futures, so any early exit unblocks their computations.
  old_get = std::async(std::launch::async, [&] { return cache.Get(7); });

  if (entered_future.wait_for(5s) != std::future_status::ready) {
    release_guard.Release();
    old_get.wait();
    FAIL() << "The old computation did not start";
  }
  clear = std::async(std::launch::async, [&] {
    cache.Clear();
    cache.Clear();
  });
  if (clear.wait_for(5s) != std::future_status::ready) {
    release_guard.Release();
    old_get.wait();
    clear.wait();
    FAIL() << "Unlimited Clear blocked on an in-flight computation";
  }
  clear.get();
  EXPECT_EQ(cache.MemorySize(), 0);

  // Reuse the same key before the old computation finishes.
  new_get = std::async(std::launch::async, [&] { return cache.Get(7); });
  if (new_get.wait_for(5s) != std::future_status::ready) {
    release_guard.Release();
    old_get.wait();
    new_get.wait();
    FAIL() << "Clear did not remove the old in-flight entry";
  }
  EXPECT_EQ(new_get.get(), 34);
  EXPECT_EQ(cache.MemorySize(), 34);
  EXPECT_EQ(cache.Get(7), 34);
  EXPECT_EQ(calls, 2);
  cache.Clear();
  EXPECT_EQ(cache.MemorySize(), 0);
  EXPECT_EQ(cache.Get(7), 51);
  EXPECT_EQ(cache.MemorySize(), 51);

  release_guard.Release();
  if (fail_first) {
    EXPECT_THROW(old_get.get(), std::runtime_error);
  } else {
    EXPECT_EQ(old_get.get(), 17);
  }
  EXPECT_EQ(cache.MemorySize(), 51);
  EXPECT_EQ(cache.Get(7), 51);
  EXPECT_EQ(calls, 3);
}
```

这段测试代码中有两个非常关键的工程防卫设计：

1. **C++ 栈对象逆序析构与 `ReleaseGuard` 声明位置**：`std::async` 返回的 `std::future` 在析构时会隐式阻塞等待异步线程结束。如果 `release_guard` 声明在 `old_get` 之前，一旦测试中途因某条断言失败或异常提前退栈，`old_get` 会先于 `release_guard` 析构，而此时 `old_get` 还在等待 `release_future`，从而造成测试进程永久死锁。将 `release_guard` 声明在 `old_get, new_get, clear` **之后**，保证栈展开时 `~ReleaseGuard()` 第一个执行，自动解开 `release` 屏障。
2. **异步重查与 5 秒死锁超时门卫**：在 `Clear()` 之后，第 2 代首次查询同一个 `key = 7`（`new_get`）同样放在 `std::async` 中并用 `wait_for(5s)` 保护。这样即便未来有人误删了 `cache_.GetMap().clear()` 导致 `new_get` 拿到旧的未就绪 `shared_future`，测试也会在 5 秒后干净报错退出，而不会挂死整个 CI。

## 五、全量编译审计、回归验证与实现边界

在修改头文件 `cpp/support/thread_safe_cache.h` 后，我首先使用 `clang++ -MM` 对仓库全部 43 个编译单元（Translation Units）做了传递头文件依赖扫描（保存于 `dependency-audit.json`），确认仅有编号 15（`cpp/grammar_compiler.cc`）与编号 39（`tests/cpp/test_thread_safe_cache.cc`）依赖该头文件。

在相同编译参数（macOS arm64，`clang++ -std=c++17 -O1 -pthread`）下，将新增的 4 项参数化回归测试分别链接到未修改基线与修复后的目标文件：

| 测试用例 | 测试目标 | 未修改基线（Baseline） | 修复后（Fixed） |
| --- | --- | --- | --- |
| `CompilerCacheClearTest/0` (`max_bytes = -1`) | Unlimited 公开编译器 `ClearCache` 与重编译记账 | **FAILED**（Clear 后残留 `504 B`，重编后涨至 `1012 B` vs 预期 `508 B`） | **PASSED**（`0 -> 508 -> 0 -> 508 -> 0`） |
| `CompilerCacheClearTest/1` (`max_bytes = 1 MiB`) | Finite 公开编译器 `ClearCache` 与重编译记账 | **PASSED**（`0 -> 508 -> 0 -> 508 -> 0`） | **PASSED**（`0 -> 508 -> 0 -> 508 -> 0`） |
| `UnlimitedCacheClearTest/0` (`fail_first = false`) | 并发在途编译成功完成时不污染新代际缓存值与字节数 | **FAILED**（Clear 后未清零，且旧任务完成后多加 `17 B`） | **PASSED**（保持第 3 代 `51 B` 不变，旧调用方仍拿到 `17`） |
| `UnlimitedCacheClearTest/1` (`fail_first = true`) | 并发在途编译抛异常时不污染新代际状态 | **FAILED**（跨代清理期间 `MemorySize()` 未清零） | **PASSED**（旧调用方正常捕获 `runtime_error`，新代际保持 `51 B`） |
| **全量原生 C++ 测试集（87 项）** | 覆盖 EBNF、JSON Schema、Regex、Structural Tag 等全部单测 | 83 passed（不含新增 4 项） | **87 passed, 0 failed**（总耗时 `1.02 s`） |

### 适用范围与工程边界说明

1. **修复的是容量记账缺陷（Accounting Defect），而非物理内存泄漏**：在修复前，`cache_.GetMap().clear()` 已经正确析构了哈希表中的 `std::shared_future` 节点，底层 `CompiledGrammar` 的引用计数在无人持有时已正常释放；缺陷在于原子计数器 `current_size_` 未归零且未对跨代在途任务加围栏，导致对外报告的缓存水位失真。
2. **语法级缓存与规则级缓存的边界**：本次修复针对的是 `ThreadSafeLRUCache`（即 `grammar_level_cache_`）。如果在调用 `GrammarCompiler::ClearCache()` 时恰好有在途的多线程语法编译正在运行，该在途编译内部的 `GrammarMatcherForTokenMaskCache` 仍可能继续向独立的 `rule_level_cache_` 写入子规则掩码。要将 `rule_level_cache_` 也纳入同一代际围栏，需要跨组件传递编译代际上下文，不在本次最小正交补丁的修改范围内。
3. **验证环境边界**：本地回归验证基于 C++17 原生构建与公开 C++ API 探针（并已通过仓库全量 `pre-commit`、`clang-format 19.1.7` 与 `ruff check`），不包含 Python 绑定层或 GPU 端到端推理吞吐测试。补丁已提交为上游 [PR #1011](https://github.com/mlc-ai/xgrammar/pull/1011)。

*本文封面与两张架构示意图使用 [aigc-image-generator](https://github.com/liangqianxing/aigc-image-generator) 辅助生成并经人工核对，用于展示缓存层级与代际并发时序；精确锁粒度与状态转移以文中 C++ 源码为准。*
