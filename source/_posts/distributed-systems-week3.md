---
title: Week 3：分布式系统速成——MapReduce、Raft、容错与 Distributed KV Store
date: 2026-05-05
categories: 技术
tags:
  - 分布式系统
  - MIT 6.824
  - MapReduce
  - Raft
  - KV Store
  - Ray
  - Agent
---

Week 1 我们用 mini autograd 理解了深度学习框架的本质；Week 2 我们从 GPU、Kernel、KV cache 和 batching 理解了推理系统的性能瓶颈。Week 3 要补的是另一块底层能力：**分布式系统思想**。

这部分不需要做 MIT 6.824 的 lab，也不需要陷入每个 RPC 细节。只要抓住四个主题：MapReduce 思想、Raft 共识、Fault tolerance、Distributed KV store。学完之后，你会发现 Ray、分布式推理、多 agent 系统、workflow engine 的底层心智模型高度相似。甚至 DeepScientist 本质上也可以看成一个 mini distributed system。
<!-- more -->

![MapReduce pipeline](/images/posts/distributed-systems/mapreduce.svg)

## 1. 为什么 AI 系统也要学分布式

很多 AI 项目表面上是模型应用，实际运行起来都是分布式系统：

```text
用户请求
  -> API gateway
  -> planner
  -> retriever
  -> LLM worker
  -> tool worker
  -> memory store
  -> evaluator
  -> result aggregator
```

只要系统里有多个 worker、多个服务、异步任务、失败重试、状态保存、并发请求，就已经进入分布式系统范畴。

分布式系统最核心的问题不是“怎么让很多机器一起干活”，而是：

- 任务怎么拆；
- 状态放哪里；
- 节点挂了怎么办；
- 重试会不会产生副作用；
- 多个副本如何保持一致；
- 如何在吞吐、延迟、可靠性之间取舍。

这些问题在 Ray、LLM serving、multi-agent、workflow engine 里都会反复出现。

## 2. 分布式系统的基本矛盾

单机程序默认有一个很强的假设：函数调用会返回，内存状态是本地的，失败通常是进程级别的。但分布式系统里这些假设都不成立。

你必须接受几个事实：

1. 网络请求可能丢、可能重复、可能乱序、可能超时；
2. 节点可能 crash，也可能只是变慢；
3. 远程调用不等于本地函数调用；
4. “没收到回复”不代表对方没执行；
5. 多个副本之间天然会出现状态差异；
6. 想要高可用，就必须设计重试和恢复；
7. 想要强一致，就必须付出通信和延迟成本。

这就是为什么分布式系统的关键词永远绕不开：partition、replication、consensus、idempotency、retry、timeout、lease、log、snapshot。

## 3. MapReduce 思想：把大任务拆成可重试的小任务

MapReduce 最重要的不是 `map` 和 `reduce` 两个函数，而是它背后的任务调度与容错思想。

一个大任务被拆成很多小任务：

```text
input files
  -> split
  -> map tasks
  -> shuffle by key
  -> reduce tasks
  -> output files
```

MapReduce 的经典例子是词频统计：

```python
def map(doc):
    for word in doc.split():
        emit(word, 1)


def reduce(word, counts):
    emit(word, sum(counts))
```

但系统层面真正关键的是：

- 输入数据被切成多个 split；
- master 负责给 worker 分配任务；
- worker 执行 map 或 reduce；
- map 输出中间文件；
- shuffle 把相同 key 的数据送到同一个 reduce；
- worker 挂了，master 重新调度任务；
- 慢 worker 会拖尾，master 可以 speculative execution。

MapReduce 的心法是：**让任务变成确定性的、可重试的、可重新调度的单元**。

## 4. MapReduce 为什么影响后来的系统

很多现代系统都继承了 MapReduce 的思想，只是形态变了。

| 系统 | MapReduce 式思想 |
|---|---|
| Spark | 把计算图拆成 stage 和 task，失败后按 lineage 重算 |
| Ray | 把 Python 函数变成远程 task，把对象放入 object store |
| Workflow engine | 把流程拆成 step，每步可重试、可恢复 |
| 多 agent 系统 | 把复杂任务拆给不同 agent，再聚合结果 |
| 分布式推理 | 把请求、batch、layer、token 分给不同 worker |

如果你理解了 MapReduce，就会自然理解为什么系统要有 driver/master、task queue、worker heartbeat、retry、checkpoint、shuffle、object store。

## 5. Ray：把 MapReduce 思想推广到通用任务图

Ray 可以理解成一个通用分布式执行框架。它不像 MapReduce 只适合 map 和 reduce，而是允许你把任意 Python 函数变成远程任务。

```python
@ray.remote
def f(x):
    return x * x

refs = [f.remote(i) for i in range(10)]
results = ray.get(refs)
```

这里的核心概念是：

- remote function：远程执行的任务；
- object ref：远程对象引用；
- scheduler：决定任务在哪个 worker 上跑；
- object store：保存任务输出，供下游任务读取；
- actor：有状态的远程 worker；
- fault tolerance：worker 挂了后重建或重跑任务。

Ray 的心智模型可以写成：

```text
driver 构建任务图
  -> scheduler 分配 task/actor
  -> worker 执行
  -> object store 保存中间结果
  -> 下游任务继续消费
```

这和 MapReduce 一脉相承，只是任务图更灵活。

## 6. Fault Tolerance：容错不是异常处理

容错不是写一个 `try/except`。分布式容错关心的是：系统某些部分失败时，整体还能不能给出正确或可接受的结果。

常见失败类型：

| 失败 | 例子 | 处理思路 |
|---|---|---|
| Crash failure | worker 进程挂了 | heartbeat + retry |
| Omission failure | 请求或响应丢了 | timeout + retry |
| Slow node | 某个 worker 很慢 | speculative execution / 剔除 |
| Network partition | 节点之间网络断开 | quorum / leader election |
| Data loss | 本地状态丢失 | replication / checkpoint |
| Duplicate execution | 重试导致执行两次 | idempotency / exactly-once 设计 |

最容易踩坑的是重试。重试看似简单，但如果操作有副作用，就可能出错。

例如：

```text
用户支付请求超时
  -> client 重试
  -> server 实际执行了两次扣款
```

所以很多系统要求操作具备幂等性：同一个请求执行多次，结果和执行一次一样。

常见做法：

- 给请求分配唯一 request id；
- 服务端记录已处理请求；
- 重复请求直接返回之前结果；
- 对外部副作用做事务或补偿。

## 7. Timeout：超时不是失败证明

分布式系统里，timeout 只能说明“我在规定时间内没收到回复”，不能说明对方没有执行。

```text
client -> server: put(x=1)
server 执行成功
response 在网络中丢失
client timeout
```

此时 client 如果重试，server 可能再次执行。因此所有可重试操作都要考虑幂等性。

这也是 workflow engine 经常保存 step 状态的原因：

```text
PENDING -> RUNNING -> SUCCEEDED / FAILED
```

状态持久化以后，系统重启或重试时才能知道某一步到底执行到哪里。

## 8. Raft：为什么需要共识

![Raft consensus](/images/posts/distributed-systems/raft.svg)

如果一个服务只有单副本，机器挂了就不可用。为了高可用，我们会复制多个副本。但复制带来一个问题：多个副本如何保持一致？

比如一个 KV store 有三个副本：

```text
node1: x = 1
node2: x = 1
node3: x = ?
```

如果客户端写入 `x = 2` 时 node1 成功、node2 成功、node3 网络断了，那么系统应该认为写成功吗？之后 node3 恢复时怎么追上？如果两个节点同时认为自己是 leader 怎么办？

Raft 解决的是：在不可靠网络和节点故障下，让多个节点对同一串日志达成一致。

## 9. Raft 的三个核心：Leader、Log、Majority

Raft 可以用一句话概括：**Leader 接收客户端写请求，把命令作为日志复制给 follower；当日志被多数派保存后，leader commit，并让所有节点按日志顺序应用到状态机**。

关键组件：

### 9.1 Leader

Raft 中任意时刻最多应该只有一个有效 leader。客户端写请求通常发给 leader。Leader 负责：

- 接收写请求；
- 追加本地 log；
- 发送 AppendEntries 给 followers；
- 等待多数派确认；
- 推进 commit index；
- 通知状态机 apply。

### 9.2 Follower

Follower 被动接收 leader 的日志复制和心跳。如果长时间收不到 leader 心跳，就会发起选举。

### 9.3 Candidate

Candidate 是选举过程中的临时角色。节点超时后变成 candidate，增加 term，向其他节点请求投票。如果拿到多数票，就成为 leader。

### 9.4 Log

Log 是 Raft 的核心数据结构。每条 log 包含：

```text
index, term, command
```

例如：

```text
1: term=1, put x=1
2: term=1, put y=2
3: term=2, delete x
```

状态机只按 committed log 的顺序执行命令。只要所有节点执行同一串 committed log，最终状态就一致。

### 9.5 Majority

Raft 不要求所有节点都成功，只要求多数派成功。

对于 3 个节点，多数派是 2；对于 5 个节点，多数派是 3。多数派的关键性质是：任意两个多数派集合一定有交集。

这个交集保证了 committed log 不会在下一任 leader 中消失。

## 10. Raft 写入流程

一次写入大概是：

```text
client -> leader: put(x=1)
leader: append log locally
leader -> followers: AppendEntries
followers: append log and ack
leader: receive majority ack
leader: commit log
leader: apply put(x=1) to state machine
leader -> client: success
followers: eventually commit and apply
```

重点是：写入不是直接改状态，而是先写日志，再提交，再 apply。

```text
command -> replicated log -> committed log -> state machine
```

这个结构非常重要。很多系统的可靠性都建立在 log 上：数据库 WAL、Kafka log、Raft log、workflow event history，本质上都是先记录事实，再从事实恢复状态。

## 11. Raft 选举流程

如果 follower 一段时间没收到 leader 心跳，会认为 leader 可能挂了，然后发起选举：

```text
follower timeout
  -> become candidate
  -> term += 1
  -> vote for self
  -> request votes from others
  -> majority votes -> become leader
```

为了减少多个节点同时选举导致冲突，Raft 使用随机 election timeout。这样不同节点超时点不同，更容易选出一个 leader。

Raft 把共识问题拆得比较容易理解：

- leader election：谁来当 leader；
- log replication：leader 如何复制日志；
- safety：已经 commit 的日志不能丢。

## 12. Distributed KV Store：分布式系统的最小形态

KV store 是理解分布式系统的最好载体。接口很简单：

```text
get(key) -> value
put(key, value)
delete(key)
```

但一旦分布式化，问题会立刻变复杂。

### 12.1 单副本 KV

最简单：

```text
client -> server -> memory dict
```

问题是 server 挂了就不可用，内存丢了数据也没了。

### 12.2 主从复制 KV

一个 leader 接收写请求，复制到 followers：

```text
client -> leader -> followers
```

读可以从 leader 读，也可以从 follower 读。读 follower 延迟低，但可能读到旧数据。

### 12.3 Raft KV

用 Raft 管理复制日志：

```text
put(x=1)
  -> Raft log
  -> majority commit
  -> apply to state machine dict
```

这样 KV store 的状态来自 committed log。节点挂了可以通过 log 重放恢复；新节点可以通过 snapshot + log catch up 加入。

## 13. Consistency：强一致、最终一致与读写取舍

分布式 KV store 必须面对一致性取舍。

### 13.1 Strong Consistency

写成功后，后续读一定能读到最新值。通常需要读写都经过 leader 或 quorum。

优点：语义简单。缺点：延迟更高，可用性更受网络影响。

### 13.2 Eventual Consistency

写入后不同副本可能短暂不一致，但如果没有新写入，最终会收敛。

优点：可用性和性能好。缺点：应用层要能接受读旧值。

### 13.3 Linearizability

Linearizability 是更严格的强一致：每个操作看起来像在某个瞬间原子生效，并且符合真实时间顺序。

Raft KV 通常追求 linearizable reads/writes，因为这让上层应用更容易推理。

## 14. Snapshot：日志不能无限长

Raft log 如果一直增长，会占用大量磁盘，恢复也很慢。因此系统需要 snapshot。

```text
log 1..100000 已经 apply 到状态机
  -> 保存 state machine snapshot
  -> 丢弃旧 log
  -> 后续只保留 snapshot 之后的 log
```

Snapshot 本质是 checkpoint。你会在很多系统里看到类似机制：

- 数据库 checkpoint；
- workflow 状态快照；
- Ray object spill / checkpoint；
- LLM agent 的 memory snapshot；
- 训练中的 model checkpoint。

思想都是一样的：不要每次从最早事件重放，定期保存一个可恢复状态。

## 15. Workflow Engine：分布式状态机

Workflow engine 例如 Temporal、Argo、Airflow，核心是把长流程拆成可恢复 step。

```text
step1: crawl papers
step2: parse PDFs
step3: run embedding
step4: retrieve related work
step5: draft report
step6: evaluate result
```

每个 step 都可能失败、超时、重试。Workflow engine 要保存：

- 当前执行到哪一步；
- 每一步输入输出；
- 哪些 step 已成功；
- 哪些 step 可以重试；
- 重试是否幂等；
- worker 挂了后谁接手。

这和 MapReduce master、Raft log、KV state 的思想是连在一起的：**用持久化状态描述系统进度，用可重试任务推进状态变化**。

## 16. 多 Agent 系统为什么是分布式系统

多 agent 系统表面上是多个 LLM 角色协作：planner、researcher、coder、reviewer、critic。但系统角度看，它就是一组 worker 和消息队列。

典型结构：

```text
planner agent
  -> creates tasks
research agents
  -> fetch evidence
coding agent
  -> edits files
review agent
  -> checks output
coordinator
  -> merges results and decides next step
```

它会遇到分布式系统经典问题：

- agent 输出不稳定，相当于 worker nondeterministic；
- tool call 失败，需要 retry；
- 多个 agent 可能写同一份状态，需要冲突解决；
- 任务执行时间不一致，需要调度和超时；
- 中间结果要存入 memory store；
- 最终结果要聚合，避免重复和矛盾；
- 如果 coordinator 挂了，要能恢复执行进度。

所以设计 multi-agent 不只是 prompt engineering，还需要 distributed systems thinking。

## 17. 分布式推理：模型服务里的分布式问题

分布式推理也不是简单“多放几张 GPU”。它同时涉及模型切分、请求调度和状态管理。

常见问题：

- 一个模型单卡放不下，需要 tensor parallel；
- 不同层放在不同 GPU，需要 pipeline parallel；
- 请求动态到达，需要 continuous batching；
- 每个请求有 KV cache，需要分配、迁移、释放；
- GPU worker 可能 OOM 或 crash，需要摘除和重试；
- speculative decoding 里 draft model 和 target model 要协作；
- 多副本服务要做负载均衡和健康检查。

这和分布式 KV store 的相似点是：系统里都有“状态”。KV store 的状态是 key-value；LLM serving 的状态是 request queue、KV cache、batch 状态、生成进度。

## 18. DeepScientist 为什么是 mini distributed system

DeepScientist 如果拆成系统组件，大概是：

```text
User Query
  -> Planner
  -> Search / Retrieval Workers
  -> PDF / Web Parser Workers
  -> Memory / Vector Store
  -> Draft Writer
  -> Critic / Evaluator
  -> Final Aggregator
```

这就是一个 mini distributed system：

| 分布式概念 | DeepScientist 对应物 |
|---|---|
| Master / coordinator | planner / orchestrator |
| Worker | searcher、parser、writer、critic |
| Task queue | 待检索、待阅读、待总结的子任务 |
| KV store | memory、cache、metadata store |
| MapReduce | 多路检索与证据聚合 |
| Fault tolerance | tool retry、fallback search、partial result |
| Checkpoint | 中间笔记、引用、草稿版本 |
| Consensus-like decision | 多 reviewer / evaluator 投票或打分 |
| Workflow log | agent trajectory / event history |

它不一定需要 Raft 这种强共识协议，但它一定需要分布式系统的思维：任务可拆、状态可恢复、失败可重试、结果可聚合、冲突可处理。

## 19. 读系统论文的心法

遇到 Ray、vLLM、workflow engine、多 agent 框架，可以按这几个问题拆：

1. 系统里的 coordinator 是谁？
2. worker 是无状态还是有状态？
3. 状态存在哪里，内存、磁盘、KV、object store，还是 log？
4. 任务失败后是重试、跳过、补偿，还是回滚？
5. 重试是否幂等？
6. 是否需要强一致，还是最终一致就够？
7. 调度目标是吞吐、延迟、公平性，还是成本？
8. 有没有 checkpoint / snapshot？
9. 有没有 straggler，如何处理慢节点？
10. 系统如何扩容和缩容？

这些问题比记住某个框架 API 更重要。

## 20. Week 3 学完应该掌握什么

学完这一周，目标不是能实现完整 Raft，而是能讲清楚：

- MapReduce 如何把大任务拆成可重试的小任务；
- master / worker / task queue / shuffle 各自解决什么问题；
- fault tolerance 为什么离不开 timeout、retry、idempotency、checkpoint；
- Raft 为什么需要 leader、log、term、majority；
- Raft 写入为什么先复制日志再 apply 状态机；
- distributed KV store 如何从单副本演化到 Raft 复制；
- 强一致和最终一致的取舍；
- Ray、workflow engine、多 agent、分布式推理分别对应哪些分布式模式；
- DeepScientist 为什么本质是 mini distributed system。

## 21. 最后总结

分布式系统的核心不是把代码部署到多台机器上，而是用系统化方式处理不可靠性：网络不可靠、节点不可靠、时间不可靠、状态不可靠。MapReduce 教我们如何拆任务和重试，Raft 教我们如何复制状态并达成一致，fault tolerance 教我们如何面对失败，distributed KV store 则把这些思想浓缩成最小可理解系统。

当你把这套心智模型带回 AI 系统，就会发现 Ray、分布式推理、多 agent、workflow engine 都不再神秘。它们都是在不同场景下回答同一组问题：任务怎么调度，状态怎么保存，失败怎么恢复，多个 worker 如何协作产出一个可靠结果。
