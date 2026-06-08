---
title: Kafka 入门：生产者、Broker、消费者和“消费”到底是什么意思
date: 2026-06-09
description: 用推荐系统里的用户行为日志为例，讲清楚 Kafka 的作用、Producer、Broker、Consumer、Topic、Partition、Offset 和消费语义。
categories:
  - 技术
tags:
  - Kafka
  - 消息队列
  - 后端
  - 面试
  - 推荐系统
---

Kafka 可以理解成一个**高吞吐、可持久化、可横向扩展的消息中间件**。

最简单的结构是：

```text
Producer  ->  Kafka Broker  ->  Consumer
生产消息        存储消息           读取并处理消息
```

如果把 Kafka 想成一个物流系统：

- Producer 是发货的人。
- Broker 是中转仓库。
- Consumer 是收货并处理包裹的人。
- Topic 是不同类型货物的分类。
- Partition 是同一个分类下的多个货架。
- Offset 是货架上一件货物的编号。

## Kafka 的作用是什么

Kafka 最核心的作用是：**解耦、削峰填谷、持久化事件流**。

比如推荐系统里，用户会不断产生行为：

```text
曝光
点击
点赞
收藏
评论
关注
下单
```

这些行为不能每来一条就同步通知所有下游系统。否则推荐主链路会被拖慢，下游任何一个系统变慢，也会反过来影响用户请求。

更合理的方式是先把事件写入 Kafka：

```text
用户点击视频
  -> 日志服务把点击事件写入 Kafka
  -> 实时特征任务消费 Kafka，更新用户短期兴趣
  -> 数据仓库任务消费 Kafka，做离线分析
  -> 风控任务消费 Kafka，识别异常行为
  -> 模型训练任务消费 Kafka，生成训练样本
```

这样推荐服务只需要快速把事件写出去，不需要等待所有下游处理完。

这就是 Kafka 的价值：

- 解耦：生产者和消费者互相不依赖。
- 削峰：流量高峰时先堆在 Kafka，消费者按自己的能力慢慢处理。
- 持久化：消息会存到磁盘，消费者挂了也可以恢复后继续读。
- 广播给多系统：同一份用户行为日志可以被多个业务系统使用。

## 生产者 Producer 是什么

Producer 就是**生产消息、写入 Kafka 的系统**。

常见生产者：

- 推荐服务。
- 日志采集服务。
- 订单服务。
- 埋点 SDK。
- 后端业务服务。

一条用户点击消息可能长这样：

```json
{
  "user_id": 123,
  "item_id": 456,
  "event": "click",
  "timestamp": 1710000000
}
```

Producer 会把这条消息发送到某个 Kafka topic，比如：

```text
user_behavior_topic
```

一句话：

> Producer 负责把业务事件变成消息，并写入 Kafka。

## Broker 是什么

Broker 是 Kafka 的**服务器节点**。

一个 Kafka 集群通常由多个 Broker 组成：

```text
Kafka Cluster
  Broker 1
  Broker 2
  Broker 3
```

Broker 负责：

- 接收 Producer 发来的消息。
- 把消息写入磁盘。
- 按 Topic 和 Partition 管理消息。
- 给 Consumer 提供读取能力。
- 通过副本机制提高可靠性。

可以简单理解成：

```text
Broker = Kafka 的存储服务器 / 消息仓库
```

消息写进 Kafka 后，不会因为消费者读取了就立刻消失。Kafka 会根据保留策略保存一段时间，比如保存 7 天，或者保存到磁盘空间达到某个上限。

## 消费者 Consumer 是什么

Consumer 就是**从 Kafka 读取消息并处理消息的系统**。

常见消费者：

- Flink 实时任务。
- Spark Streaming 任务。
- 实时特征服务。
- 数据仓库同步任务。
- 风控任务。
- 搜索索引更新任务。
- 推荐训练样本构造任务。

比如实时特征服务消费点击事件：

```text
读到 user_id=123 点击 item_id=456
  -> 更新用户最近点击类目
  -> 更新用户短期兴趣
  -> 更新用户最近活跃时间
  -> 写入 Redis / 特征存储
```

一句话：

> Consumer 负责从 Kafka 读取消息，然后完成自己的业务逻辑。

## “消费”是什么意思

消费就是：**消费者从 Kafka 读取消息，并处理这些消息**。

比如 Kafka 里有三条消息：

```text
offset=0 用户 A 点击视频 1
offset=1 用户 B 点赞视频 2
offset=2 用户 A 收藏视频 3
```

消费者会按顺序读取并处理：

```text
读 offset=0
  -> 更新用户 A 的点击特征

读 offset=1
  -> 更新用户 B 的点赞特征

读 offset=2
  -> 更新用户 A 的收藏特征
```

这里要注意：**消费消息不等于删除消息**。

Kafka 的消息通常还会继续保留。消费者只是记录自己读到了哪里，这个位置叫 `offset`。

## Topic 是什么

Topic 是消息的逻辑分类。

比如：

```text
user_behavior_topic      用户行为日志
order_event_topic        订单事件
recommend_log_topic      推荐日志
payment_event_topic      支付事件
```

Producer 往某个 topic 写消息，Consumer 从某个 topic 读消息。

可以理解成：

```text
Topic = 一类消息的名字
```

## Partition 是什么

一个 topic 可以拆成多个 partition。

例如：

```text
user_behavior_topic
  partition 0
  partition 1
  partition 2
```

Partition 的作用是提高并发能力。

如果一个 topic 只有一个 partition，那么同一时刻能并行处理的能力有限。拆成多个 partition 后，Kafka 可以把数据分散到不同 Broker 上，消费者也可以并行消费。

但要注意：

> Kafka 只能保证单个 partition 内部有序，不能保证多个 partition 之间全局有序。

如果想保证同一个用户的行为有序，可以用 `user_id` 作为 key，让同一个用户的消息进入同一个 partition。

```text
key = user_id
```

这样用户 A 的行为会进入同一个 partition，在这个 partition 内按顺序消费。

## Offset 是什么

Offset 是消息在 partition 里的编号。

例如：

```text
partition 0:
  offset 0  用户 A 点击
  offset 1  用户 B 点赞
  offset 2  用户 C 收藏
```

Consumer 通过 offset 记录自己消费到哪里了。

如果消费者处理到 offset 100 后挂了，重启后可以从 offset 101 继续读。

这也是 Kafka 能支持失败恢复的原因之一。

## Consumer Group 是什么

Consumer Group 是一组消费者。

同一个 Consumer Group 里的多个消费者会共同消费一个 topic。

例如：

```text
user_behavior_topic 有 3 个 partition

Consumer Group: feature_job
  Consumer 1 消费 partition 0
  Consumer 2 消费 partition 1
  Consumer 3 消费 partition 2
```

这样可以提高消费速度。

同一个 partition 在同一个 Consumer Group 内，同一时刻只会分配给一个 Consumer。这样可以避免同一组里的多个消费者重复处理同一条消息。

但不同 Consumer Group 之间互不影响。

例如：

```text
feature_job 消费用户行为，更新实时特征
warehouse_job 消费用户行为，写入数仓
risk_job 消费用户行为，做风控检测
```

它们可以同时消费同一个 topic，各自维护自己的 offset。

## 推荐系统里的完整例子

以抖音推荐里的点击事件为例：

```text
用户点击视频
  -> 推荐服务生成 click event
  -> Producer 写入 user_behavior_topic
  -> Kafka Broker 持久化消息
  -> Flink Consumer 消费消息
  -> 更新用户实时兴趣特征
  -> 特征写入在线特征存储
  -> 下一次推荐请求使用新特征
```

这条链路里：

- 推荐服务是 Producer。
- Kafka 集群里的机器是 Broker。
- Flink 实时任务是 Consumer。
- 读取并处理点击事件的过程叫消费。
- `user_behavior_topic` 是 Topic。
- Topic 下面的多个分片是 Partition。
- 每条消息在 Partition 里的编号是 Offset。

## 面试怎么一句话总结

可以这样说：

> Kafka 是一个分布式消息队列，核心作用是把生产者和消费者解耦，并承接高吞吐事件流。Producer 负责写消息，Broker 负责存储和分发消息，Consumer 负责读取并处理消息。消费不是删除消息，而是消费者读取消息后提交 offset，记录自己处理到哪里。

放到推荐系统里：

> Kafka 常用于承接用户行为日志，把在线推荐、实时特征、离线数仓、模型训练、风控等系统解耦起来。

## 容易混淆的点

1. 消费不是删除消息。Kafka 消息会按保留策略保存，消费者只是提交 offset。
2. Topic 是逻辑分类，Partition 是物理分片。
3. 单个 Partition 内有序，多个 Partition 之间不保证全局有序。
4. 同一个 Consumer Group 内，一个 Partition 同时只会被一个 Consumer 消费。
5. 不同 Consumer Group 可以各自独立消费同一个 Topic。

把这些概念讲清楚之后，再去背 Kafka 为什么快、如何保证不丢、如何保证顺序，就会顺很多。
