---
title: 链表与缓存高频题：LRU Cache、合并 K 个有序链表、反转链表
date: 2026-06-09
description: 面试高频算法题速记，整理 LRU Cache、合并 K 个有序链表、反转链表的核心思路、复杂度和 C++ 代码。
categories:
  - 技术
tags:
  - 算法
  - 链表
  - LRU
  - LeetCode
  - 面试
---

这篇整理三道面试特别高频的题：

- LRU Cache。
- 合并 K 个有序链表。
- 反转链表。

这三题都不难背代码，但很容易在面试里卡在边界处理上。准备时重点不是“记住每一行”，而是记住数据结构选择和指针移动顺序。

## LRU Cache

LRU 是 Least Recently Used，最近最少使用。

题目要求通常是：

```text
get(key)：如果 key 存在，返回 value，并把 key 标记为最近使用。
put(key, value)：插入或更新 key。如果容量超了，删除最久没被使用的 key。
get 和 put 都要 O(1)。
```

要做到 O(1)，需要两个结构配合：

| 结构 | 作用 |
|---|---|
| 双向链表 | 维护访问顺序，头部最近使用，尾部最久未使用 |
| 哈希表 | key -> 链表节点迭代器，支持 O(1) 查找 |

为什么不能只用哈希表？

因为哈希表只能快速查 key，不能快速知道哪个 key 最久没用。

为什么不能只用链表？

因为链表能维护顺序，但查找 key 是 O(n)。

所以答案是：

```text
unordered_map + list
```

### 代码

LeetCode 环境下不要加 `#define int long long`，否则可能改坏接口签名。

```cpp
#include <bits/stdc++.h>
using namespace std;

class LRUCache {
    int cap;
    list<pair<int, int>> q;
    unordered_map<int, list<pair<int, int>>::iterator> mp;

public:
    LRUCache(int capacity) {
        cap = capacity;
    }

    int get(int key) {
        if (!mp.count(key)) return -1;
        auto it = mp[key];
        int val = it->second;
        q.erase(it);
        q.push_front({key, val});
        mp[key] = q.begin();
        return val;
    }

    void put(int key, int value) {
        if (mp.count(key)) {
            q.erase(mp[key]);
        } else if ((int)q.size() == cap) {
            auto [old_key, old_val] = q.back();
            q.pop_back();
            mp.erase(old_key);
        }
        q.push_front({key, value});
        mp[key] = q.begin();
    }
};
```

### 复杂度

```text
get: O(1)
put: O(1)
空间复杂度: O(capacity)
```

### 易错点

1. `get` 命中后也要更新访问顺序。
2. `put` 更新已有 key 时，要先删除旧节点，再插入到链表头。
3. 淘汰尾部节点时，要同时从链表和哈希表中删除。
4. `list` 的迭代器在对应节点被 erase 后失效，要重新写入 `mp[key]`。

## 合并 K 个有序链表

题目：

```text
给你 k 个升序链表，把它们合并成一个升序链表。
```

最常用做法是小根堆。

思路：

```text
1. 把每个链表的头节点放入小根堆。
2. 每次取出堆顶最小节点，接到答案链表后面。
3. 如果这个节点还有 next，就把 next 放入堆。
4. 直到堆为空。
```

堆里最多有 k 个节点，所以每次 push / pop 是 `O(log k)`。

### 代码

```cpp
#include <bits/stdc++.h>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        auto cmp = [](ListNode* a, ListNode* b) {
            return a->val > b->val;
        };
        priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);

        for (auto node : lists) {
            if (node) pq.push(node);
        }

        ListNode dummy(0);
        ListNode* cur = &dummy;

        while (!pq.empty()) {
            auto node = pq.top();
            pq.pop();
            cur->next = node;
            cur = cur->next;

            if (node->next) pq.push(node->next);
        }

        return dummy.next;
    }
};
```

### 复杂度

设总节点数为 `N`，链表个数为 `k`。

```text
时间复杂度: O(N log k)
空间复杂度: O(k)
```

### 易错点

1. 堆里放的是节点指针，不是节点值。否则还要重新建节点。
2. 空链表不要放进堆。
3. `priority_queue` 默认是大根堆，所以比较函数要写成 `a->val > b->val`。
4. 用 dummy 节点可以避免处理头节点为空的特殊情况。

## 反转链表

题目：

```text
给定单链表头节点 head，反转整个链表，返回新的头节点。
```

核心就是三指针：

```text
prev：已经反转好的前半部分头节点
cur：当前正在处理的节点
nxt：提前保存 cur->next，避免断链后找不到后续节点
```

每轮做四件事：

```text
nxt = cur->next
cur->next = prev
prev = cur
cur = nxt
```

### 代码

```cpp
#include <bits/stdc++.h>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* cur = head;

        while (cur) {
            ListNode* nxt = cur->next;
            cur->next = prev;
            prev = cur;
            cur = nxt;
        }

        return prev;
    }
};
```

### 复杂度

```text
时间复杂度: O(n)
空间复杂度: O(1)
```

### 易错点

1. 一定要先保存 `nxt = cur->next`，再改 `cur->next`。
2. 最后返回 `prev`，不是 `cur`。循环结束时 `cur` 已经是 `nullptr`。
3. 空链表和单节点链表天然兼容，不需要单独判断。

## 递归版反转链表

递归版面试偶尔会追问。

思路：

```text
先反转 head->next 后面的链表
再让 head->next->next 指回 head
最后断开 head->next
```

代码：

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (!head || !head->next) return head;

        ListNode* new_head = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;

        return new_head;
    }
};
```

递归版时间复杂度也是 `O(n)`，但空间复杂度是递归栈 `O(n)`。面试中如果没有特殊要求，迭代版更稳。

## 三题总结

| 题目 | 核心数据结构 / 技巧 | 复杂度 |
|---|---|---|
| LRU Cache | `unordered_map + list` | `O(1)` |
| 合并 K 个有序链表 | 小根堆 | `O(N log k)` |
| 反转链表 | 三指针 | `O(n)` |

面试表达可以这样说：

> LRU 的关键是哈希表定位节点、双向链表维护访问顺序；合并 K 个有序链表用小根堆维护当前 k 个链表头节点，每次取最小；反转链表用 `prev / cur / nxt` 三个指针，先保存后继，再反转当前指针。

这三题都属于基础但高频的“不能错”题。代码最好能手写到没有停顿，尤其是 LRU 的哈希表和链表同步更新，以及反转链表的指针顺序。
