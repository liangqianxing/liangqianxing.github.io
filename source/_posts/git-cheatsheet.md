---
title: Git 常用操作备忘
date: 2026-04-05 14:00:00
tags:
  - Git
  - 工具
categories:
  - 技术笔记
---

记录一些 Git 日常高频操作，方便查阅。

<!-- more -->

## 基础操作

```bash
# 初始化仓库
git init

# 克隆远程仓库
git clone <url>

# 查看状态
git status

# 暂存所有修改
git add .

# 提交
git commit -m "feat: add new feature"

# 推送
git push origin main
```

## 分支管理

```bash
# 创建并切换分支
git checkout -b feature/my-feature

# 查看所有分支
git branch -a

# 合并分支
git merge feature/my-feature

# 删除本地分支
git branch -d feature/my-feature

# 删除远程分支
git push origin --delete feature/my-feature
```

## 撤销操作

```bash
# 撤销工作区修改（未暂存）
git checkout -- <file>

# 取消暂存
git reset HEAD <file>

# 撤销最近一次提交（保留修改）
git reset --soft HEAD~1

# 撤销最近一次提交（丢弃修改）
git reset --hard HEAD~1
```

## 查看历史

```bash
# 简洁日志
git log --oneline --graph --all

# 查看某个文件的修改历史
git log -p <file>

# 查看某次提交的改动
git show <commit-hash>
```

## 暂存工作区（stash）

```bash
# 暂存当前修改
git stash

# 查看 stash 列表
git stash list

# 恢复最近一次 stash
git stash pop

# 恢复指定 stash
git stash apply stash@{2}
```

## 远程仓库

```bash
# 查看远程仓库
git remote -v

# 添加远程仓库
git remote add origin <url>

# 拉取远程更新（不合并）
git fetch origin

# 拉取并合并
git pull origin main

# 强制推送（谨慎使用）
git push --force-with-lease origin main
```

## Commit 规范

推荐使用 [Conventional Commits](https://www.conventionalcommits.org/)：

| 前缀 | 含义 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档更新 |
| `style` | 代码格式（不影响逻辑） |
| `refactor` | 重构 |
| `chore` | 构建/工具链相关 |

示例：`git commit -m "fix: resolve login redirect issue"`
