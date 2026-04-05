---
title: GitHub Actions 入门：自动化你的工作流
date: 2026-04-04 09:00:00
tags:
  - GitHub Actions
  - CI/CD
  - 自动化
categories:
  - 技术笔记
---

GitHub Actions 是 GitHub 内置的 CI/CD 平台，可以在代码推送、PR 创建等事件触发时自动执行任务。本文介绍基本用法。

<!-- more -->

## 核心概念

- **Workflow**：工作流，定义在 `.github/workflows/*.yml`
- **Event**：触发条件，如 `push`、`pull_request`、`schedule`
- **Job**：工作流中的一个任务，运行在独立的虚拟机上
- **Step**：Job 中的单个步骤，可以是命令或 Action
- **Action**：可复用的步骤单元，来自 GitHub Marketplace 或自定义

## 最简单的例子

```yaml
name: Hello World

on: push

jobs:
  greet:
    runs-on: ubuntu-latest
    steps:
      - name: Say hello
        run: echo "Hello, GitHub Actions!"
```

## 常用触发条件

```yaml
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 9 * * 1'  # 每周一早上 9 点
  workflow_dispatch:       # 手动触发
```

## 实用示例：Node.js 项目 CI

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [18, 20]

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'

      - run: npm ci
      - run: npm test
```

## 使用 Secrets

敏感信息（Token、密钥）存在仓库的 Settings → Secrets 中，通过 `${{ secrets.MY_SECRET }}` 引用：

```yaml
- name: Deploy
  env:
    API_TOKEN: ${{ secrets.API_TOKEN }}
  run: ./deploy.sh
```

## 缓存依赖

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-
```

`actions/setup-node` 的 `cache: 'npm'` 参数会自动处理缓存，大多数情况下不需要手动配置。

## 常用 Actions

| Action | 用途 |
|--------|------|
| `actions/checkout` | 检出代码 |
| `actions/setup-node` | 配置 Node.js |
| `actions/setup-python` | 配置 Python |
| `actions/upload-artifact` | 上传构建产物 |
| `peaceiris/actions-gh-pages` | 部署到 GitHub Pages |

## 小技巧

**只在特定路径变更时触发：**

```yaml
on:
  push:
    paths:
      - 'src/**'
      - 'package.json'
```

**Job 之间传递数据：**

```yaml
jobs:
  build:
    outputs:
      version: ${{ steps.get-version.outputs.version }}
    steps:
      - id: get-version
        run: echo "version=$(node -p "require('./package.json').version")" >> $GITHUB_OUTPUT

  deploy:
    needs: build
    steps:
      - run: echo "Deploying version ${{ needs.build.outputs.version }}"
```

GitHub Actions 的文档很完善，遇到具体需求直接查 [官方文档](https://docs.github.com/en/actions) 或 [Marketplace](https://github.com/marketplace?type=actions) 找现成的 Action 即可。
