---
title: 如何搭建自己的 codex 不限量号池
date: 2026-03-08 23:30:00
tags:
  - codex
  - api
  - docker
categories:
  - 技术实践
---

需要用到的：腾讯云服务器、代理、codex 注册机、API 系统。
整体均使用 `docker-compose` 管理。

> 您也可以使用我搭好的网站试一下效果：39 元一个月 2000 刀余额。
> [Codex 小店](https://pay.ldxp.cn/shop/TUQKNUNV)
> [Codex 中转站](http://81.70.32.82/)

## 云服务器

## 免费云服务器领取（0 元）

云服务器我是领取的腾讯云 2G。注册一个类似 VS Code 的 IDE 可以送一个月免费服务器，签到七天还可以延长两个月，部署时选择 `docker` 镜像。

免费云服务器可以在这领取：[codebuddy](https://www.codebuddy.ai/promotion/?ref=ylmssxl97sw06m5)

![免费云服务器活动页](/images/posts/codex-account-pool/cloud-server-offer.png)

![云服务器控制台](/images/posts/codex-account-pool/cloud-server-console.png)

由于国内站点无法直连 `chatgpt.com` 和 `api.openai.com`，所以我设置了一下连接规则。

## 代理（云服务器可以直连 chatgpt 不需要设置）

用 `Resin` 反代网络连接规则。

`Resin` 是用 `Go` 语言实现的一个代理池，可以直接从日常使用的套餐反代出可行节点。

![Resin 代理示意](/images/posts/codex-account-pool/resin-proxy.png)

## API 系统

### [New-API](https://github.com/QuantumNous/new-api)

现在大部分中间站都搭建在 `New-API`，可以一键导入号池、设置模型等。

![New-API 控制台](/images/posts/codex-account-pool/new-api-dashboard.png)

模型可以自己导入，我这边只导入了 OpenAI 的模型。

![模型导入配置 1](/images/posts/codex-account-pool/model-import-1.png)

![模型导入配置 2](/images/posts/codex-account-pool/model-import-2.png)

### CLI Proxy API

这个系统最方便的是可以看每个账号的额度，但是接入付费功能还是不如上面的 `new-api` 方便。

![CLI Proxy API](/images/posts/codex-account-pool/cli-proxy-api.jpg)

## 号池

号池搭建不便在这里分享。

可以在这里获取：`https://pay.ldxp.cn/item/ynvs99`

## 一天的收入

我第一天的收入是 99.5 + 66.66 = 166.16 元，基本上能覆盖住未来的一些费用。

![收入截图 1](/images/posts/codex-account-pool/daily-income-1.png)

![收入截图 2](/images/posts/codex-account-pool/daily-income-2.jpg)

如果您有问题可以和我本人联系：QQ 2264535298。
