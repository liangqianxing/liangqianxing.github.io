# Blog maintenance rules

This repository is the source of truth for Enhao Gu's personal homepage, academic profile, and technical blog at `https://liangqianxing.github.io`.

## Posts

- Store posts in `content/posts/<slug>.md`.
- Set `hidden: false` and `draft: false` for public posts on the blog.
- Keep `date`, `categories`, `tags`, `description`, `cover`, and `coverAlt` fields accurate.
- Never delete a post to unpublish it. Set `hidden: true` or `draft: true` instead.

Example front matter:

```yaml
---
title: Article title
date: '2026-10-23'
description: Short summary.
cover: /images/posts/<slug>/<cover-filename>.webp
coverAlt: Concise description of the cover illustration
categories:
  - AI
tags:
  - LLM
  - Agent
hidden: false
---
```

## Images

- Store article images below `public/images/posts/<slug>/`.
- Reference them as `/images/posts/<slug>/<filename>` in Markdown.

## Publishing

1. Create or edit the Markdown source in `content/posts/`.
2. Run `npm run build` to validate static site generation.
3. Commit and push to `main`. GitHub Actions builds and deploys `.output/public` to GitHub Pages.
