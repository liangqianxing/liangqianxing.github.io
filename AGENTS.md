# Blog maintenance rules

This repository is the source of truth for both the legacy GitHub Pages profile and the Halo blog at `https://diycv.top`.

## Posts

- Store posts in `content/posts/<slug>.md`.
- Keep `hidden: true` so posts do not appear on the legacy GitHub Pages frontend.
- Use `haloPublished: false` for drafts and `haloPublished: true` for published Halo posts.
- Keep the original `date`, `categories`, `tags`, and `description` fields.
- Never delete a post to unpublish it. Set `haloPublished: false` instead.

Example front matter:

```yaml
---
title: Article title
date: 2026-07-22 14:00:00
description: Short summary.
categories:
  - Technology
tags:
  - AI
hidden: true
haloPublished: false
---
```

## Images

- Store article images below `public/images/posts/<slug>/`.
- Reference them as `/images/posts/<slug>/<filename>` in Markdown.
- Keep image filenames unique across the repository because Halo attachments use the filename as their identity.
- The Halo sync uploads referenced images to Halo storage and rewrites their URLs.

## Publishing

1. Create or edit the Markdown source.
2. Run `npm run build` to validate the legacy site.
3. Set `haloPublished: true` only when the article is ready.
4. Commit and push to `main`. GitHub Actions synchronizes the change to Halo.

Do not store the Halo token, server password, or other credentials in tracked files. The Halo token belongs only in the `HALO_PAT` GitHub Actions secret.
