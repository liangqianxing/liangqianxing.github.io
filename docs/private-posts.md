# 私密文章配置

私密文章使用 Supabase Auth 和 Row Level Security。正文只存放在 `blog_posts` 表中，不进入公开 Git 仓库或 GitHub Pages 静态 payload。

## 1. 创建 Supabase 项目

在 Supabase 创建项目后，打开 SQL Editor，执行：

```text
supabase/migrations/202607200001_private_blog.sql
```

## 2. 配置登录回调

在 Authentication → URL Configuration 中设置：

- Site URL：`https://liangqianxing.github.io`
- Redirect URLs：
  - `http://localhost:3000/auth/callback**`
  - `http://localhost:3001/auth/callback**`
  - `https://liangqianxing.github.io/auth/callback**`

建议先用管理员邮箱发送一次 Magic Link，让 Supabase 创建用户。然后在 SQL Editor 登记管理员：

```sql
insert into public.blog_admins (user_id)
select id
from auth.users
where lower(email) = lower('<ADMIN_EMAIL>')
on conflict (user_id) do nothing;
```

确认管理员能够登录后，可在 Authentication 设置中关闭公开注册；需要“登录可见”的读者时，通过邀请创建账号。

## 3. 本地环境变量

在 `.env` 中配置：

```bash
NUXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NUXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```

Anon Key 本来就会发送到浏览器，安全边界来自 RLS。绝不能把 Service Role Key 放进前端环境变量、GitHub Variables 或仓库。

## 4. GitHub Pages 变量

在仓库 Settings → Secrets and variables → Actions → Variables 中增加：

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`

推送 `main` 后，部署工作流会把它们注入 Nuxt 公共运行配置。

## 5. 使用入口

- `/vault`：当前账号可访问的托管文章。
- `/login`：Magic Link 登录。
- `/admin`：管理员创建文章并设置 `公开 / 登录可见 / 仅管理员`。

## 6. 迁移现有文章

`content/posts/*.md`、Git 历史和 `.output/public` 都是公开的。后台不能把已经生成的静态正文即时变成私密内容。

迁移步骤：

1. 在 `/admin` 创建托管副本并先设为“仅管理员”。
2. 登录 `/vault` 核对 Markdown、公式和图片。
3. 从 `content/posts` 删除公开源文件，同时检查 README、系列顺序和专属图片。
4. 重新运行 `npm run build` 并部署。

已经发布过的内容可能仍存在于 Git 历史、搜索引擎缓存或第三方存档中，无法通过权限开关追溯撤回。真正从未公开的文章应直接在 `/admin` 创建，不要先提交到公开仓库。
