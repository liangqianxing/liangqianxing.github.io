create extension if not exists pgcrypto;

create table if not exists public.blog_admins (
  user_id uuid primary key references auth.users(id) on delete cascade,
  created_at timestamptz not null default now()
);

create table if not exists public.blog_posts (
  id uuid primary key default gen_random_uuid(),
  slug text not null unique check (slug ~ '^[a-z0-9]+(?:-[a-z0-9]+)*$'),
  title text not null check (char_length(title) between 1 and 180),
  description text not null default '',
  body_markdown text not null default '',
  tags text[] not null default '{}',
  categories text[] not null default '{}',
  visibility text not null default 'admin' check (visibility in ('public', 'authenticated', 'admin')),
  published_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create or replace function public.is_blog_admin()
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1
    from public.blog_admins
    where user_id = auth.uid()
  );
$$;

create or replace function public.touch_blog_post_updated_at()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists blog_posts_touch_updated_at on public.blog_posts;
create trigger blog_posts_touch_updated_at
before update on public.blog_posts
for each row execute function public.touch_blog_post_updated_at();

alter table public.blog_admins enable row level security;
alter table public.blog_posts enable row level security;

drop policy if exists "admins can read their membership" on public.blog_admins;
create policy "admins can read their membership"
on public.blog_admins
for select
to authenticated
using (user_id = auth.uid());

drop policy if exists "read posts by visibility" on public.blog_posts;
create policy "read posts by visibility"
on public.blog_posts
for select
to anon, authenticated
using (
  visibility = 'public'
  or (visibility = 'authenticated' and auth.uid() is not null)
  or public.is_blog_admin()
);

drop policy if exists "admins can insert posts" on public.blog_posts;
create policy "admins can insert posts"
on public.blog_posts
for insert
to authenticated
with check (public.is_blog_admin());

drop policy if exists "admins can update posts" on public.blog_posts;
create policy "admins can update posts"
on public.blog_posts
for update
to authenticated
using (public.is_blog_admin())
with check (public.is_blog_admin());

drop policy if exists "admins can delete posts" on public.blog_posts;
create policy "admins can delete posts"
on public.blog_posts
for delete
to authenticated
using (public.is_blog_admin());

revoke all on public.blog_admins from anon;
revoke all on public.blog_admins from authenticated;
grant select on public.blog_admins to authenticated;

revoke all on public.blog_posts from anon;
revoke all on public.blog_posts from authenticated;
grant select on public.blog_posts to anon, authenticated;
grant insert, update, delete on public.blog_posts to authenticated;

grant execute on function public.is_blog_admin() to anon, authenticated;
