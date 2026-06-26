-- Run this in the Supabase SQL Editor (Dashboard → SQL Editor → New query)

create table if not exists projects (
  id          uuid primary key default gen_random_uuid(),
  name        text not null,
  description text not null default '',
  created_at  timestamptz not null default now()
);

create table if not exists summarizations (
  id                uuid primary key default gen_random_uuid(),
  project_id        uuid not null references projects(id) on delete cascade,
  meeting_title     text not null,
  language          text not null default 'en',
  detected_language text not null default 'en',
  transcript_chars  integer not null default 0,
  summary           text not null,
  key_takeaways     jsonb not null default '[]',
  action_items      jsonb not null default '[]',
  created_at        timestamptz not null default now()
);

-- Allow full access via anon key (no auth required for this POC)
alter table projects      enable row level security;
alter table summarizations enable row level security;

create policy "public read projects"       on projects      for select using (true);
create policy "public insert projects"     on projects      for insert with check (true);
create policy "public delete projects"     on projects      for delete using (true);

create policy "public read summarizations"   on summarizations for select using (true);
create policy "public insert summarizations" on summarizations for insert with check (true);
create policy "public delete summarizations" on summarizations for delete using (true);
