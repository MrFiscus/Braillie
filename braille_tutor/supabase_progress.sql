-- Run this once in the Supabase SQL editor to let the website keep each learner's braille progress in their account.
-- One row per signed-in learner; row level security means a learner can only ever read and write their own row.
-- (The website only needs it for the "kept in your account" sync: without it, progress is still kept on the tutor's computer.)

create table if not exists public.learning_progress (
  user_id    uuid primary key references auth.users (id) on delete cascade,
  data       jsonb not null default '{}'::jsonb,   -- the tutor's progress record (see braille_tutor/progress.py)
  updated_at timestamptz not null default now()
);

alter table public.learning_progress enable row level security;

drop policy if exists "read own progress" on public.learning_progress;
create policy "read own progress" on public.learning_progress
  for select using (auth.uid() = user_id);

drop policy if exists "insert own progress" on public.learning_progress;
create policy "insert own progress" on public.learning_progress
  for insert with check (auth.uid() = user_id);

drop policy if exists "update own progress" on public.learning_progress;
create policy "update own progress" on public.learning_progress
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
