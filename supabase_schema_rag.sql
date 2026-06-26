drop index if exists summarizations_embedding_idx;
alter table summarizations drop column if exists embedding;
alter table summarizations add column embedding vector(1536);

create index summarizations_embedding_idx
  on summarizations
  using hnsw (embedding vector_cosine_ops);

drop function if exists match_summarizations;
create or replace function match_summarizations(
  query_embedding  vector(1536),
  match_project_id uuid,
  match_count      int default 3
)
returns table (
  id               uuid,
  meeting_title    text,
  summary          text,
  key_takeaways    jsonb,
  action_items     jsonb,
  created_at       timestamptz,
  similarity       float
)
language sql stable
as $$
  select
    s.id,
    s.meeting_title,
    s.summary,
    s.key_takeaways,
    s.action_items,
    s.created_at,
    1 - (s.embedding <=> query_embedding) as similarity
  from summarizations s
  where s.project_id = match_project_id
    and s.embedding is not null
  order by s.embedding <=> query_embedding
  limit match_count;
$$;

grant execute on function match_summarizations to anon;
