CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DROP TABLE IF EXISTS chunk_embeddings CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS user_profiles CASCADE;

-- Documents
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  file_name TEXT NOT NULL,
  summary TEXT,
  body TEXT NOT NULL,
  topic TEXT,
  source_path TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Chunks
CREATE TABLE chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_idx INT NOT NULL,
  hpath TEXT,
  body TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- User profiles
CREATE TABLE user_profiles (
  user_id UUID PRIMARY KEY,
  role TEXT,
  level INT,
  preferences JSONB DEFAULT '{}'::jsonb,
  completed_material_ids UUID[],
  time_budget_default INT
);

-- Embeddings
CREATE TABLE chunk_embeddings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  embedding vector(1536),
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);
