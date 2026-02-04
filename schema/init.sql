-- ============================================================================
-- Trainer Agent - Esquema inicial (1536 dims + IVFFLAT)
-- Idempotente: elimina tablas/índices si existen y recrea todo limpio
-- ============================================================================

-- Extensiones necesarias
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- para gen_random_uuid()

-- Elimina índices/tablas si existen (orden seguro por dependencias)
DROP INDEX IF EXISTS material_chunks_embedding_ivf_idx;
DROP INDEX IF EXISTS material_chunks_embedding_hnsw_idx;

DROP TABLE IF EXISTS access_policies CASCADE;
DROP TABLE IF EXISTS user_profiles CASCADE;
DROP TABLE IF EXISTS material_chunks CASCADE;
DROP TABLE IF EXISTS materials CASCADE;

-- ============================================================================
-- Tabla de materiales (documentos “base”)
-- ============================================================================
CREATE TABLE materials (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  summary TEXT,
  body TEXT NOT NULL,
  modality TEXT CHECK (modality IN ('doc','video','lab','faq')) DEFAULT 'doc',
  product_area TEXT,
  pipeline_stage TEXT,           -- ej. pre-selection | selection | deploy | ...
  difficulty INT CHECK (difficulty BETWEEN 1 AND 5) DEFAULT 2,
  audience TEXT,                 -- ej. data-eng | ml-eng | analyst | admin
  prereqs TEXT[],
  duration_min INT,
  version TEXT,
  loe_date DATE,                 -- si aplica (IP/LoE)
  ip_3d JSONB,                   -- si aplica: {best:"", likely:"", low:""}
  source_url TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================================
-- Chunks para RAG (vector 1536)
-- ============================================================================
CREATE TABLE material_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  material_id UUID NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
  chunk_idx INT NOT NULL,
  hpath TEXT,                    -- jerarquía (H1/H2/H3) opcional
  text TEXT NOT NULL,
  embedding VECTOR(1536)         -- **IMPORTANTE**: 1536 dims para text-embedding-3-small
);

-- Índice vectorial IVFFLAT (válido hasta 2000 dims)
CREATE INDEX IF NOT EXISTS material_chunks_embedding_ivf_idx
  ON material_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Búsqueda por similitud de texto (trigram) en materiales
CREATE INDEX IF NOT EXISTS materials_body_trgm_idx
  ON materials USING GIN (body gin_trgm_ops);

CREATE INDEX IF NOT EXISTS materials_summary_trgm_idx
  ON materials USING GIN (summary gin_trgm_ops);

-- ============================================================================
-- Perfiles de usuario (para preferencias y contexto)
-- ============================================================================
CREATE TABLE user_profiles (
  user_id UUID PRIMARY KEY,
  role TEXT,
  level INT,
  preferences JSONB,
  completed_material_ids UUID[],
  product_version TEXT,
  time_budget_default INT
);

-- ============================================================================
-- Políticas de acceso (ACL) por material
-- ============================================================================
CREATE TABLE access_policies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  material_id UUID NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
  allowed_roles TEXT[],          -- p.ej. ['data-eng','admin']
  classification TEXT CHECK (classification IN ('public','internal','restricted')) DEFAULT 'public'
);

-- Opcional: pequeñas ayudas
-- Evita duplicado (material_id, chunk_idx)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'uq_material_chunks_material_idx'
  ) THEN
    ALTER TABLE material_chunks
      ADD CONSTRAINT uq_material_chunks_material_idx UNIQUE (material_id, chunk_idx);
  END IF;
END$$;

-- Trigger simple para updated_at en materials (opcional)
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_materials_updated ON materials;
CREATE TRIGGER trg_materials_updated
BEFORE UPDATE ON materials
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
