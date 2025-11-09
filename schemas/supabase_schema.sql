-- Free Asset Pipeline - Supabase Database Schema
-- This schema provides asset metadata storage with vector search capabilities

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm; -- For fuzzy text search

-- Asset metadata table with vector support for semantic search
CREATE TABLE IF NOT EXISTS asset_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    type TEXT NOT NULL,
    url TEXT NOT NULL,
    thumbnail_url TEXT,
    title TEXT,
    description TEXT,
    tags TEXT[],
    license TEXT,
    duration INTEGER,
    width INTEGER,
    height INTEGER,
    file_size BIGINT,
    search_keywords TEXT[],
    embedding VECTOR(1536), -- For semantic search with OpenAI embeddings
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    download_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMPTZ
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_asset_source_type ON asset_metadata(source, type);
CREATE INDEX IF NOT EXISTS idx_asset_type ON asset_metadata(type);
CREATE INDEX IF NOT EXISTS idx_asset_tags ON asset_metadata USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_asset_keywords ON asset_metadata USING GIN(search_keywords);
CREATE INDEX IF NOT EXISTS idx_asset_embedding ON asset_metadata USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_asset_created ON asset_metadata(created_at DESC);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_asset_fts ON asset_metadata USING GIN(
    to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(description, ''))
);

-- Fuzzy text search on title
CREATE INDEX IF NOT EXISTS idx_asset_title_trgm ON asset_metadata USING GIN(title gin_trgm_ops);

-- User collections for organizing assets
CREATE TABLE IF NOT EXISTS asset_collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for user collections
CREATE INDEX IF NOT EXISTS idx_collections_user ON asset_collections(user_id);
CREATE INDEX IF NOT EXISTS idx_collections_public ON asset_collections(is_public) WHERE is_public = TRUE;

-- Many-to-many relationship between collections and assets
CREATE TABLE IF NOT EXISTS collection_assets (
    collection_id UUID REFERENCES asset_collections(id) ON DELETE CASCADE,
    asset_id UUID REFERENCES asset_metadata(id) ON DELETE CASCADE,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    notes TEXT,
    PRIMARY KEY (collection_id, asset_id)
);

-- Index for collection_assets
CREATE INDEX IF NOT EXISTS idx_collection_assets_collection ON collection_assets(collection_id);
CREATE INDEX IF NOT EXISTS idx_collection_assets_asset ON collection_assets(asset_id);

-- Asset usage tracking
CREATE TABLE IF NOT EXISTS asset_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID REFERENCES asset_metadata(id) ON DELETE CASCADE,
    user_id UUID,
    project_id UUID,
    usage_type TEXT, -- 'download', 'view', 'embed', etc.
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

-- Index for usage tracking
CREATE INDEX IF NOT EXISTS idx_asset_usage_asset ON asset_usage(asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_usage_user ON asset_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_asset_usage_created ON asset_usage(created_at DESC);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at
CREATE TRIGGER update_asset_metadata_updated_at
    BEFORE UPDATE ON asset_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_asset_collections_updated_at
    BEFORE UPDATE ON asset_collections
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function for semantic search using vector similarity
CREATE OR REPLACE FUNCTION search_assets_by_embedding(
    query_embedding VECTOR(1536),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10,
    filter_type TEXT DEFAULT NULL,
    filter_source TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    asset_id TEXT,
    source TEXT,
    type TEXT,
    url TEXT,
    thumbnail_url TEXT,
    title TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        am.id,
        am.asset_id,
        am.source,
        am.type,
        am.url,
        am.thumbnail_url,
        am.title,
        1 - (am.embedding <=> query_embedding) AS similarity
    FROM asset_metadata am
    WHERE
        (filter_type IS NULL OR am.type = filter_type) AND
        (filter_source IS NULL OR am.source = filter_source) AND
        am.embedding IS NOT NULL AND
        1 - (am.embedding <=> query_embedding) > match_threshold
    ORDER BY am.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function for full-text search
CREATE OR REPLACE FUNCTION search_assets_fulltext(
    search_query TEXT,
    match_count INT DEFAULT 20,
    filter_type TEXT DEFAULT NULL,
    filter_source TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    asset_id TEXT,
    source TEXT,
    type TEXT,
    url TEXT,
    thumbnail_url TEXT,
    title TEXT,
    rank FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        am.id,
        am.asset_id,
        am.source,
        am.type,
        am.url,
        am.thumbnail_url,
        am.title,
        ts_rank(
            to_tsvector('english', COALESCE(am.title, '') || ' ' || COALESCE(am.description, '')),
            plainto_tsquery('english', search_query)
        ) AS rank
    FROM asset_metadata am
    WHERE
        (filter_type IS NULL OR am.type = filter_type) AND
        (filter_source IS NULL OR am.source = filter_source) AND
        to_tsvector('english', COALESCE(am.title, '') || ' ' || COALESCE(am.description, ''))
        @@ plainto_tsquery('english', search_query)
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Row Level Security (RLS) policies
-- Enable RLS on tables
ALTER TABLE asset_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE asset_collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE collection_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE asset_usage ENABLE ROW LEVEL SECURITY;

-- Policy: Anyone can read public asset metadata
CREATE POLICY "Public assets are viewable by everyone"
    ON asset_metadata FOR SELECT
    USING (true);

-- Policy: Only service role can insert/update asset metadata
CREATE POLICY "Service role can manage assets"
    ON asset_metadata FOR ALL
    USING (auth.role() = 'service_role');

-- Policy: Users can only access their own collections
CREATE POLICY "Users can view their own collections"
    ON asset_collections FOR SELECT
    USING (auth.uid() = user_id OR is_public = true);

CREATE POLICY "Users can manage their own collections"
    ON asset_collections FOR ALL
    USING (auth.uid() = user_id);

-- Policy: Users can only access assets in their collections
CREATE POLICY "Users can view collection assets"
    ON collection_assets FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM asset_collections
            WHERE id = collection_id
            AND (user_id = auth.uid() OR is_public = true)
        )
    );

CREATE POLICY "Users can manage their collection assets"
    ON collection_assets FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM asset_collections
            WHERE id = collection_id AND user_id = auth.uid()
        )
    );

-- Policy: Users can view their own usage data
CREATE POLICY "Users can view their own usage"
    ON asset_usage FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can create usage records"
    ON asset_usage FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Comments for documentation
COMMENT ON TABLE asset_metadata IS 'Stores metadata for all cached assets from external sources';
COMMENT ON TABLE asset_collections IS 'User-created collections for organizing assets';
COMMENT ON TABLE collection_assets IS 'Many-to-many relationship between collections and assets';
COMMENT ON TABLE asset_usage IS 'Tracks asset usage for analytics and optimization';

COMMENT ON FUNCTION search_assets_by_embedding IS 'Performs semantic search using vector embeddings';
COMMENT ON FUNCTION search_assets_fulltext IS 'Performs full-text search on asset titles and descriptions';
