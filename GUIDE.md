# Complete Guide: Free Asset Pipeline for AI Content Creation (2025 Edition)

## Table of Contents

1. [Comprehensive Asset Sources & API Access](#1-comprehensive-asset-sources--api-access)
2. [API Integration Patterns](#2-api-integration-patterns)
3. [Model Context Protocol (MCP) Server Integration](#3-model-context-protocol-mcp-server-integration)
4. [Next.js + Supabase Integration Architecture](#4-nextjs--supabase-integration-architecture)
5. [AI Agent Orchestration Workflows](#5-ai-agent-orchestration-workflows)
6. [Deployment & Production Optimization](#6-deployment--production-optimization)
7. [Google Vertex AI Agent Integration](#7-google-vertex-ai-agent-integration)
8. [Complete End-to-End Example](#8-complete-end-to-end-example)

---

## Overview

Building an automated pipeline for sourcing, indexing, and integrating free assets (images, videos, audio, GIFs, SVGs) into your AI agentic content creation workflow is essential for scaling your production without licensing overhead. This guide provides production-ready implementation patterns, API integrations, and orchestration strategies specifically tailored for your tech stack (Next.js, Supabase, Vercel, Google Cloud).

## 1. Comprehensive Asset Sources & API Access

### Images & SVG Vectors

#### Pexels

- **License**: Free (attribution suggested, not required)
- **API Access**: ✅ Full REST API
- **Rate Limits**: 200 requests/hour, 20,000/month (default)
- **Unlimited Access**: Free for eligible apps - email api@pexels.com with platform details and attribution proof
- **Best For**: High-quality photos + videos in one API
- **Documentation**: https://help.pexels.com/hc/en-us/articles/900005852323

#### Pixabay

- **License**: CC0-like (free for commercial use)
- **API Access**: ✅ Full REST API with comprehensive filtering
- **Rate Limits**: 5,000 requests/hour (production)
- **Best For**: Large library with images, videos, vectors, and illustrations
- **Documentation**: https://pixabay.com/api/docs/

#### Unsplash

- **License**: Free with attribution required
- **API Access**: ✅ OAuth2 + API key authentication
- **Rate Limits**: 50 requests/hour (demo), upgrade available
- **Best For**: High-resolution artistic photography
- **Documentation**: https://unsplash.com/documentation

#### Openverse

- **License**: CC0 / CC-BY aggregator
- **API Access**: ✅ Unified search across 800M+ assets
- **Best For**: Cross-platform search (images + audio) with clear licensing metadata
- **Documentation**: https://api.openverse.org

### Audio & Sound Effects

#### Freesound

- **License**: CC0 / CC-BY user-uploaded content
- **API Access**: ✅ RESTful API with OAuth2
- **Features**: Content-based similarity search, advanced metadata filtering
- **Rate Limits**: API key based (generous for non-commercial)
- **Documentation**: https://freesound.org/docs/api/

#### Free Music Archive

- **License**: CC-BY / CC0 music tracks
- **API Access**: ✅ Search by genre, mood, instrument
- **Best For**: Background music for video content

#### Openverse Audio

- **License**: CC0 / CC-BY
- **API Access**: ✅ Same unified API as images
- **Best For**: Quick audio searches with licensing clarity

### Video & GIFs

#### Pexels Video

- **License**: Free (same as Pexels photos)
- **API Access**: ✅ Same API endpoint as images
- **Best For**: Professional stock video clips

#### Pixabay Video

- **License**: CC0
- **API Access**: ✅ Integrated with main Pixabay API
- **Best For**: B-roll footage

#### Giphy

- **License**: Free for non-commercial use
- **API Access**: ✅ Robust API with search, trending, stickers
- **Rate Limits**: 100 requests/hour (beta), 21,000/hour + 50,000/day (production)
- **Best For**: GIF animations and stickers
- **Documentation**: https://developers.giphy.com/docs/

## 2. API Integration Patterns

### Direct API Implementation (Python)

See [examples/python/asset_api.py](./examples/python/asset_api.py) for the complete implementation.

**Key Features:**
- Unified interface for multiple asset sources
- Automatic error handling and retry logic
- Response normalization across different APIs
- Type hints for better IDE support

```python
import os
import requests
from typing import List, Dict

class AssetAPI:
    def __init__(self):
        self.pexels_key = os.getenv("PEXELS_API_KEY")
        self.unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY")
        self.pixabay_key = os.getenv("PIXABAY_API_KEY")
        self.freesound_key = os.getenv("FREESOUND_API_KEY")
        self.giphy_key = os.getenv("GIPHY_API_KEY")

    def search_pexels_images(self, query: str, per_page: int = 15) -> List[Dict]:
        """Search Pexels for images with caching"""
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": self.pexels_key}
        params = {"query": query, "per_page": per_page}

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return [{
                "id": photo["id"],
                "url": photo["src"]["original"],
                "thumbnail": photo["src"]["medium"],
                "photographer": photo["photographer"],
                "source": "pexels",
                "type": "image",
                "width": photo["width"],
                "height": photo["height"]
            } for photo in data.get("photos", [])]
        return []
```

### Local SQLite Caching Layer

See [examples/python/asset_cache.py](./examples/python/asset_cache.py) for the complete implementation.

**Benefits:**
- Reduces API calls and costs
- Improves response times
- Works offline with cached data
- Automatic expiration management

```python
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict

class AssetCache:
    def __init__(self, db_path: str = "assets_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.setup_database()

    def setup_database(self):
        """Initialize cache database with metadata support"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                type TEXT NOT NULL,
                query TEXT NOT NULL,
                url TEXT NOT NULL,
                thumbnail TEXT,
                metadata TEXT,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_type
            ON assets(query, type, source)
        """)

        self.conn.commit()
```

## 3. Model Context Protocol (MCP) Server Integration

### Python MCP Server for Asset Management

See [examples/mcp-server/asset_mcp_server.py](./examples/mcp-server/asset_mcp_server.py) for the complete implementation.

**Features:**
- FastMCP-based server for easy integration
- Multiple tool definitions for different asset types
- Built-in caching support
- Resource endpoints for monitoring

```python
from fastmcp import FastMCP
from typing import Dict, List

# Initialize MCP server
mcp = FastMCP(name="AssetSearchServer")

# Initialize API and cache clients
api_client = AssetAPI()
cache = AssetCache()

@mcp.tool()
def search_images(query: str, sources: List[str] = ["pexels", "pixabay"]) -> Dict:
    """
    Search for royalty-free images across multiple sources.

    Args:
        query: Search keywords
        sources: List of sources to search (pexels, pixabay, unsplash)

    Returns:
        Dictionary with results from each source
    """
    # Check cache first
    cached = cache.get_cached_assets(query, "image")
    if cached:
        return {"cached": True, "results": cached}

    results = []

    if "pexels" in sources:
        results.extend(api_client.search_pexels_images(query))

    if "pixabay" in sources:
        results.extend(api_client.search_pixabay_images(query))

    if "unsplash" in sources:
        results.extend(api_client.search_unsplash_images(query))

    # Cache results
    cache.cache_assets(query, results)

    return {"cached": False, "count": len(results), "results": results}
```

### MCP Configuration for Cursor/VSCode

Add to your `.cursor/mcp.json` or Claude Desktop config:

```json
{
  "mcpServers": {
    "asset-search": {
      "command": "python",
      "args": ["-m", "uvx", "run", "asset_mcp_server.py"],
      "env": {
        "PEXELS_API_KEY": "${PEXELS_API_KEY}",
        "PIXABAY_API_KEY": "${PIXABAY_API_KEY}",
        "UNSPLASH_ACCESS_KEY": "${UNSPLASH_ACCESS_KEY}",
        "FREESOUND_API_KEY": "${FREESOUND_API_KEY}",
        "GIPHY_API_KEY": "${GIPHY_API_KEY}"
      }
    }
  }
}
```

## 4. Next.js + Supabase Integration Architecture

### Supabase Schema for Asset Metadata

See [schemas/supabase_schema.sql](./schemas/supabase_schema.sql) for the complete schema.

**Features:**
- Vector embeddings for semantic search
- Full-text search capabilities
- User collections for organizing assets
- Optimized indexes for performance

```sql
-- Asset metadata table with vector support
CREATE TABLE asset_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id TEXT NOT NULL,
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
    metadata JSONB
);

-- Indexes for performance
CREATE INDEX idx_asset_source_type ON asset_metadata(source, type);
CREATE INDEX idx_asset_tags ON asset_metadata USING GIN(tags);
CREATE INDEX idx_asset_keywords ON asset_metadata USING GIN(search_keywords);
CREATE INDEX idx_asset_embedding ON asset_metadata USING ivfflat(embedding vector_cosine_ops);

-- Full-text search index
CREATE INDEX idx_asset_fts ON asset_metadata USING GIN(
    to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(description, ''))
);
```

### Next.js API Routes

See [examples/nextjs/app/api/assets/search/route.ts](./examples/nextjs/app/api/assets/search/route.ts) for the complete implementation.

**Features:**
- Supabase caching layer
- Parallel API calls
- Automatic result storage
- Error handling

### React Component with Optimized Image Loading

See [examples/nextjs/components/AssetGallery.tsx](./examples/nextjs/components/AssetGallery.tsx) for the complete implementation.

**Features:**
- Next.js Image optimization
- Lazy loading
- Responsive grid layout
- Loading states

## 5. AI Agent Orchestration Workflows

### LangChain Agent with Asset Tools

See [examples/ai-agents/langchain_agent.py](./examples/ai-agents/langchain_agent.py) for the complete implementation.

**Features:**
- OpenAI Functions agent
- Custom tool definitions
- Conversational interface
- Structured output

### AutoGen Multi-Agent Content Pipeline

See [examples/ai-agents/autogen_workflow.py](./examples/ai-agents/autogen_workflow.py) for the complete implementation.

**Features:**
- Multi-agent collaboration
- Specialized roles (researcher, fetcher, assembler)
- Group chat coordination
- Automated workflow execution

### CrewAI Production Workflow

See [examples/ai-agents/crewai_workflow.py](./examples/ai-agents/crewai_workflow.py) for the complete implementation.

**Features:**
- Task-based workflow
- Agent specialization
- Sequential processing
- Custom tools integration

## 6. Deployment & Production Optimization

### Docker Compose for Local Development

See [examples/deployment/docker-compose.yml](./examples/deployment/docker-compose.yml) for the complete configuration.

**Services:**
- MCP server
- Next.js application
- Asset processor

### Vercel Deployment Configuration

See [examples/deployment/vercel.json](./examples/deployment/vercel.json) for the complete configuration.

**Features:**
- Optimized function settings
- Image optimization configuration
- Environment variable management
- Remote pattern support

### Performance Optimization Tips

#### Image Optimization with Vercel

- Use `next/image` component for automatic optimization
- Vercel optimizes images on-demand without slowing builds
- Supports automatic format conversion (WebP/AVIF)
- Implements lazy loading by default

#### Caching Strategy

- Cache API responses for 24 hours in SQLite
- Use Supabase for long-term asset metadata storage
- Implement Redis/Upstash for distributed caching in production
- Normalize search queries to maximize cache hits

#### Rate Limit Management

- Implement request queuing for API calls
- Use exponential backoff for 429 responses
- Batch requests where APIs support it
- Monitor usage with custom logging

## 7. Google Vertex AI Agent Integration

See [examples/ai-agents/vertex_ai_agent.py](./examples/ai-agents/vertex_ai_agent.py) for the complete implementation.

**Features:**
- Vertex AI Reasoning Engine
- Gemini model integration
- Custom tool deployment
- Cloud-native architecture

## 8. Complete End-to-End Example

See [examples/python/content_pipeline.py](./examples/python/content_pipeline.py) for the complete implementation.

This example demonstrates a complete content production pipeline:

1. **Brief Analysis**: Use LLM to extract asset requirements
2. **Parallel Asset Search**: Fetch images, videos, and audio simultaneously
3. **Download & Organization**: Save assets locally with proper structure
4. **Metadata Storage**: Store in Supabase for future retrieval

**Pipeline Steps:**

```python
async def generate_social_video_assets(
    brief: str,
    style: str = "modern",
    duration: int = 60
) -> Dict:
    # Step 1: Analyze brief with LLM
    requirements = await analyze_brief(brief, duration)

    # Step 2: Search for assets in parallel
    images, videos, audio = await asyncio.gather(
        fetch_images(requirements["image_queries"]),
        fetch_videos(requirements["video_queries"]),
        fetch_audio(requirements["audio_queries"])
    )

    # Step 3: Download and organize
    manifest = await download_and_organize({
        "images": images,
        "videos": videos,
        "audio": audio
    })

    # Step 4: Store in Supabase
    await store_in_supabase(manifest)

    return manifest
```

## Conclusion

This complete pipeline enables you to:

✅ **Automatically source** free assets from 10+ APIs
✅ **Cache efficiently** with SQLite + Supabase for fast retrieval
✅ **Integrate seamlessly** with Next.js, Vercel, and Google Cloud
✅ **Orchestrate intelligently** using LangChain, AutoGen, or CrewAI agents
✅ **Scale to production** with proper rate limiting and error handling
✅ **Support MCP** for Cursor AI and Claude Desktop integration

The architecture is designed for CineFilm.tech platform and scales from local development to enterprise production deployment.

## References

- [Pexels API Documentation](https://help.pexels.com/hc/en-us/articles/900005852323)
- [Pixabay API Documentation](https://pixabay.com/api/docs/)
- [Unsplash API Documentation](https://unsplash.com/documentation)
- [Freesound API Documentation](https://freesound.org/docs/api/)
- [Giphy API Documentation](https://developers.giphy.com/docs/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Next.js Image Optimization](https://vercel.com/docs/image-optimization)
- [Supabase Vector Database](https://supabase.com/features/vector-database)
- [LangChain Documentation](https://docs.langchain.com/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [CrewAI Documentation](https://www.crewai.com)
- [Vertex AI Agent Builder](https://cloud.google.com/products/agent-builder)
