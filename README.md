Now I'll compile this comprehensive research into a production-ready guide for you.

# Complete Guide: Free Asset Pipeline for AI Content Creation (2025 Edition)

## Overview

Building an automated pipeline for sourcing, indexing, and integrating free assets (images, videos, audio, GIFs, SVGs) into your AI agentic content creation workflow is essential for scaling your production without licensing overhead. This guide provides production-ready implementation patterns, API integrations, and orchestration strategies specifically tailored for your tech stack (Next.js, Supabase, Vercel, Google Cloud).

## 1. Comprehensive Asset Sources & API Access

### Images & SVG Vectors

**Pexels**[1][2][3]
- **License**: Free (attribution suggested, not required)
- **API Access**: ✅ Full REST API
- **Rate Limits**: 200 requests/hour, 20,000/month (default)
- **Unlimited Access**: Free for eligible apps - email api@pexels.com with platform details and attribution proof[1]
- **Best For**: High-quality photos + videos in one API

**Pixabay**[4][5][6]
- **License**: CC0-like (free for commercial use)
- **API Access**: ✅ Full REST API with comprehensive filtering
- **Rate Limits**: 5,000 requests/hour (production)[4]
- **Best For**: Large library with images, videos, vectors, and illustrations

**Unsplash**[7][8][9]
- **License**: Free with attribution required
- **API Access**: ✅ OAuth2 + API key authentication
- **Rate Limits**: 50 requests/hour (demo), upgrade available[8][7]
- **Best For**: High-resolution artistic photography

**Openverse**[10][11][12]
- **License**: CC0 / CC-BY aggregator
- **API Access**: ✅ Unified search across 800M+ assets[11]
- **Best For**: Cross-platform search (images + audio) with clear licensing metadata

### Audio & Sound Effects

**Freesound**[13][14][15]
- **License**: CC0 / CC-BY user-uploaded content
- **API Access**: ✅ RESTful API with OAuth2[16][13]
- **Features**: Content-based similarity search, advanced metadata filtering
- **Rate Limits**: API key based (generous for non-commercial)

**Free Music Archive**
- **License**: CC-BY / CC0 music tracks
- **API Access**: ✅ Search by genre, mood, instrument
- **Best For**: Background music for video content

**Openverse Audio**[10]
- **License**: CC0 / CC-BY
- **API Access**: ✅ Same unified API as images
- **Best For**: Quick audio searches with licensing clarity

### Video & GIFs

**Pexels Video**[17][1]
- **License**: Free (same as Pexels photos)
- **API Access**: ✅ Same API endpoint as images
- **Best For**: Professional stock video clips

**Pixabay Video**[4]
- **License**: CC0
- **API Access**: ✅ Integrated with main Pixabay API
- **Best For**: B-roll footage

**Giphy**[18][19][20][21]
- **License**: Free for non-commercial use
- **API Access**: ✅ Robust API with search, trending, stickers
- **Rate Limits**: 100 requests/hour (beta), 21,000/hour + 50,000/day (production)[21][18]
- **Best For**: GIF animations and stickers

## 2. API Integration Patterns

### Direct API Implementation (Python)

```python
import os
import requests
from typing import List, Dict
import sqlite3
from datetime import datetime

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
    
    def search_pixabay_videos(self, query: str, per_page: int = 10) -> List[Dict]:
        """Search Pixabay for video content"""
        url = "https://pixabay.com/api/videos/"
        params = {
            "key": self.pixabay_key,
            "q": query,
            "per_page": per_page
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return [{
                "id": video["id"],
                "url": video["videos"]["large"]["url"],
                "thumbnail": video["userImageURL"],
                "duration": video["duration"],
                "source": "pixabay",
                "type": "video"
            } for video in data.get("hits", [])]
        return []
    
    def search_freesound(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search Freesound for audio files"""
        url = "https://freesound.org/apiv2/search/text/"
        params = {
            "query": query,
            "token": self.freesound_key,
            "page_size": max_results,
            "fields": "id,name,tags,duration,license,previews"
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return [{
                "id": result["id"],
                "name": result["name"],
                "preview_url": result["previews"]["preview-hq-mp3"],
                "duration": result["duration"],
                "license": result["license"],
                "source": "freesound",
                "type": "audio"
            } for result in data.get("results", [])]
        return []
    
    def search_giphy_gifs(self, query: str, limit: int = 25) -> List[Dict]:
        """Search Giphy for GIFs"""
        url = "https://api.giphy.com/v1/gifs/search"
        params = {
            "api_key": self.giphy_key,
            "q": query,
            "limit": limit,
            "rating": "g"
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return [{
                "id": gif["id"],
                "url": gif["images"]["original"]["url"],
                "thumbnail": gif["images"]["preview_gif"]["url"],
                "title": gif["title"],
                "source": "giphy",
                "type": "gif"
            } for gif in data.get("data", [])]
        return []
```

### Local SQLite Caching Layer

```python
import sqlite3
import json
from datetime import datetime, timedelta

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
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires 
            ON assets(expires_at)
        """)
        
        self.conn.commit()
    
    def get_cached_assets(self, query: str, asset_type: str = None) -> List[Dict]:
        """Retrieve cached assets if not expired"""
        cursor = self.conn.cursor()
        
        sql = """
            SELECT id, source, type, url, thumbnail, metadata
            FROM assets
            WHERE query = ? AND (expires_at IS NULL OR expires_at > ?)
        """
        params = [query, datetime.now()]
        
        if asset_type:
            sql += " AND type = ?"
            params.append(asset_type)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        return [{
            "id": row[0],
            "source": row[1],
            "type": row[2],
            "url": row[3],
            "thumbnail": row[4],
            "metadata": json.loads(row[5]) if row[5] else {}
        } for row in rows]
    
    def cache_assets(self, query: str, assets: List[Dict], ttl_hours: int = 24):
        """Cache assets with expiration"""
        cursor = self.conn.cursor()
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        for asset in assets:
            cursor.execute("""
                INSERT OR REPLACE INTO assets 
                (id, source, type, query, url, thumbnail, metadata, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{asset['source']}_{asset['id']}",
                asset['source'],
                asset['type'],
                query,
                asset['url'],
                asset.get('thumbnail'),
                json.dumps(asset.get('metadata', {})),
                expires_at
            ))
        
        self.conn.commit()
    
    def evict_expired(self):
        """Remove expired cache entries"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM assets WHERE expires_at < ?", (datetime.now(),))
        self.conn.commit()
```

## 3. Model Context Protocol (MCP) Server Integration

### Python MCP Server for Asset Management

```python
from fastmcp import FastMCP
from typing import Dict, List
import asyncio

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
        # Pixabay images endpoint
        results.extend(api_client.search_pixabay_images(query))
    
    if "unsplash" in sources:
        results.extend(api_client.search_unsplash_images(query))
    
    # Cache results
    cache.cache_assets(query, results)
    
    return {"cached": False, "count": len(results), "results": results}

@mcp.tool()
def search_videos(query: str, max_duration: int = None) -> Dict:
    """
    Search for royalty-free video clips.
    
    Args:
        query: Search keywords
        max_duration: Maximum video duration in seconds
    
    Returns:
        List of video assets with metadata
    """
    cached = cache.get_cached_assets(query, "video")
    if cached:
        return {"cached": True, "results": cached}
    
    results = []
    results.extend(api_client.search_pexels_videos(query))
    results.extend(api_client.search_pixabay_videos(query))
    
    # Filter by duration if specified
    if max_duration:
        results = [r for r in results if r.get("duration", 0) <= max_duration]
    
    cache.cache_assets(query, results)
    
    return {"cached": False, "count": len(results), "results": results}

@mcp.tool()
def search_audio(query: str, min_duration: int = 0, max_duration: int = 300) -> Dict:
    """
    Search for royalty-free audio files and sound effects.
    
    Args:
        query: Search keywords
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds
    
    Returns:
        List of audio assets with licensing information
    """
    cached = cache.get_cached_assets(query, "audio")
    if cached:
        return {"cached": True, "results": cached}
    
    results = api_client.search_freesound(query)
    
    # Filter by duration
    results = [
        r for r in results 
        if min_duration <= r.get("duration", 0) <= max_duration
    ]
    
    cache.cache_assets(query, results)
    
    return {"cached": False, "count": len(results), "results": results}

@mcp.tool()
def search_gifs(query: str, limit: int = 20) -> Dict:
    """
    Search for GIF animations.
    
    Args:
        query: Search keywords
        limit: Maximum number of results
    
    Returns:
        List of GIF assets
    """
    cached = cache.get_cached_assets(query, "gif")
    if cached:
        return {"cached": True, "results": cached[:limit]}
    
    results = api_client.search_giphy_gifs(query, limit)
    cache.cache_assets(query, results)
    
    return {"cached": False, "count": len(results), "results": results}

@mcp.resource("asset://cache/stats")
def get_cache_stats() -> str:
    """Get cache statistics and health metrics"""
    cursor = cache.conn.cursor()
    
    cursor.execute("""
        SELECT 
            type,
            source,
            COUNT(*) as count,
            MAX(cached_at) as latest
        FROM assets
        WHERE expires_at > ?
        GROUP BY type, source
    """, (datetime.now(),))
    
    stats = cursor.fetchall()
    
    return json.dumps({
        "statistics": [
            {"type": s[0], "source": s[1], "count": s[2], "latest": s[3]}
            for s in stats
        ]
    }, indent=2)

if __name__ == "__main__":
    mcp.run()
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

-- User collections for organizing assets
CREATE TABLE asset_collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id),
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Many-to-many relationship
CREATE TABLE collection_assets (
    collection_id UUID REFERENCES asset_collections(id) ON DELETE CASCADE,
    asset_id UUID REFERENCES asset_metadata(id) ON DELETE CASCADE,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (collection_id, asset_id)
);
```

### Next.js API Routes

```typescript
// app/api/assets/search/route.ts
import { createClient } from '@supabase/supabase-js';
import { NextRequest, NextResponse } from 'next/server';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

export async function POST(request: NextRequest) {
  try {
    const { query, type, sources, limit = 20 } = await request.json();
    
    // First, check Supabase cache
    let { data: cachedAssets, error: cacheError } = await supabase
      .from('asset_metadata')
      .select('*')
      .contains('search_keywords', [query])
      .eq('type', type)
      .limit(limit);
    
    if (cachedAssets && cachedAssets.length > 0) {
      return NextResponse.json({ 
        cached: true, 
        results: cachedAssets 
      });
    }
    
    // If not cached, fetch from external APIs
    const results = await fetchFromAPIs(query, type, sources);
    
    // Store in Supabase for future queries
    await storeAssets(results);
    
    return NextResponse.json({ 
      cached: false, 
      results 
    });
    
  } catch (error) {
    console.error('Asset search error:', error);
    return NextResponse.json(
      { error: 'Failed to search assets' },
      { status: 500 }
    );
  }
}

async function fetchFromAPIs(
  query: string, 
  type: string, 
  sources: string[]
): Promise<any[]> {
  const results: any[] = [];
  
  // Parallel API calls for better performance
  const promises = sources.map(async (source) => {
    switch (source) {
      case 'pexels':
        return fetchPexels(query, type);
      case 'pixabay':
        return fetchPixabay(query, type);
      case 'unsplash':
        return fetchUnsplash(query, type);
      default:
        return [];
    }
  });
  
  const apiResults = await Promise.all(promises);
  return apiResults.flat();
}

async function storeAssets(assets: any[]): Promise<void> {
  const { error } = await supabase
    .from('asset_metadata')
    .upsert(
      assets.map(asset => ({
        asset_id: `${asset.source}_${asset.id}`,
        source: asset.source,
        type: asset.type,
        url: asset.url,
        thumbnail_url: asset.thumbnail,
        title: asset.title || asset.query,
        tags: asset.tags || [],
        search_keywords: [asset.query],
        metadata: asset
      })),
      { onConflict: 'asset_id' }
    );
  
  if (error) console.error('Failed to cache assets:', error);
}
```

### React Component with Optimized Image Loading

```tsx
// components/AssetGallery.tsx
'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';

interface Asset {
  id: string;
  url: string;
  thumbnail_url: string;
  title: string;
  source: string;
  type: string;
}

export default function AssetGallery({ 
  query, 
  type = 'image' 
}: { 
  query: string; 
  type: string;
}) {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function searchAssets() {
      setLoading(true);
      try {
        const response = await fetch('/api/assets/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query,
            type,
            sources: ['pexels', 'pixabay', 'unsplash'],
            limit: 30
          })
        });
        
        const data = await response.json();
        setAssets(data.results);
      } catch (error) {
        console.error('Search failed:', error);
      } finally {
        setLoading(false);
      }
    }

    if (query) {
      searchAssets();
    }
  }, [query, type]);

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 p-4">
      {loading ? (
        <div>Loading assets...</div>
      ) : (
        assets.map((asset) => (
          <div 
            key={asset.id} 
            className="relative aspect-square overflow-hidden rounded-lg hover:scale-105 transition-transform"
          >
            <Image
              src={asset.thumbnail_url || asset.url}
              alt={asset.title}
              fill
              className="object-cover"
              sizes="(max-width: 768px) 50vw, (max-width: 1200px) 33vw, 25vw"
              loading="lazy"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 p-2">
              <p className="text-white text-xs truncate">{asset.title}</p>
              <span className="text-white/60 text-xs">{asset.source}</span>
            </div>
          </div>
        ))
      )}
    </div>
  );
}
```

## 5. AI Agent Orchestration Workflows

### LangChain Agent with Asset Tools

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define asset search tools
def search_images_tool(query: str) -> str:
    """Search for royalty-free images"""
    results = api_client.search_pexels_images(query, per_page=5)
    return json.dumps(results, indent=2)

def search_videos_tool(query: str) -> str:
    """Search for royalty-free videos"""
    results = api_client.search_pixabay_videos(query, per_page=3)
    return json.dumps(results, indent=2)

def search_audio_tool(query: str) -> str:
    """Search for royalty-free audio"""
    results = api_client.search_freesound(query, max_results=5)
    return json.dumps(results, indent=2)

# Create tools list
tools = [
    Tool(
        name="SearchImages",
        func=search_images_tool,
        description="Search for royalty-free images. Input should be descriptive keywords."
    ),
    Tool(
        name="SearchVideos",
        func=search_videos_tool,
        description="Search for royalty-free video clips. Input should be descriptive keywords."
    ),
    Tool(
        name="SearchAudio",
        func=search_audio_tool,
        description="Search for royalty-free audio and sound effects. Input should be descriptive keywords."
    )
]

# Create agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a content creation assistant that helps find royalty-free assets.
    When a user requests media assets, use the available tools to search across multiple sources.
    Always provide direct URLs and attribute the source properly."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create and run agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Example usage
result = agent_executor.invoke({
    "input": "Find me 3 high-quality images of mountain landscapes and 2 royalty-free ambient music tracks"
})
```

### AutoGen Multi-Agent Content Pipeline

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Configuration
config_list = [{
    "model": "gpt-4",
    "api_key": os.getenv("OPENAI_API_KEY")
}]

# Asset Research Agent
asset_researcher = AssistantAgent(
    name="AssetResearcher",
    system_message="""You are an asset research specialist. Your job is to:
    1. Understand content requirements
    2. Identify needed asset types (images, videos, audio)
    3. Generate specific search queries
    4. Return structured asset requests""",
    llm_config={"config_list": config_list}
)

# Asset Fetcher Agent  
asset_fetcher = AssistantAgent(
    name="AssetFetcher",
    system_message="""You fetch assets from APIs. When given search queries:
    1. Call appropriate API endpoints
    2. Filter and rank results
    3. Return downloadable URLs with metadata
    Use functions: search_images, search_videos, search_audio""",
    llm_config={
        "config_list": config_list,
        "functions": [
            {
                "name": "search_images",
                "description": "Search royalty-free images",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "sources": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        ]
    }
)

# Content Assembler Agent
content_assembler = AssistantAgent(
    name="ContentAssembler",
    system_message="""You assemble final content packages:
    1. Review fetched assets
    2. Select best matches for the content brief
    3. Generate attribution and licensing notes
    4. Create final asset manifest""",
    llm_config={"config_list": config_list}
)

# User proxy for execution
user_proxy = UserProxyAgent(
    name="ContentCreator",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "assets_output"}
)

# Group chat for coordination
groupchat = GroupChat(
    agents=[asset_researcher, asset_fetcher, content_assembler, user_proxy],
    messages=[],
    max_round=10
)

manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

# Execute workflow
user_proxy.initiate_chat(
    manager,
    message="""Create a social media video about sustainable living. 
    I need: 3 nature images, 2 short video clips of renewable energy, 
    and 1 uplifting background music track."""
)
```

### CrewAI Production Workflow

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool

# Define custom tools
@tool("Search Images")
def search_images(query: str) -> str:
    """Search for royalty-free images"""
    results = api_client.search_pexels_images(query)
    return json.dumps(results[:5])

@tool("Search Videos")
def search_videos(query: str) -> str:
    """Search for royalty-free videos"""
    results = api_client.search_pixabay_videos(query)
    return json.dumps(results[:3])

@tool("Download Asset")
def download_asset(url: str, filename: str) -> str:
    """Download asset to local storage"""
    response = requests.get(url)
    with open(f"assets/{filename}", "wb") as f:
        f.write(response.content)
    return f"Downloaded: {filename}"

# Define agents
content_strategist = Agent(
    role="Content Strategist",
    goal="Understand content requirements and plan asset needs",
    backstory="Expert in content strategy and visual storytelling",
    tools=[],
    verbose=True
)

media_curator = Agent(
    role="Media Curator",
    goal="Find and curate high-quality royalty-free assets",
    backstory="Specialist in sourcing and evaluating visual media",
    tools=[search_images, search_videos],
    verbose=True
)

asset_manager = Agent(
    role="Asset Manager",
    goal="Download, organize, and prepare assets for production",
    backstory="Expert in digital asset management and file organization",
    tools=[download_asset],
    verbose=True
)

# Define tasks
task1 = Task(
    description="""Analyze this content brief: {content_brief}
    Identify all required assets (images, videos, audio) and create detailed search queries.""",
    expected_output="List of asset requirements with specific search queries",
    agent=content_strategist
)

task2 = Task(
    description="""Using the asset requirements, search for:
    - High-quality images matching the queries
    - Relevant video clips
    - Appropriate audio tracks
    Return the top 3 options for each category with URLs.""",
    expected_output="Curated list of assets with URLs and metadata",
    agent=media_curator
)

task3 = Task(
    description="""Download selected assets and organize them:
    - Create folder structure by asset type
    - Rename files with descriptive names
    - Generate manifest.json with all asset metadata and attributions""",
    expected_output="Downloaded assets with organized structure and manifest",
    agent=asset_manager
)

# Create and run crew
crew = Crew(
    agents=[content_strategist, media_curator, asset_manager],
    tasks=[task1, task2, task3],
    process=Process.sequential,
    verbose=2
)

result = crew.kickoff(inputs={
    "content_brief": "Create a 60-second explainer video about AI in healthcare"
})
```

## 6. Deployment & Production Optimization

### Docker Compose for Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    build: ./mcp-server
    ports:
      - "8080:8080"
    environment:
      - PEXELS_API_KEY=${PEXELS_API_KEY}
      - PIXABAY_API_KEY=${PIXABAY_API_KEY}
      - UNSPLASH_ACCESS_KEY=${UNSPLASH_ACCESS_KEY}
      - FREESOUND_API_KEY=${FREESOUND_API_KEY}
    volumes:
      - ./cache:/app/cache
      - ./assets:/app/assets
    restart: unless-stopped

  nextjs-app:
    build: ./nextjs-app
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_SERVICE_ROLE_KEY=${SUPABASE_SERVICE_KEY}
      - MCP_SERVER_URL=http://mcp-server:8080
    depends_on:
      - mcp-server
    restart: unless-stopped

  asset-processor:
    build: ./asset-processor
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_SERVICE_KEY}
    volumes:
      - ./assets:/app/assets
    depends_on:
      - mcp-server
```

### Vercel Deployment Configuration

```json
// vercel.json
{
  "buildCommand": "npm run build",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "nextjs",
  "outputDirectory": ".next",
  "env": {
    "NEXT_PUBLIC_SUPABASE_URL": "@supabase-url",
    "SUPABASE_SERVICE_ROLE_KEY": "@supabase-service-key",
    "PEXELS_API_KEY": "@pexels-api-key",
    "PIXABAY_API_KEY": "@pixabay-api-key"
  },
  "functions": {
    "app/api/**/*.ts": {
      "maxDuration": 30,
      "memory": 1024
    }
  },
  "images": {
    "domains": [
      "images.pexels.com",
      "pixabay.com",
      "images.unsplash.com",
      "media.giphy.com"
    ],
    "remotePatterns": [
      {
        "protocol": "https",
        "hostname": "**.pexels.com"
      },
      {
        "protocol": "https",
        "hostname": "**.pixabay.com"
      }
    ]
  }
}
```

### Performance Optimization Tips

**Image Optimization with Vercel**[22][23][24]
- Use `next/image` component for automatic optimization
- Vercel optimizes images on-demand without slowing builds
- Supports automatic format conversion (WebP/AVIF)
- Implements lazy loading by default

**Caching Strategy**[2][25][26]
- Cache API responses for 24 hours in SQLite[25][26]
- Use Supabase for long-term asset metadata storage[27][28]
- Implement Redis/Upstash for distributed caching in production
- Normalize search queries to maximize cache hits[2]

**Rate Limit Management**
- Implement request queuing for API calls
- Use exponential backoff for 429 responses
- Batch requests where APIs support it
- Monitor usage with custom logging[29][1]

## 7. Google Vertex AI Agent Integration

### Vertex AI Agent with Asset Tools

```python
from google.cloud import aiplatform
from vertexai.preview import reasoning_engines

# Initialize Vertex AI
aiplatform.init(project="your-project-id", location="us-central1")

# Define asset search function for Vertex AI
def vertex_search_assets(query: str, asset_type: str = "image") -> dict:
    """
    Search for royalty-free assets - callable by Vertex AI agents
    """
    api = AssetAPI()
    cache = AssetCache()
    
    # Try cache first
    cached = cache.get_cached_assets(query, asset_type)
    if cached:
        return {"status": "cached", "results": cached}
    
    # Fetch from APIs
    results = []
    if asset_type == "image":
        results = api.search_pexels_images(query)
    elif asset_type == "video":
        results = api.search_pixabay_videos(query)
    elif asset_type == "audio":
        results = api.search_freesound(query)
    
    cache.cache_assets(query, results)
    return {"status": "fresh", "count": len(results), "results": results}

# Deploy as Vertex AI Agent
agent = reasoning_engines.LangchainAgent(
    model="gemini-1.5-pro",
    tools=[vertex_search_assets],
    agent_executor_kwargs={
        "return_intermediate_steps": True,
        "verbose": True
    }
)

# Deploy to Agent Engine
remote_agent = reasoning_engines.ReasoningEngine.create(
    agent,
    requirements=["langchain", "google-cloud-aiplatform"],
    display_name="asset-search-agent"
)
```

## 8. Complete End-to-End Example

### Production-Ready Content Generation Pipeline

```python
import asyncio
from typing import List, Dict
import json

class ContentProductionPipeline:
    def __init__(self):
        self.api = AssetAPI()
        self.cache = AssetCache()
        
    async def generate_social_video_assets(
        self, 
        brief: str,
        style: str = "modern",
        duration: int = 60
    ) -> Dict:
        """
        Complete pipeline: Brief → Asset Discovery → Download → Manifest
        """
        
        # Step 1: Analyze brief with LLM
        requirements = await self.analyze_brief(brief, duration)
        
        # Step 2: Search for assets in parallel
        tasks = [
            self.fetch_images(requirements["image_queries"]),
            self.fetch_videos(requirements["video_queries"]),
            self.fetch_audio(requirements["audio_queries"])
        ]
        images, videos, audio = await asyncio.gather(*tasks)
        
        # Step 3: Download and organize
        manifest = await self.download_and_organize({
            "images": images,
            "videos": videos,
            "audio": audio
        })
        
        # Step 4: Store in Supabase
        await self.store_in_supabase(manifest)
        
        return manifest
    
    async def analyze_brief(self, brief: str, duration: int) -> Dict:
        """Use LLM to extract asset requirements"""
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Extract media asset requirements from content briefs"
            }, {
                "role": "user",
                "content": f"""Analyze this content brief and generate search queries:
                
                Brief: {brief}
                Duration: {duration} seconds
                
                Return JSON with:
                - image_queries: [list of 3-5 specific image search terms]
                - video_queries: [list of 2-3 video clip search terms]
                - audio_queries: [1 background music search term]
                """
            }],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    async def fetch_images(self, queries: List[str]) -> List[Dict]:
        """Fetch images for all queries"""
        results = []
        for query in queries:
            # Check cache
            cached = self.cache.get_cached_assets(query, "image")
            if cached:
                results.extend(cached[:2])  # Top 2 per query
            else:
                fresh = self.api.search_pexels_images(query, per_page=2)
                self.cache.cache_assets(query, fresh)
                results.extend(fresh)
        return results
    
    async def fetch_videos(self, queries: List[str]) -> List[Dict]:
        """Fetch video clips"""
        results = []
        for query in queries:
            cached = self.cache.get_cached_assets(query, "video")
            if cached:
                results.extend(cached[:1])
            else:
                fresh = self.api.search_pixabay_videos(query, per_page=1)
                self.cache.cache_assets(query, fresh)
                results.extend(fresh)
        return results
    
    async def fetch_audio(self, queries: List[str]) -> List[Dict]:
        """Fetch audio tracks"""
        results = []
        for query in queries:
            cached = self.cache.get_cached_assets(query, "audio")
            if cached:
                results.append(cached[0])
            else:
                fresh = self.api.search_freesound(query, max_results=1)
                self.cache.cache_assets(query, fresh)
                if fresh:
                    results.append(fresh[0])
        return results
    
    async def download_and_organize(self, assets: Dict) -> Dict:
        """Download assets and create manifest"""
        import aiohttp
        import os
        
        manifest = {
            "project_id": f"project_{int(time.time())}",
            "created_at": datetime.now().isoformat(),
            "assets": []
        }
        
        os.makedirs("assets/images", exist_ok=True)
        os.makedirs("assets/videos", exist_ok=True)
        os.makedirs("assets/audio", exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            for asset_type, items in assets.items():
                for idx, item in enumerate(items):
                    url = item["url"]
                    ext = url.split(".")[-1].split("?")[0]
                    filename = f"{asset_type}/{asset_type}_{idx}.{ext}"
                    
                    async with session.get(url) as resp:
                        content = await resp.read()
                        with open(f"assets/{filename}", "wb") as f:
                            f.write(content)
                    
                    manifest["assets"].append({
                        "type": asset_type,
                        "filename": filename,
                        "source": item["source"],
                        "original_url": url,
                        "attribution": item.get("photographer") or item.get("name")
                    })
        
        # Save manifest
        with open("assets/manifest.json", "w") as f:
            json.dumps(manifest, f, indent=2)
        
        return manifest
    
    async def store_in_supabase(self, manifest: Dict):
        """Store asset metadata in Supabase"""
        from supabase import create_client
        
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )
        
        for asset in manifest["assets"]:
            supabase.table("asset_metadata").insert({
                "project_id": manifest["project_id"],
                "type": asset["type"],
                "filename": asset["filename"],
                "source": asset["source"],
                "original_url": asset["original_url"],
                "attribution": asset["attribution"],
                "metadata": asset
            }).execute()

# Run the pipeline
async def main():
    pipeline = ContentProductionPipeline()
    
    result = await pipeline.generate_social_video_assets(
        brief="""Create an inspiring 60-second video about remote work. 
        Show diverse people working from different locations - home offices, 
        cafes, outdoor spaces. Include shots of collaboration tools and happy 
        productive moments. Uplifting background music.""",
        style="modern",
        duration=60
    )
    
    print(f"✅ Generated assets for project: {result['project_id']}")
    print(f"📦 Total assets: {len(result['assets'])}")
    print(f"📁 Manifest saved to: assets/manifest.json")

if __name__ == "__main__":
    asyncio.run(main())
```

## Conclusion

This complete pipeline enables you to:

✅ **Automatically source** free assets from 10+ APIs  
✅ **Cache efficiently** with SQLite + Supabase for fast retrieval  
✅ **Integrate seamlessly** with Next.js, Vercel, and Google Cloud  
✅ **Orchestrate intelligently** using LangChain, AutoGen, or CrewAI agents  
✅ **Scale to production** with proper rate limiting and error handling  
✅ **Support MCP** for Cursor AI and Claude Desktop integration  

The architecture is designed for your CineFilm.tech platform and scales from local development to enterprise production deployment.[30][31][32][33][34][35][36][37][38][17]

[1](https://help.pexels.com/hc/en-us/articles/900005852323-How-do-I-get-unlimited-requests)
[2](https://help.pexels.com/hc/en-us/articles/900006470063-What-steps-can-I-take-to-avoid-hitting-the-rate-limit)
[3](https://codewithbish.com/build-free-stock-photos-website-with-pexels-api-beginners-guide/)
[4](https://webflow.com/integrations/pixabay)
[5](https://publicapis.io/pixabay-api)
[6](https://github.com/dderevjanik/pixabay-api)
[7](https://featuredimageplugin.com/help/how-to-register-and-setup-an-api-access-key-for-unsplash/)
[8](https://javascript.plainenglish.io/a-beginners-guide-to-unsplash-api-in-javascript-2524c51ae1f3)
[9](https://www.pluralsight.com/resources/blog/guides/using-the-unsplash-api)
[10](https://api.openverse.org)
[11](https://openverse.org/about)
[12](http://pressable.com/blog/benefits-of-openverse-for-wordpress-sites/)
[13](https://freesound.org/docs/api/overview.html)
[14](https://publicapi.dev/freesound-api)
[15](https://freesound.org/docs/api/)
[16](https://pypi.org/project/freesound-api/)
[17](https://www.youtube.com/watch?v=AYsg5gAMWyo)
[18](https://stackoverflow.com/questions/54359770/how-api-rate-limits-works-in-giphy)
[19](https://xenforo.com/docs/xf2/giphy/)
[20](https://community.latenode.com/t/understanding-request-limits-for-giphy-api-in-different-modes/38044)
[21](https://developers.giphy.com/docs/)
[22](https://vercel.com/docs/frameworks/full-stack/nextjs)
[23](https://vercel.com/guides/how-to-optimize-next.js-sitecore-jss)
[24](https://vercel.com/docs/image-optimization)
[25](https://www.sqliteforum.com/p/implementing-cache-strategies-for)
[26](https://dev.to/waylonwalker/how-i-setup-a-sqlite-cache-in-python-17ni)
[27](https://supabase.com/docs/guides/ai/python/metadata)
[28](https://supabase.com/docs/guides/ai/structured-unstructured)
[29](https://help.pexels.com/hc/en-us/articles/900005368726-How-do-I-see-how-many-requests-I-have-remaining)
[30](https://wpaiworkflowautomation.com/workflows/ai-powered-content-generation/)
[31](https://www.oneclickitsolution.com/centerofexcellence/aiml/automate-digital-asset-creation-with-ai-agent)
[32](https://sparkco.ai/blog/deep-dive-into-api-integration-for-ai-agents)
[33](https://sparkco.ai/blog/deep-dive-into-autogen-multi-agent-patterns-2025)
[34](https://cloud.google.com/blog/products/ai-machine-learning/build-and-manage-multi-system-agents-with-vertex-ai)
[35](https://cloud.google.com/products/agent-builder)
[36](https://www.getmaxim.ai/articles/top-5-ai-agent-frameworks-in-2025-a-practical-guide-for-ai-builders/)
[37](https://mem0.ai/blog/crewai-guide-multi-agent-ai-teams)
[38](https://www.crewai.com)
[39](https://www.youtube.com/watch?v=CycDep25NEc)
[40](https://encord.com/blog/ai-agents-guide-to-agentic-ai/)
[41](https://www.linkedin.com/posts/artyushenkokatia_i-built-an-ai-workflow-that-creates-and-uploads-activity-7389292525355765760-kNxO)
[42](https://github.com/geeknik/my-awesome-stars)
[43](https://developer.nvidia.com/blog/build-an-agentic-video-workflow-with-video-search-and-summarization/)
[44](https://www.youtube.com/watch?v=_AprVrgnq4w)
[45](https://startintegrate.com/apps/productivity/pixabay-yo27d6)
[46](https://www.instagram.com/reel/C6wTI5dp-eV/)
[47](https://auth0.com/blog/mcp-specs-update-all-about-auth/)
[48](https://dev.to/creativetim_official/10-free-and-open-source-n8n-workflow-automation-templates-to-boost-your-productivity-18j9)
[49](https://northflank.com/blog/how-to-build-and-deploy-a-model-context-protocol-mcp-server)
[50](https://datasciencedojo.com/blog/react-agent-with-langchain-toolkit/)
[51](https://n8n.io)
[52](https://www.reddit.com/r/mcp/comments/1l8d69i/best_model_context_protocol_mcp_servers_in_2025/)
[53](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
[54](https://www.youtube.com/watch?v=FsG2_ScIYBA)
[55](https://modelcontextprotocol.info/blog/mcp-next-version-update/)
[56](https://docs.langchain.com/oss/python/langgraph/functional-api)
[57](https://n8n.io/workflows/)
[58](https://digitalproduction.com/2025/11/05/das-element-2-2-major-step-forward-for-asset-management/)
[59](https://www.reddit.com/r/VIDEOENGINEERING/comments/1kuyleh/54_ffmpeg_commands_for_video_automation_based_on/)
[60](https://aiagentinsider.ai/autogen-review/)
[61](https://netflixtechblog.com/data-pipeline-asset-management-with-dataflow-86525b3e21ca)
[62](https://github.com/rendi-api/ffmpeg-cheatsheet)
[63](https://www.sevensquaretech.com/autogen-vs-langgraph-ai-workflow/)
[64](https://www.fabi.ai/blog/data-pipelines-in-python-a-quickstart-guide-with-practical-tips)
[65](https://ffmpeg-api.com/learn/integrations/n8n)
[66](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/01/WEF-2025_Leave-Behind_AutoGen.pdf)
[67](https://community.amazonquicksight.com/t/best-practices-for-quick-sight-ci-cd-pipeline-implementation-with-python/46247)
[68](https://wideo.co/blog/step-by-step-guide-to-automating-video-editing/)
[69](https://cloudinary.com/blog/guest_post/create-an-optimized-image-gallery-in-next-js/)
[70](https://github.com/ruslanmv/Simple-MCP-Server-with-Python)
[71](https://www.freecodecamp.org/news/build-an-image-gallery-with-nextjs/)
[72](https://gofastmcp.com/tutorials/create-mcp-server)
[73](https://community.n8n.io/t/how-to-include-metadata-when-inserting-pdf-to-supabase-vector-store/90857)
[74](https://dev.to/theedgebreaker/building-a-responsive-image-gallery-with-nextjs-typescript-and-tailwind-css-46ee)
[75](https://www.freecodecamp.org/news/how-to-build-your-own-mcp-server-with-python/)
[76](https://supabase.com/features/vector-database)
[77](https://www.buildwithmatija.com/blog/handling-500-images-in-a-gallery-with-lazy-loading-in-next-js-15)
[78](https://modelcontextprotocol.io/docs/develop/build-server)
[79](https://www.linkedin.com/pulse/revolutionizing-qa-automation-cursor-ai-selenium-mcp-lahar-modhiya-toqff)
[80](https://www.youtube.com/watch?v=HN47tveqfQU)
[81](https://google.github.io/adk-docs/deploy/agent-engine/)
[82](https://forum.cursor.com/t/guide-a-simpler-more-autonomous-ai-workflow-for-cursor-new-update/70688)
[83](https://docs.github.com/en/copilot/how-tos/provide-context/use-mcp/use-the-github-mcp-server)
[84](https://dev.to/kevinmeyvaert/automating-my-workflow-with-ai-a-web-engineers-journey-4apb)
[85](https://docs.github.com/en/copilot/tutorials/enhance-agent-mode-with-mcp)
[86](https://www.tietoevry.com/en/blog/2025/07/building-multi-agents-google-ai-services/)
[87](https://appwrite.io/blog/post/the-future-of-coding-cursor-ai-and-the-rise-of-backend-automation-with-appwrite)
[88](https://docs.github.com/copilot/customizing-copilot/using-model-context-protocol/extending-copilot-chat-with-mcp)
[89](https://octopus.com/devops/ci-cd/ci-cd-with-docker/)
[90](https://stackoverflow.com/questions/28951133/sharing-precompiled-assets-across-docker-containers)
[91](https://dev.to/docker/unlocking-seamless-machine-learning-deployment-with-docker-a-guide-to-essential-cicd-tools-4af2)
[92](https://stackoverflow.com/questions/2714402/updating-a-local-sqlite-db-that-is-used-for-local-metadata-caching-from-a-serv)
[93](https://buddy.works/guides/docker-introduction)
[94](https://github.com/vercel/next.js/discussions/55747)
[95](https://pypi.org/project/sqlite3-cache/)
[96](https://www.youtube.com/watch?v=R2sy6kI-uBk)
[97](https://fastapi.tiangolo.com/advanced/custom-response/)
[98](https://developers.llamaindex.ai/python/llamaagents/workflows/)
[99](https://stackoverflow.com/questions/55873174/how-do-i-return-an-image-in-fastapi)
[100](https://www.crewai.com/signal-2025)
[101](https://sparkco.ai/blog/deep-dive-into-llamaindex-agent-framework)
[102](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
[103](https://www.llamaindex.ai/workflows)
[104](https://fastapi.tiangolo.com)
