# Free Asset Pipeline for AI Content Creation

> Production-ready pipeline for sourcing, indexing, and integrating free assets (images, videos, audio, GIFs, SVGs) into AI agentic content creation workflows.

## Overview

This repository provides a comprehensive, production-ready system for automating the discovery, caching, and integration of royalty-free digital assets into AI-powered content creation pipelines. Built specifically for modern tech stacks including Next.js, Supabase, Vercel, and Google Cloud.

## Features

- **Multi-Source Asset Discovery**: Integrate with 10+ free asset APIs (Pexels, Pixabay, Unsplash, Freesound, Giphy, and more)
- **Intelligent Caching**: SQLite and Supabase-based caching to minimize API calls and maximize performance
- **MCP Server Integration**: Model Context Protocol server for seamless integration with Cursor AI and Claude Desktop
- **AI Agent Orchestration**: Ready-to-use workflows for LangChain, AutoGen, CrewAI, and Vertex AI
- **Next.js Integration**: Complete API routes and React components for web applications
- **Production Deployment**: Docker Compose and Vercel configurations included

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/free-asset-pipeline.git
cd free-asset-pipeline
```

### 2. Install Dependencies

**Python (for MCP server and AI agents):**
```bash
pip install -r requirements.txt
```

**Node.js (for Next.js integration):**
```bash
npm install
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# Asset API Keys
PEXELS_API_KEY=your_pexels_key
PIXABAY_API_KEY=your_pixabay_key
UNSPLASH_ACCESS_KEY=your_unsplash_key
FREESOUND_API_KEY=your_freesound_key
GIPHY_API_KEY=your_giphy_key

# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_key

# OpenAI (for AI agents)
OPENAI_API_KEY=your_openai_key
```

### 4. Run the MCP Server

```bash
python examples/mcp-server/asset_mcp_server.py
```

### 5. Try the Examples

```bash
# Run the Python API example
python examples/python/asset_search.py

# Start the Next.js development server
cd examples/nextjs && npm run dev
```

## Documentation

📖 **[Complete Implementation Guide](./GUIDE.md)** - Comprehensive documentation covering:
- Asset sources and API access details
- Implementation patterns and code examples
- MCP server integration
- Next.js and Supabase architecture
- AI agent orchestration workflows
- Deployment and optimization strategies

## Repository Structure

```
free-asset-pipeline/
├── README.md                          # This file
├── GUIDE.md                           # Comprehensive implementation guide
├── examples/                          # Code examples
│   ├── python/                       # Python implementation examples
│   │   ├── asset_api.py             # Asset API client
│   │   ├── asset_cache.py           # SQLite caching layer
│   │   └── content_pipeline.py      # Complete pipeline example
│   ├── mcp-server/                   # MCP server implementation
│   │   ├── asset_mcp_server.py      # FastMCP server
│   │   └── mcp_config.json          # Configuration example
│   ├── nextjs/                       # Next.js integration
│   │   ├── app/api/assets/          # API routes
│   │   ├── components/              # React components
│   │   └── lib/                     # Utility functions
│   ├── ai-agents/                    # AI agent workflows
│   │   ├── langchain_agent.py       # LangChain implementation
│   │   ├── autogen_workflow.py      # AutoGen multi-agent system
│   │   ├── crewai_workflow.py       # CrewAI implementation
│   │   └── vertex_ai_agent.py       # Google Vertex AI agent
│   └── deployment/                   # Deployment configurations
│       ├── docker-compose.yml       # Docker setup
│       └── vercel.json              # Vercel configuration
├── schemas/                          # Database schemas
│   └── supabase_schema.sql          # Supabase table definitions
└── requirements.txt                  # Python dependencies
```

## Supported Asset Sources

| Source | Type | License | API Access | Rate Limits |
|--------|------|---------|------------|-------------|
| **Pexels** | Images, Videos | Free | ✅ REST API | 200/hour (upgradable) |
| **Pixabay** | Images, Videos, Vectors | CC0 | ✅ REST API | 5,000/hour |
| **Unsplash** | Images | Free (attribution) | ✅ REST API | 50/hour (upgradable) |
| **Freesound** | Audio, SFX | CC0/CC-BY | ✅ REST API | Generous |
| **Giphy** | GIFs | Free | ✅ REST API | 21,000/hour |
| **Openverse** | Images, Audio | CC0/CC-BY | ✅ Unified API | Generous |

## Use Cases

- **AI Content Generation**: Automatically source assets for AI-generated blog posts, social media, and marketing content
- **Video Production**: Build automated video editing pipelines with royalty-free B-roll and music
- **Creative Tools**: Power asset search features in design and content creation applications
- **Marketing Automation**: Generate on-brand visual content at scale
- **Educational Platforms**: Provide students with access to free, legal media assets

## Technology Stack

- **Backend**: Python 3.9+, FastAPI, FastMCP
- **Frontend**: Next.js 14+, React, TypeScript
- **Database**: Supabase (PostgreSQL + Vector), SQLite (local caching)
- **AI Frameworks**: LangChain, AutoGen, CrewAI, Vertex AI
- **Deployment**: Docker, Vercel, Google Cloud Platform
- **APIs**: Pexels, Pixabay, Unsplash, Freesound, Giphy, Openverse

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built for integration with [CineFilm.tech](https://cinefilm.tech)
- Inspired by the growing need for accessible, legal media assets in AI workflows
- Special thanks to Pexels, Pixabay, Unsplash, Freesound, and Giphy for providing free APIs

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the [comprehensive guide](./GUIDE.md) for detailed documentation
- Review the [examples](./examples) directory for working code

---

**Note**: Make sure to comply with each API provider's terms of service and attribution requirements when using their assets in production applications.
