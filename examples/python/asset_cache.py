"""
Asset Cache Layer - SQLite-based caching for API responses

This module provides a local caching layer to minimize API calls and improve
performance. Assets are cached with configurable TTL and automatic expiration.

Usage:
    cache = AssetCache()

    # Check cache
    cached_assets = cache.get_cached_assets("mountain", "image")

    # Cache new assets
    cache.cache_assets("mountain", results, ttl_hours=24)

    # Clean expired
    cache.evict_expired()
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssetCache:
    """SQLite-based caching layer for asset search results"""

    def __init__(self, db_path: str = "assets_cache.db"):
        """
        Initialize asset cache

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.setup_database()
        logger.info(f"Asset cache initialized at {db_path}")

    def setup_database(self):
        """Initialize cache database with metadata support"""
        cursor = self.conn.cursor()

        # Main assets table
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
                expires_at TIMESTAMP,
                hit_count INTEGER DEFAULT 0
            )
        """)

        # Index for fast query lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_type
            ON assets(query, type, source)
        """)

        # Index for expiration cleanup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires
            ON assets(expires_at)
        """)

        # Index for cache statistics
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cached_at
            ON assets(cached_at)
        """)

        self.conn.commit()
        logger.info("Database schema initialized")

    def get_cached_assets(
        self,
        query: str,
        asset_type: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve cached assets if not expired

        Args:
            query: Search query string
            asset_type: Optional filter by type (image, video, audio, gif)
            source: Optional filter by source (pexels, pixabay, etc.)

        Returns:
            List of cached asset dictionaries
        """
        cursor = self.conn.cursor()

        sql = """
            SELECT id, source, type, url, thumbnail, metadata, hit_count
            FROM assets
            WHERE LOWER(query) = LOWER(?)
            AND (expires_at IS NULL OR expires_at > ?)
        """
        params = [query, datetime.now()]

        if asset_type:
            sql += " AND type = ?"
            params.append(asset_type)

        if source:
            sql += " AND source = ?"
            params.append(source)

        sql += " ORDER BY hit_count DESC, cached_at DESC"

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        # Increment hit count for all returned results
        if rows:
            ids = [row[0] for row in rows]
            placeholders = ','.join(['?' for _ in ids])
            cursor.execute(f"""
                UPDATE assets
                SET hit_count = hit_count + 1
                WHERE id IN ({placeholders})
            """, ids)
            self.conn.commit()

            logger.info(f"Cache hit: {len(rows)} assets for query '{query}'")

        results = [{
            "id": row[0].split('_', 1)[1] if '_' in row[0] else row[0],
            "source": row[1],
            "type": row[2],
            "url": row[3],
            "thumbnail": row[4],
            "metadata": json.loads(row[5]) if row[5] else {},
            "hit_count": row[6]
        } for row in rows]

        return results

    def cache_assets(
        self,
        query: str,
        assets: List[Dict],
        ttl_hours: int = 24
    ):
        """
        Cache assets with expiration

        Args:
            query: Original search query
            assets: List of asset dictionaries to cache
            ttl_hours: Time-to-live in hours (default 24)
        """
        cursor = self.conn.cursor()
        expires_at = datetime.now() + timedelta(hours=ttl_hours)

        cached_count = 0

        for asset in assets:
            try:
                # Create unique cache ID
                cache_id = f"{asset.get('source', 'unknown')}_{asset.get('id', hash(asset.get('url', '')))}"

                # Extract metadata (everything except standard fields)
                standard_fields = {'id', 'source', 'type', 'url', 'thumbnail'}
                metadata = {k: v for k, v in asset.items() if k not in standard_fields}

                cursor.execute("""
                    INSERT OR REPLACE INTO assets
                    (id, source, type, query, url, thumbnail, metadata, expires_at, hit_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                        COALESCE((SELECT hit_count FROM assets WHERE id = ?), 0))
                """, (
                    cache_id,
                    asset.get('source', 'unknown'),
                    asset.get('type', 'unknown'),
                    query.lower(),
                    asset.get('url', ''),
                    asset.get('thumbnail'),
                    json.dumps(metadata),
                    expires_at,
                    cache_id  # For preserving hit_count
                ))

                cached_count += 1

            except Exception as e:
                logger.error(f"Failed to cache asset: {e}")
                continue

        self.conn.commit()
        logger.info(f"Cached {cached_count} assets for query '{query}' (TTL: {ttl_hours}h)")

    def evict_expired(self) -> int:
        """
        Remove expired cache entries

        Returns:
            Number of entries removed
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM assets
            WHERE expires_at IS NOT NULL AND expires_at < ?
        """, (datetime.now(),))

        count = cursor.fetchone()[0]

        if count > 0:
            cursor.execute("""
                DELETE FROM assets
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (datetime.now(),))

            self.conn.commit()
            logger.info(f"Evicted {count} expired cache entries")

        return count

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics and health metrics

        Returns:
            Dictionary with cache statistics
        """
        cursor = self.conn.cursor()

        # Total assets
        cursor.execute("SELECT COUNT(*) FROM assets")
        total_assets = cursor.fetchone()[0]

        # Active (non-expired) assets
        cursor.execute("""
            SELECT COUNT(*) FROM assets
            WHERE expires_at IS NULL OR expires_at > ?
        """, (datetime.now(),))
        active_assets = cursor.fetchone()[0]

        # Assets by type
        cursor.execute("""
            SELECT type, COUNT(*) as count
            FROM assets
            WHERE expires_at IS NULL OR expires_at > ?
            GROUP BY type
        """, (datetime.now(),))
        by_type = {row[0]: row[1] for row in cursor.fetchall()}

        # Assets by source
        cursor.execute("""
            SELECT source, COUNT(*) as count
            FROM assets
            WHERE expires_at IS NULL OR expires_at > ?
            GROUP BY source
        """, (datetime.now(),))
        by_source = {row[0]: row[1] for row in cursor.fetchall()}

        # Most popular queries
        cursor.execute("""
            SELECT query, SUM(hit_count) as total_hits, COUNT(*) as asset_count
            FROM assets
            WHERE expires_at IS NULL OR expires_at > ?
            GROUP BY query
            ORDER BY total_hits DESC
            LIMIT 10
        """, (datetime.now(),))
        top_queries = [{
            "query": row[0],
            "total_hits": row[1],
            "asset_count": row[2]
        } for row in cursor.fetchall()]

        return {
            "total_assets": total_assets,
            "active_assets": active_assets,
            "expired_assets": total_assets - active_assets,
            "by_type": by_type,
            "by_source": by_source,
            "top_queries": top_queries,
            "database_path": self.db_path
        }

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cache entries

        Args:
            older_than_days: If specified, only clear entries older than this many days
        """
        cursor = self.conn.cursor()

        if older_than_days:
            cutoff = datetime.now() - timedelta(days=older_than_days)
            cursor.execute("DELETE FROM assets WHERE cached_at < ?", (cutoff,))
            logger.info(f"Cleared cache entries older than {older_than_days} days")
        else:
            cursor.execute("DELETE FROM assets")
            logger.info("Cleared entire cache")

        self.conn.commit()

    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Cache connection closed")


def main():
    """Example usage of AssetCache"""
    from asset_api import AssetAPI

    cache = AssetCache()
    api = AssetAPI()

    print("📊 Cache Statistics Before:")
    stats = cache.get_cache_stats()
    print(f"  Total assets: {stats['total_assets']}")
    print(f"  Active assets: {stats['active_assets']}")
    print()

    # Try to get cached results
    query = "mountain landscape"
    print(f"🔍 Looking for cached results for '{query}'...")
    cached = cache.get_cached_assets(query, "image")

    if cached:
        print(f"✅ Found {len(cached)} cached results!")
    else:
        print("❌ No cached results, fetching from API...")
        results = api.search_pexels_images(query, per_page=5)
        cache.cache_assets(query, results, ttl_hours=24)
        print(f"✅ Cached {len(results)} new results")

    print("\n📊 Cache Statistics After:")
    stats = cache.get_cache_stats()
    print(f"  Total assets: {stats['total_assets']}")
    print(f"  Active assets: {stats['active_assets']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  By source: {stats['by_source']}")

    if stats['top_queries']:
        print("\n🔥 Top Queries:")
        for i, q in enumerate(stats['top_queries'][:5], 1):
            print(f"  {i}. '{q['query']}' - {q['total_hits']} hits, {q['asset_count']} assets")

    # Clean up expired entries
    print("\n🧹 Cleaning expired entries...")
    evicted = cache.evict_expired()
    print(f"  Removed {evicted} expired entries")

    cache.close()


if __name__ == "__main__":
    main()
