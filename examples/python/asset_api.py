"""
Asset API Client - Unified interface for multiple free asset sources

This module provides a clean interface for searching and retrieving assets
from multiple free APIs including Pexels, Pixabay, Unsplash, Freesound, and Giphy.

Usage:
    api = AssetAPI()
    images = api.search_pexels_images("mountain landscape", per_page=10)
    videos = api.search_pixabay_videos("ocean waves", per_page=5)
    audio = api.search_freesound("ambient music", max_results=3)
    gifs = api.search_giphy_gifs("celebration", limit=20)
"""

import os
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AssetResult:
    """Standardized asset result structure"""
    id: str
    url: str
    thumbnail: str
    source: str
    asset_type: str
    metadata: Dict


class AssetAPI:
    """Unified API client for multiple free asset sources"""

    def __init__(self):
        """Initialize API client with credentials from environment variables"""
        self.pexels_key = os.getenv("PEXELS_API_KEY")
        self.unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY")
        self.pixabay_key = os.getenv("PIXABAY_API_KEY")
        self.freesound_key = os.getenv("FREESOUND_API_KEY")
        self.giphy_key = os.getenv("GIPHY_API_KEY")

        self._verify_credentials()

    def _verify_credentials(self):
        """Verify that required API keys are set"""
        missing_keys = []
        if not self.pexels_key:
            missing_keys.append("PEXELS_API_KEY")
        if not self.pixabay_key:
            missing_keys.append("PIXABAY_API_KEY")

        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")

    def search_pexels_images(self, query: str, per_page: int = 15) -> List[Dict]:
        """
        Search Pexels for high-quality images

        Args:
            query: Search keywords
            per_page: Number of results to return (max 80)

        Returns:
            List of image dictionaries with standardized structure
        """
        if not self.pexels_key:
            logger.error("PEXELS_API_KEY not set")
            return []

        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": self.pexels_key}
        params = {"query": query, "per_page": per_page}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return [{
                "id": str(photo["id"]),
                "url": photo["src"]["original"],
                "thumbnail": photo["src"]["medium"],
                "photographer": photo["photographer"],
                "source": "pexels",
                "type": "image",
                "width": photo["width"],
                "height": photo["height"],
                "alt": photo.get("alt", query)
            } for photo in data.get("photos", [])]

        except requests.exceptions.RequestException as e:
            logger.error(f"Pexels API error: {e}")
            return []

    def search_pexels_videos(self, query: str, per_page: int = 10) -> List[Dict]:
        """
        Search Pexels for video clips

        Args:
            query: Search keywords
            per_page: Number of results to return

        Returns:
            List of video dictionaries
        """
        if not self.pexels_key:
            logger.error("PEXELS_API_KEY not set")
            return []

        url = "https://api.pexels.com/videos/search"
        headers = {"Authorization": self.pexels_key}
        params = {"query": query, "per_page": per_page}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return [{
                "id": str(video["id"]),
                "url": video["video_files"][0]["link"] if video["video_files"] else "",
                "thumbnail": video["image"],
                "photographer": video["user"]["name"],
                "source": "pexels",
                "type": "video",
                "duration": video["duration"],
                "width": video["width"],
                "height": video["height"]
            } for video in data.get("videos", []) if video.get("video_files")]

        except requests.exceptions.RequestException as e:
            logger.error(f"Pexels Video API error: {e}")
            return []

    def search_pixabay_images(self, query: str, per_page: int = 20) -> List[Dict]:
        """
        Search Pixabay for images, vectors, and illustrations

        Args:
            query: Search keywords
            per_page: Number of results to return (max 200)

        Returns:
            List of image dictionaries
        """
        if not self.pixabay_key:
            logger.error("PIXABAY_API_KEY not set")
            return []

        url = "https://pixabay.com/api/"
        params = {
            "key": self.pixabay_key,
            "q": query,
            "per_page": per_page,
            "image_type": "all"
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return [{
                "id": str(hit["id"]),
                "url": hit["largeImageURL"],
                "thumbnail": hit["previewURL"],
                "photographer": hit["user"],
                "source": "pixabay",
                "type": "image",
                "width": hit["imageWidth"],
                "height": hit["imageHeight"],
                "tags": hit["tags"].split(", ")
            } for hit in data.get("hits", [])]

        except requests.exceptions.RequestException as e:
            logger.error(f"Pixabay API error: {e}")
            return []

    def search_pixabay_videos(self, query: str, per_page: int = 10) -> List[Dict]:
        """
        Search Pixabay for video content

        Args:
            query: Search keywords
            per_page: Number of results to return

        Returns:
            List of video dictionaries
        """
        if not self.pixabay_key:
            logger.error("PIXABAY_API_KEY not set")
            return []

        url = "https://pixabay.com/api/videos/"
        params = {
            "key": self.pixabay_key,
            "q": query,
            "per_page": per_page
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return [{
                "id": str(video["id"]),
                "url": video["videos"]["large"]["url"],
                "thumbnail": video["userImageURL"],
                "duration": video["duration"],
                "source": "pixabay",
                "type": "video",
                "width": video["videos"]["large"]["width"],
                "height": video["videos"]["large"]["height"],
                "tags": video["tags"].split(", ")
            } for video in data.get("hits", [])]

        except requests.exceptions.RequestException as e:
            logger.error(f"Pixabay Video API error: {e}")
            return []

    def search_freesound(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search Freesound for audio files and sound effects

        Args:
            query: Search keywords
            max_results: Maximum number of results

        Returns:
            List of audio dictionaries with licensing info
        """
        if not self.freesound_key:
            logger.error("FREESOUND_API_KEY not set")
            return []

        url = "https://freesound.org/apiv2/search/text/"
        params = {
            "query": query,
            "token": self.freesound_key,
            "page_size": max_results,
            "fields": "id,name,tags,duration,license,previews,username"
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return [{
                "id": str(result["id"]),
                "name": result["name"],
                "preview_url": result["previews"]["preview-hq-mp3"],
                "duration": result["duration"],
                "license": result["license"],
                "source": "freesound",
                "type": "audio",
                "tags": result.get("tags", []),
                "username": result.get("username", "")
            } for result in data.get("results", [])]

        except requests.exceptions.RequestException as e:
            logger.error(f"Freesound API error: {e}")
            return []

    def search_giphy_gifs(self, query: str, limit: int = 25) -> List[Dict]:
        """
        Search Giphy for GIF animations

        Args:
            query: Search keywords
            limit: Number of results (max 50)

        Returns:
            List of GIF dictionaries
        """
        if not self.giphy_key:
            logger.error("GIPHY_API_KEY not set")
            return []

        url = "https://api.giphy.com/v1/gifs/search"
        params = {
            "api_key": self.giphy_key,
            "q": query,
            "limit": limit,
            "rating": "g"
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return [{
                "id": gif["id"],
                "url": gif["images"]["original"]["url"],
                "thumbnail": gif["images"]["preview_gif"]["url"],
                "title": gif["title"],
                "source": "giphy",
                "type": "gif",
                "width": int(gif["images"]["original"]["width"]),
                "height": int(gif["images"]["original"]["height"])
            } for gif in data.get("data", [])]

        except requests.exceptions.RequestException as e:
            logger.error(f"Giphy API error: {e}")
            return []


def main():
    """Example usage of AssetAPI"""
    api = AssetAPI()

    print("🔍 Searching for mountain images...")
    images = api.search_pexels_images("mountain landscape", per_page=3)
    print(f"Found {len(images)} images from Pexels")

    print("\n🎥 Searching for ocean videos...")
    videos = api.search_pixabay_videos("ocean waves", per_page=2)
    print(f"Found {len(videos)} videos from Pixabay")

    print("\n🎵 Searching for ambient audio...")
    audio = api.search_freesound("ambient", max_results=2)
    print(f"Found {len(audio)} audio tracks from Freesound")

    print("\n🎬 Searching for celebration GIFs...")
    gifs = api.search_giphy_gifs("celebration", limit=3)
    print(f"Found {len(gifs)} GIFs from Giphy")


if __name__ == "__main__":
    main()
