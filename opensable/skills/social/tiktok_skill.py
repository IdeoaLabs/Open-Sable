"""
TikTok Skill,  Browse trending, search users/videos, get data from TikTok.

Uses the TikTokApi library (unofficial TikTok API wrapper).
This library is READ-ONLY,  it cannot post or upload content.
Uses Playwright for browser-based session creation.

Features:
- Get trending videos
- Search videos by keyword/hashtag
- Search users
- Get user profiles and videos
- Get video details and comments
- Get hashtag info and videos
- Download video content

Limitations:
    This API is designed to RETRIEVE data from TikTok only.
    It cannot post, upload, like, follow, or perform any write actions.
    For posting, consider using browser automation via the BrowserSkill.

Setup:
    Set in .env (optional, improves reliability):
        TIKTOK_MS_TOKEN=your_ms_token_cookie
        TIKTOK_BROWSER=chromium  (or firefox, webkit)

    Get ms_token:
        1. Login to tiktok.com in your browser
        2. Open DevTools → Application → Cookies
        3. Copy the 'msToken' cookie value

    Install:
        pip install TikTokApi
        python -m playwright install
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from TikTokApi import TikTokApi

    TIKTOKAPI_AVAILABLE = True
except ImportError:
    TIKTOKAPI_AVAILABLE = False
    logger.info(
        "TikTokApi not installed. Install with: "
        "pip install TikTokApi && python -m playwright install"
    )


class TikTokSkill:
    """
    TikTok data retrieval via TikTokApi (unofficial, read-only).
    Uses Playwright browser sessions for TikTok's anti-bot detection.
    All API methods are async. No posting/uploading capability.
    """

    def __init__(self, config):
        self.config = config
        self._api: Optional[Any] = None
        self._initialized = False
        self._action_delay = getattr(config, "tiktok_action_delay", 2.0)
        self._ms_token = None
        self._browser = None

    async def initialize(self) -> bool:
        """Initialize TikTok API with optional ms_token for better reliability."""
        if not TIKTOKAPI_AVAILABLE:
            logger.warning("TikTokApi not available,  TikTok skill disabled")
            return False

        # ── Close any existing session first (prevents Playwright browser leak) ──
        await self.close()

        self._ms_token = (
            getattr(self.config, "tiktok_ms_token", None)
            or os.getenv("TIKTOK_MS_TOKEN", "")
        ) or None

        self._browser = (
            getattr(self.config, "tiktok_browser", None)
            or os.getenv("TIKTOK_BROWSER", "chromium")
        )

        try:
            self._api = TikTokApi()
            ms_tokens = [self._ms_token] if self._ms_token else []

            # ── Mobile device session ──
            # TikTokApi uses Playwright browser sessions. We configure
            # mobile device emulation to match X skill's mobile-first approach.
            _mobile_context = {
                "user_agent": (
                    "Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.6778.200 Mobile Safari/537.36"
                ),
                "viewport": {"width": 412, "height": 915},
                "device_scale_factor": 2.625,
                "is_mobile": True,
                "has_touch": True,
            }

            await self._api.create_sessions(
                ms_tokens=ms_tokens,
                num_sessions=1,
                sleep_after=3,
                browser=self._browser,
                context_options=_mobile_context,
                # Mute audio,  prevents headless Chrome from playing TikTok video sound
                override_browser_args=["--mute-audio", "--autoplay-policy=user-gesture-required"],
                # Don't load media resources,  saves bandwidth and prevents audio/video playback
                suppress_resource_load_types=["media", "image", "font", "stylesheet"],
            )

            self._initialized = True
            logger.info("✅ TikTok: API session created")
            return True

        except Exception as e:
            logger.error(f"TikTok initialization failed: {e}")
            logger.info(
                "Hint: Run 'python -m playwright install' and set "
                "TIKTOK_MS_TOKEN in .env for better reliability"
            )
            return False

    def _ensure_initialized(self):
        if not self._initialized or not self._api:
            raise RuntimeError(
                "TikTok not initialized. Install TikTokApi + Playwright "
                "and optionally set TIKTOK_MS_TOKEN in .env"
            )

    # ──────────────────────────────────────────────────────────────────────
    # Trending
    # ──────────────────────────────────────────────────────────────────────

    async def get_trending_videos(self, count: int = 20) -> Dict[str, Any]:
        """
        Get trending videos on TikTok.

        Args:
            count: Number of videos to retrieve (default 20)
        """
        self._ensure_initialized()
        try:
            results = []
            async for video in self._api.trending.videos(count=count):
                d = video.as_dict
                author = d.get("author", {})
                stats = d.get("stats", {})
                results.append({
                    "id": d.get("id", ""),
                    "desc": (d.get("desc", "") or "")[:200],
                    "author": author.get("uniqueId", ""),
                    "author_nickname": author.get("nickname", ""),
                    "play_count": stats.get("playCount", 0),
                    "like_count": stats.get("diggCount", 0),
                    "comment_count": stats.get("commentCount", 0),
                    "share_count": stats.get("shareCount", 0),
                    "duration": d.get("video", {}).get("duration", 0),
                    "create_time": d.get("createTime", ""),
                    "url": f"https://www.tiktok.com/@{author.get('uniqueId', '')}/video/{d.get('id', '')}",
                })

            await asyncio.sleep(self._action_delay)
            return {
                "success": True,
                "count": len(results),
                "videos": results,
            }
        except Exception as e:
            logger.error(f"TikTok get_trending error: {e}")
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────────────────────────────

    async def search_videos(
        self, query: str, count: int = 10
    ) -> Dict[str, Any]:
        """
        Search for videos by keyword.

        Args:
            query: Search query
            count: Max results
        """
        self._ensure_initialized()
        try:
            results = []
            async for video in self._api.search.videos(query, count=count):
                d = video.as_dict
                author = d.get("author", {})
                stats = d.get("stats", {})
                results.append({
                    "id": d.get("id", ""),
                    "desc": (d.get("desc", "") or "")[:200],
                    "author": author.get("uniqueId", ""),
                    "play_count": stats.get("playCount", 0),
                    "like_count": stats.get("diggCount", 0),
                    "comment_count": stats.get("commentCount", 0),
                    "share_count": stats.get("shareCount", 0),
                    "url": f"https://www.tiktok.com/@{author.get('uniqueId', '')}/video/{d.get('id', '')}",
                })

            await asyncio.sleep(self._action_delay)
            return {
                "success": True,
                "query": query,
                "count": len(results),
                "videos": results,
            }
        except Exception as e:
            logger.error(f"TikTok search_videos error: {e}")
            return {"success": False, "error": str(e)}

    async def search_users(
        self, query: str, count: int = 10
    ) -> Dict[str, Any]:
        """
        Search for users by keyword.

        Args:
            query: Search query (username, name, etc.)
            count: Max results
        """
        self._ensure_initialized()
        try:
            results = []
            async for user in self._api.search.users(query, count=count):
                d = user.as_dict
                stats = d.get("stats", {})
                results.append({
                    "id": d.get("id", ""),
                    "unique_id": d.get("uniqueId", ""),
                    "nickname": d.get("nickname", ""),
                    "signature": (d.get("signature", "") or "")[:200],
                    "verified": d.get("verified", False),
                    "follower_count": stats.get("followerCount", 0),
                    "following_count": stats.get("followingCount", 0),
                    "heart_count": stats.get("heartCount", 0),
                    "video_count": stats.get("videoCount", 0),
                    "url": f"https://www.tiktok.com/@{d.get('uniqueId', '')}",
                })

            await asyncio.sleep(self._action_delay)
            return {
                "success": True,
                "query": query,
                "count": len(results),
                "users": results,
            }
        except Exception as e:
            logger.error(f"TikTok search_users error: {e}")
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # Users
    # ──────────────────────────────────────────────────────────────────────

    async def get_user_info(self, username: str) -> Dict[str, Any]:
        """
        Get a TikTok user's profile info.

        Args:
            username: TikTok username (without @)
        """
        self._ensure_initialized()
        try:
            user = self._api.user(username=username)
            user_data = await user.info()

            d = user_data
            user_info = d.get("userInfo", {}) or d
            user_obj = user_info.get("user", {})
            stats = user_info.get("stats", {})

            await asyncio.sleep(self._action_delay)
            return {
                "success": True,
                "id": user_obj.get("id", ""),
                "username": user_obj.get("uniqueId", username),
                "nickname": user_obj.get("nickname", ""),
                "bio": (user_obj.get("signature", "") or "")[:500],
                "verified": user_obj.get("verified", False),
                "private": user_obj.get("privateAccount", False),
                "follower_count": stats.get("followerCount", 0),
                "following_count": stats.get("followingCount", 0),
                "heart_count": stats.get("heartCount", 0),
                "video_count": stats.get("videoCount", 0),
                "avatar": user_obj.get("avatarLarger", ""),
                "url": f"https://www.tiktok.com/@{username}",
            }
        except Exception as e:
            logger.error(f"TikTok get_user_info error: {e}")
            return {"success": False, "error": str(e)}

    async def get_user_videos(
        self, username: str, count: int = 20
    ) -> Dict[str, Any]:
        """
        Get a user's posted videos.

        Args:
            username: TikTok username (without @)
            count: Max videos to return
        """
        self._ensure_initialized()
        try:
            user = self._api.user(username=username)
            results = []
            async for video in user.videos(count=count):
                d = video.as_dict
                stats = d.get("stats", {})
                results.append({
                    "id": d.get("id", ""),
                    "desc": (d.get("desc", "") or "")[:200],
                    "play_count": stats.get("playCount", 0),
                    "like_count": stats.get("diggCount", 0),
                    "comment_count": stats.get("commentCount", 0),
                    "share_count": stats.get("shareCount", 0),
                    "duration": d.get("video", {}).get("duration", 0),
                    "create_time": d.get("createTime", ""),
                    "url": f"https://www.tiktok.com/@{username}/video/{d.get('id', '')}",
                })

            await asyncio.sleep(self._action_delay)
            return {
                "success": True,
                "username": username,
                "count": len(results),
                "videos": results,
            }
        except Exception as e:
            logger.error(f"TikTok get_user_videos error: {e}")
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # Videos
    # ──────────────────────────────────────────────────────────────────────

    async def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """
        Get info about a specific video by ID.

        Args:
            video_id: TikTok video ID
        """
        self._ensure_initialized()
        try:
            video = self._api.video(id=video_id)
            info = await video.info()

            d = info
            author = d.get("author", {})
            stats = d.get("stats", {})
            music = d.get("music", {})

            await asyncio.sleep(self._action_delay)
            return {
                "success": True,
                "id": d.get("id", video_id),
                "desc": (d.get("desc", "") or "")[:500],
                "author": author.get("uniqueId", ""),
                "author_nickname": author.get("nickname", ""),
                "play_count": stats.get("playCount", 0),
                "like_count": stats.get("diggCount", 0),
                "comment_count": stats.get("commentCount", 0),
                "share_count": stats.get("shareCount", 0),
                "duration": d.get("video", {}).get("duration", 0),
                "create_time": d.get("createTime", ""),
                "music_title": music.get("title", ""),
                "music_author": music.get("authorName", ""),
                "url": f"https://www.tiktok.com/@{author.get('uniqueId', '')}/video/{d.get('id', video_id)}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_video_comments(
        self, video_id: str, count: int = 20
    ) -> Dict[str, Any]:
        """
        Get comments on a video.

        Args:
            video_id: TikTok video ID
            count: Max comments to return
        """
        self._ensure_initialized()
        try:
            video = self._api.video(id=video_id)
            results = []
            async for comment in video.comments(count=count):
                d = comment.as_dict
                user = d.get("user", {})
                results.append({
                    "id": d.get("cid", ""),
                    "text": d.get("text", ""),
                    "username": user.get("unique_id", "") or user.get("uniqueId", ""),
                    "nickname": user.get("nickname", ""),
                    "likes": d.get("digg_count", 0),
                    "reply_count": d.get("reply_comment_total", 0),
                    "create_time": d.get("create_time", ""),
                })

            await asyncio.sleep(self._action_delay)
            return {
                "success": True,
                "video_id": video_id,
                "count": len(results),
                "comments": results,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # Hashtags
    # ──────────────────────────────────────────────────────────────────────

    async def get_hashtag_info(self, hashtag: str) -> Dict[str, Any]:
        """
        Get info about a hashtag.

        Args:
            hashtag: Hashtag name (without #)
        """
        self._ensure_initialized()
        try:
            tag = self._api.hashtag(name=hashtag)
            info = await tag.info()

            d = info.get("challengeInfo", {}) or info
            challenge = d.get("challenge", {}) or {}
            stats = d.get("stats", {}) or {}

            await asyncio.sleep(self._action_delay)
            return {
                "success": True,
                "id": challenge.get("id", ""),
                "title": challenge.get("title", hashtag),
                "desc": (challenge.get("desc", "") or "")[:300],
                "video_count": stats.get("videoCount", 0),
                "view_count": stats.get("viewCount", 0),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_hashtag_videos(
        self, hashtag: str, count: int = 20
    ) -> Dict[str, Any]:
        """
        Get videos for a hashtag.

        Args:
            hashtag: Hashtag name (without #)
            count: Max videos to return
        """
        self._ensure_initialized()
        try:
            tag = self._api.hashtag(name=hashtag)
            results = []
            async for video in tag.videos(count=count):
                d = video.as_dict
                author = d.get("author", {})
                stats = d.get("stats", {})
                results.append({
                    "id": d.get("id", ""),
                    "desc": (d.get("desc", "") or "")[:200],
                    "author": author.get("uniqueId", ""),
                    "play_count": stats.get("playCount", 0),
                    "like_count": stats.get("diggCount", 0),
                    "comment_count": stats.get("commentCount", 0),
                    "url": f"https://www.tiktok.com/@{author.get('uniqueId', '')}/video/{d.get('id', '')}",
                })

            await asyncio.sleep(self._action_delay)
            return {
                "success": True,
                "hashtag": hashtag,
                "count": len(results),
                "videos": results,
            }
        except Exception as e:
            logger.error(f"TikTok get_hashtag_videos error: {e}")
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # Sound / Music
    # ──────────────────────────────────────────────────────────────────────

    async def get_sound_videos(
        self, sound_id: str, count: int = 20
    ) -> Dict[str, Any]:
        """
        Get videos using a specific sound/music.

        Args:
            sound_id: TikTok sound/music ID
            count: Max videos to return
        """
        self._ensure_initialized()
        try:
            sound = self._api.sound(id=sound_id)
            results = []
            async for video in sound.videos(count=count):
                d = video.as_dict
                author = d.get("author", {})
                stats = d.get("stats", {})
                results.append({
                    "id": d.get("id", ""),
                    "desc": (d.get("desc", "") or "")[:200],
                    "author": author.get("uniqueId", ""),
                    "play_count": stats.get("playCount", 0),
                    "like_count": stats.get("diggCount", 0),
                    "url": f"https://www.tiktok.com/@{author.get('uniqueId', '')}/video/{d.get('id', '')}",
                })

            return {
                "success": True,
                "sound_id": sound_id,
                "count": len(results),
                "videos": results,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # Download
    # ──────────────────────────────────────────────────────────────────────

    async def download_video(
        self, video_id: str, path: str = "/tmp"
    ) -> Dict[str, Any]:
        """
        Download a TikTok video to a local file.

        Args:
            video_id: TikTok video ID
            path: Directory to save the video to
        """
        self._ensure_initialized()
        try:
            video = self._api.video(id=video_id)
            video_bytes = await video.bytes()

            output_path = os.path.join(path, f"tiktok_{video_id}.mp4")
            with open(output_path, "wb") as f:
                f.write(video_bytes)

            return {
                "success": True,
                "path": output_path,
                "size_bytes": len(video_bytes),
                "video_id": video_id,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────

    async def close(self):
        """Close the TikTok API sessions (Playwright browser cleanup)."""
        if self._api:
            try:
                await self._api.close_sessions()
                logger.info("TikTok: Sessions closed")
            except Exception as exc:
                logger.debug(f"TikTok close error: {exc}")
            finally:
                self._api = None
                self._initialized = False

    def __del__(self):
        """Fallback cleanup, try to close Playwright if the event loop is running."""
        if self._api:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
            except Exception:
                pass

    def is_available(self) -> bool:
        """Check if TikTok skill is initialized and ready."""
        return self._initialized and self._api is not None
