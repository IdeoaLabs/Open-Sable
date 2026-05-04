"""
X (Twitter) Skill,  Post, search, interact on X using twikit.

Uses the twikit library for full X/Twitter automation without paid API keys.
Authenticates with your X account credentials (username + email + password).

Features:
- Post tweets (text, images, video)
- Search tweets by keyword
- Get trending topics
- Like, retweet, reply, quote
- Follow/unfollow users
- Send DMs
- Get user profiles and timelines
- Bookmark tweets
- Rate-limit aware with automatic delays

Setup:
    Set these in .env (same credentials as Grok):
        X_USERNAME=your_x_username
        X_EMAIL=your_x_email
        X_PASSWORD=your_x_password
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from twikit.media import Photo, Video, AnimatedGif
    _MEDIA_CLASSES = True
except ImportError:
    _MEDIA_CLASSES = False

try:
    from twikit import Client as TwikitClient

    TWIKIT_AVAILABLE = True
except ImportError:
    TWIKIT_AVAILABLE = False
    logger.info("twikit not installed. Install with: pip install twikit")


class XSkill:
    """
    Full X/Twitter automation,  post, search, engage, all for free via twikit.

    Shares X authentication cookies with GrokSkill.
    """

    def __init__(self, config):
        self.config = config
        self._client: Optional[Any] = None
        _profile = os.environ.get("_SABLE_PROFILE", "")
        _cookie_name = f"x_cookies_{_profile}.json" if _profile else "x_cookies.json"
        self._cookies_path = Path.home() / ".opensable" / _cookie_name
        self._cookies_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        # Default delay between actions (seconds) to avoid rate limits
        self._action_delay = getattr(config, "x_action_delay", 2)

    async def initialize(self) -> bool:
        """Initialize and authenticate with X."""
        if not TWIKIT_AVAILABLE:
            logger.warning("twikit not available,  X skill disabled")
            return False

        try:
            lang = getattr(self.config, "x_language", "en-US") or "en-US"

            # Use mobile user agent to avoid bot detection
            from opensable.core.x_self_heal import pick_user_agent
            ua = pick_user_agent(prefer_mobile=True)
            self._client = TwikitClient(lang, user_agent=ua)
            logger.info(f"X: Using UA: {ua[:60]}...")

            # Patch TLS fingerprint to match Android Chrome
            # (without this, X sees Linux TLS fingerprint despite mobile UA)
            from opensable.core.tls_patch import patch_twikit_client
            proxy = getattr(self.config, "x_proxy", None) or os.getenv("X_PROXY")
            tls_ok = patch_twikit_client(self._client, proxy=proxy)
            if not tls_ok:
                logger.warning("TLS patch not applied,  falling back to httpx (detectable)")
                # Fallback: at least set UA on httpx headers
                if hasattr(self._client, 'http') and hasattr(self._client.http, 'headers'):
                    self._client.http.headers["user-agent"] = ua

            # Try loading saved cookies first
            if self._cookies_path.exists():
                self._client.load_cookies(str(self._cookies_path))
                logger.info("✅ X: Loaded saved cookies")
                self._initialized = True
                return True

            # Login with credentials
            username = getattr(self.config, "x_username", None) or os.getenv("X_USERNAME")
            email = getattr(self.config, "x_email", None) or os.getenv("X_EMAIL")
            password = getattr(self.config, "x_password", None) or os.getenv("X_PASSWORD")

            if not all([username, password]):
                logger.warning("X: Missing credentials. Set X_USERNAME, X_EMAIL, X_PASSWORD in .env")
                return False

            await self._client.login(
                auth_info_1=username,
                auth_info_2=email,
                password=password,
            )

            # Save cookies for reuse (shared with GrokSkill)
            self._client.save_cookies(str(self._cookies_path))
            logger.info("✅ X: Authenticated and cookies saved")
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"X initialization failed: {e}")
            return False

    def _ensure_initialized(self):
        if not self._initialized or not self._client:
            raise RuntimeError(
                "X not initialized. Check credentials in .env "
                "(X_USERNAME, X_EMAIL, X_PASSWORD)"
            )

    # ──────────────────────────────────────────────────────────────────────
    # Media extraction
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_media(tweet_obj) -> List[Dict[str, Any]]:
        """Extract media info (photos, videos, GIFs) from a twikit Tweet object."""
        media_list = []
        if not _MEDIA_CLASSES:
            return media_list

        try:
            raw_media = getattr(tweet_obj, "media", None)
            if not raw_media:
                # Also check for card thumbnails (link previews)
                thumb_url = getattr(tweet_obj, "thumbnail_url", None)
                if thumb_url:
                    media_list.append({
                        "type": "thumbnail",
                        "url": thumb_url,
                        "title": getattr(tweet_obj, "thumbnail_title", None),
                    })
                return media_list

            for m in raw_media:
                entry: Dict[str, Any] = {
                    "type": "unknown",
                    "url": getattr(m, "media_url", None),
                    "width": getattr(m, "width", None),
                    "height": getattr(m, "height", None),
                }
                if isinstance(m, Photo):
                    entry["type"] = "photo"
                elif isinstance(m, Video):
                    entry["type"] = "video"
                    entry["duration_ms"] = getattr(m, "duration_millis", None)
                elif isinstance(m, AnimatedGif):
                    entry["type"] = "gif"
                media_list.append(entry)

            # Also grab thumbnail if present (card + embedded media)
            if not media_list:
                thumb_url = getattr(tweet_obj, "thumbnail_url", None)
                if thumb_url:
                    media_list.append({
                        "type": "thumbnail",
                        "url": thumb_url,
                        "title": getattr(tweet_obj, "thumbnail_title", None),
                    })
        except Exception as e:
            logger.debug(f"Media extraction failed: {e}")

        return media_list

    @staticmethod
    async def download_media_url(url: str, suffix: str = ".jpg") -> Optional[str]:
        """Download a media URL to a temp file. Returns the local path or None."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as http:
                resp = await http.get(url)
                resp.raise_for_status()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
                tmp.write(resp.content)
                tmp.close()
                return tmp.name
        except Exception as e:
            logger.debug(f"Media download failed ({url[:60]}): {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Posting
    # ──────────────────────────────────────────────────────────────────────

    async def post_tweet(
        self,
        text: str,
        media_paths: Optional[List[str]] = None,
        reply_to: Optional[str] = None,
        quote_tweet_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Post a tweet on X.

        Args:
            text: Tweet text (max 280 chars for free accounts, 25000 for Premium)
            media_paths: Optional list of image/video file paths to attach
            reply_to: Tweet ID to reply to (optional)
            quote_tweet_id: Tweet ID to quote (optional)

        Returns:
            Dict with tweet info or error
        """
        self._ensure_initialized()

        try:
            # Upload media if provided
            media_ids = []
            if media_paths:
                for path in media_paths:
                    if os.path.exists(path):
                        media_id = await self._client.upload_media(path)
                        media_ids.append(media_id)
                        logger.info(f"X: Uploaded media {path}")
                    else:
                        logger.warning(f"X: Media file not found: {path}")

            # Build kwargs
            kwargs = {"text": text}
            if media_ids:
                kwargs["media_ids"] = media_ids
            if reply_to:
                kwargs["reply_to"] = reply_to
            if quote_tweet_id:
                kwargs["quote_tweet_id"] = quote_tweet_id

            tweet = await self._client.create_tweet(**kwargs)

            await asyncio.sleep(self._action_delay)

            if tweet is None:
                # twikit's tweet_from_data() returns None for tombstones or
                # incomplete API responses — common under 226 shadow/shadowban.
                # Tweet may have been delivered to X but the ID is unreadable.
                logger.warning("⚠️ X: create_tweet returned None — tombstone or 226 shadow active")
                return {"success": False, "error": "tombstoned", "tombstone": True}

            tweet_id = getattr(tweet, "id", None)
            if not tweet_id:
                logger.warning(f"⚠️ X: tweet object has no ID — {tweet!r}")
                return {"success": False, "error": "no_tweet_id", "tombstone": True}

            logger.info(f"✅ X: Posted tweet {tweet_id}")

            return {
                "success": True,
                "tweet_id": tweet_id,
                "text": text,
                "media_count": len(media_ids),
                "url": f"https://x.com/i/status/{tweet_id}",
            }

        except Exception as e:
            logger.error(f"X post_tweet error: {e}")
            return {"success": False, "error": str(e)}

    async def post_thread(self, tweets: List[str]) -> Dict[str, Any]:
        """
        Post a thread (multiple connected tweets).

        Args:
            tweets: List of tweet texts in order

        Returns:
            Dict with all tweet IDs
        """
        self._ensure_initialized()

        try:
            results = []
            reply_to = None

            for i, text in enumerate(tweets):
                kwargs = {"text": text}
                if reply_to:
                    kwargs["reply_to"] = reply_to

                tweet = await self._client.create_tweet(**kwargs)
                tweet_id = getattr(tweet, "id", None) or str(tweet)
                reply_to = tweet_id

                results.append({
                    "index": i,
                    "tweet_id": tweet_id,
                    "text": text[:50] + "..." if len(text) > 50 else text,
                })

                logger.info(f"X: Thread [{i+1}/{len(tweets)}] posted: {tweet_id}")
                await asyncio.sleep(self._action_delay)

            return {
                "success": True,
                "thread_length": len(results),
                "tweets": results,
                "thread_url": f"https://x.com/i/status/{results[0]['tweet_id']}" if results else None,
            }

        except Exception as e:
            logger.error(f"X post_thread error: {e}")
            return {"success": False, "error": str(e), "posted": len(results) if 'results' in dir() else 0}

    # ──────────────────────────────────────────────────────────────────────
    # Search & Discovery
    # ──────────────────────────────────────────────────────────────────────

    async def search_tweets(
        self,
        query: str,
        search_type: str = "Latest",
        count: int = 10,
    ) -> Dict[str, Any]:
        """
        Search tweets by keyword.

        Args:
            query: Search query
            search_type: 'Latest', 'Top', 'People', 'Media'
            count: Max results to return
        """
        self._ensure_initialized()

        try:
            tweets = await self._client.search_tweet(query, search_type)

            results = []
            for i, tweet in enumerate(tweets):
                if i >= count:
                    break
                media = self._extract_media(tweet)
                results.append({
                    "id": getattr(tweet, "id", None),
                    "text": getattr(tweet, "text", str(tweet)),
                    "user": getattr(getattr(tweet, "user", None), "name", "Unknown"),
                    "username": getattr(getattr(tweet, "user", None), "screen_name", ""),
                    "created_at": str(getattr(tweet, "created_at", "")),
                    "likes": getattr(tweet, "favorite_count", 0),
                    "retweets": getattr(tweet, "retweet_count", 0),
                    "media": media,
                    "has_media": bool(media),
                })

            await asyncio.sleep(self._action_delay)

            return {
                "success": True,
                "query": query,
                "count": len(results),
                "tweets": results,
            }

        except Exception as e:
            logger.error(f"X search error: {e}")
            return {"success": False, "error": str(e)}

    async def get_home_timeline(
        self,
        count: int = 20,
        tab: str = "latest",
    ) -> Dict[str, Any]:
        """
        Read the home timeline,  like a real user opening the app and scrolling.

        Args:
            count: Max tweets to return
            tab: 'latest' (Following) or 'foryou' (For You)
        """
        self._ensure_initialized()

        try:
            if tab == "latest":
                raw_tweets = await self._client.get_latest_timeline(count=count)
            else:
                raw_tweets = await self._client.get_timeline(count=count)

            results = []
            for i, tweet in enumerate(raw_tweets):
                if i >= count:
                    break
                media = self._extract_media(tweet)
                results.append({
                    "id": getattr(tweet, "id", None),
                    "text": getattr(tweet, "text", str(tweet)),
                    "user": getattr(getattr(tweet, "user", None), "name", "Unknown"),
                    "username": getattr(getattr(tweet, "user", None), "screen_name", ""),
                    "created_at": str(getattr(tweet, "created_at", "")),
                    "likes": getattr(tweet, "favorite_count", 0),
                    "retweets": getattr(tweet, "retweet_count", 0),
                    "media": media,
                    "has_media": bool(media),
                })

            await asyncio.sleep(self._action_delay)

            return {
                "success": True,
                "tab": tab,
                "count": len(results),
                "tweets": results,
            }

        except Exception as e:
            logger.error(f"X timeline error: {e}")
            return {"success": False, "error": str(e)}

    async def get_notifications(
        self,
        notification_type: str = "Mentions",
        count: int = 10,
    ) -> Dict[str, Any]:
        """
        Get notifications (mentions, likes, etc.),  the real notifications tab.

        Args:
            notification_type: 'All', 'Verified', or 'Mentions'
            count: Max notifications to return
        """
        self._ensure_initialized()

        try:
            notifications = await self._client.get_notifications(notification_type, count=count)

            results = []
            for i, notif in enumerate(notifications):
                if i >= count:
                    break
                tweet = getattr(notif, "tweet", None)
                from_user = getattr(notif, "from_user", None)
                results.append({
                    "id": getattr(notif, "id", None),
                    "message": getattr(notif, "message", ""),
                    "timestamp_ms": getattr(notif, "timestamp_ms", 0),
                    "tweet_id": getattr(tweet, "id", None) if tweet else None,
                    "tweet_text": getattr(tweet, "text", "") if tweet else "",
                    "username": getattr(from_user, "screen_name", "") if from_user else "",
                    "user_name": getattr(from_user, "name", "") if from_user else "",
                })

            await asyncio.sleep(self._action_delay)

            return {
                "success": True,
                "type": notification_type,
                "count": len(results),
                "notifications": results,
            }

        except Exception as e:
            logger.error(f"X notifications error: {e}")
            return {"success": False, "error": str(e)}

    async def get_trends(self, category: str = "trending") -> Dict[str, Any]:
        """
        Get trending topics on X.

        Args:
            category: 'trending', 'news', 'sports', 'entertainment'
        """
        self._ensure_initialized()

        try:
            trends = await self._client.get_trends(category)

            results = []
            for trend in trends:
                results.append({
                    "name": getattr(trend, "name", str(trend)),
                    "tweet_count": getattr(trend, "tweet_count", None),
                })

            return {
                "success": True,
                "category": category,
                "trends": results[:20],
            }

        except Exception as e:
            logger.error(f"X trends error: {e}")
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # Engagement
    # ──────────────────────────────────────────────────────────────────────

    async def like_tweet(self, tweet_id: str) -> Dict[str, Any]:
        """Like a tweet."""
        self._ensure_initialized()
        try:
            await self._client.like_tweet(tweet_id)
            await asyncio.sleep(self._action_delay)
            return {"success": True, "action": "liked", "tweet_id": tweet_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def retweet(self, tweet_id: str) -> Dict[str, Any]:
        """Retweet a tweet."""
        self._ensure_initialized()
        try:
            await self._client.retweet(tweet_id)
            await asyncio.sleep(self._action_delay)
            return {"success": True, "action": "retweeted", "tweet_id": tweet_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def reply(self, tweet_id: str, text: str) -> Dict[str, Any]:
        """Reply to a tweet."""
        return await self.post_tweet(text=text, reply_to=tweet_id)

    async def quote_tweet(self, tweet_id: str, text: str) -> Dict[str, Any]:
        """Quote a tweet with your own text."""
        return await self.post_tweet(text=text, quote_tweet_id=tweet_id)

    async def bookmark_tweet(self, tweet_id: str) -> Dict[str, Any]:
        """Bookmark a tweet."""
        self._ensure_initialized()
        try:
            await self._client.bookmark_tweet(tweet_id)
            await asyncio.sleep(self._action_delay)
            return {"success": True, "action": "bookmarked", "tweet_id": tweet_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_tweet_replies(self, tweet_id: str, count: int = 20) -> Dict[str, Any]:
        """
        Get replies to a specific tweet using get_tweet_by_id.
        twikit returns reply threads as part of the tweet detail response.

        Args:
            tweet_id: The tweet ID to fetch replies for.
            count: Max replies to return.

        Returns:
            Dict with success flag and list of reply dicts.
        """
        self._ensure_initialized()
        try:
            tweet = await self._client.get_tweet_by_id(tweet_id)
            if tweet is None:
                return {"success": False, "error": "Tweet not found"}

            # The tweet detail response groups replies into conversation threads.
            # Each thread has a root reply with optional .replies sub-list.
            # We flatten them into a single list.
            results = []
            # Direct replies appear as the tweet object's thread entries
            # twikit stores them via the entries parsing in get_tweet_by_id
            if hasattr(tweet, 'replies') and tweet.replies:
                for r in tweet.replies:
                    if len(results) >= count:
                        break
                    results.append({
                        "id": getattr(r, "id", None),
                        "text": getattr(r, "text", str(r)),
                        "username": getattr(getattr(r, "user", None), "screen_name", ""),
                        "user_name": getattr(getattr(r, "user", None), "name", ""),
                        "created_at": str(getattr(r, "created_at", "")),
                        "likes": getattr(r, "favorite_count", 0),
                        "reply_count": getattr(r, "reply_count", 0),
                    })
                    # Also check nested replies within this thread
                    if hasattr(r, 'replies') and r.replies:
                        for sub in r.replies:
                            if len(results) >= count:
                                break
                            results.append({
                                "id": getattr(sub, "id", None),
                                "text": getattr(sub, "text", str(sub)),
                                "username": getattr(getattr(sub, "user", None), "screen_name", ""),
                                "user_name": getattr(getattr(sub, "user", None), "name", ""),
                                "created_at": str(getattr(sub, "created_at", "")),
                                "likes": getattr(sub, "favorite_count", 0),
                                "reply_count": getattr(sub, "reply_count", 0),
                            })

            await asyncio.sleep(self._action_delay)

            return {
                "success": True,
                "tweet_id": tweet_id,
                "original_text": getattr(tweet, "text", ""),
                "count": len(results),
                "replies": results,
            }

        except Exception as e:
            logger.error(f"X get_tweet_replies error: {e}")
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # Users
    # ──────────────────────────────────────────────────────────────────────

    async def get_user(self, username: str) -> Dict[str, Any]:
        """Get a user's profile info."""
        self._ensure_initialized()
        try:
            user = await self._client.get_user_by_screen_name(username)
            await asyncio.sleep(self._action_delay)
            return {
                "success": True,
                "id": getattr(user, "id", None),
                "name": getattr(user, "name", ""),
                "username": getattr(user, "screen_name", username),
                "bio": getattr(user, "description", ""),
                "followers": getattr(user, "followers_count", 0),
                "following": getattr(user, "following_count", 0),
                "tweets_count": getattr(user, "statuses_count", 0),
                "verified": getattr(user, "verified", False),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_user_tweets(
        self, username: str, tweet_type: str = "Tweets", count: int = 10
    ) -> Dict[str, Any]:
        """
        Get tweets from a specific user.

        Args:
            username: X username (without @)
            tweet_type: 'Tweets', 'Replies', 'Media', 'Likes'
            count: Max tweets to return
        """
        self._ensure_initialized()
        try:
            user = await self._client.get_user_by_screen_name(username)
            user_id = getattr(user, "id", None)
            if not user_id:
                return {"success": False, "error": f"User @{username} not found"}

            tweets = await self._client.get_user_tweets(user_id, tweet_type)

            results = []
            for i, tweet in enumerate(tweets):
                if i >= count:
                    break
                media = self._extract_media(tweet)
                results.append({
                    "id": getattr(tweet, "id", None),
                    "text": getattr(tweet, "text", str(tweet)),
                    "created_at": str(getattr(tweet, "created_at", "")),
                    "likes": getattr(tweet, "favorite_count", 0),
                    "retweets": getattr(tweet, "retweet_count", 0),
                    "media": media,
                    "has_media": bool(media),
                })

            await asyncio.sleep(self._action_delay)

            return {
                "success": True,
                "username": username,
                "count": len(results),
                "tweets": results,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def follow_user(self, username: str) -> Dict[str, Any]:
        """Follow a user."""
        self._ensure_initialized()
        try:
            user = await self._client.get_user_by_screen_name(username)
            user_id = getattr(user, "id", None)
            if user_id:
                await self._client.follow_user(user_id)
                await asyncio.sleep(self._action_delay)
                return {"success": True, "action": "followed", "username": username}
            return {"success": False, "error": f"User @{username} not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def unfollow_user(self, username: str) -> Dict[str, Any]:
        """Unfollow a user."""
        self._ensure_initialized()
        try:
            user = await self._client.get_user_by_screen_name(username)
            user_id = getattr(user, "id", None)
            if user_id:
                await self._client.unfollow_user(user_id)
                await asyncio.sleep(self._action_delay)
                return {"success": True, "action": "unfollowed", "username": username}
            return {"success": False, "error": f"User @{username} not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # Direct Messages
    # ──────────────────────────────────────────────────────────────────────

    async def send_dm(self, user_id: str, text: str) -> Dict[str, Any]:
        """
        Send a direct message.

        Args:
            user_id: User ID (numeric) to send DM to
            text: Message text
        """
        self._ensure_initialized()
        try:
            await self._client.send_dm(user_id, text)
            await asyncio.sleep(self._action_delay)
            return {"success": True, "action": "dm_sent", "user_id": user_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────

    async def delete_tweet(self, tweet_id: str) -> Dict[str, Any]:
        """Delete one of your own tweets."""
        self._ensure_initialized()
        try:
            await self._client.delete_tweet(tweet_id)
            await asyncio.sleep(self._action_delay)
            return {"success": True, "action": "deleted", "tweet_id": tweet_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def is_available(self) -> bool:
        """Check if X skill is initialized and ready."""
        return self._initialized and self._client is not None
