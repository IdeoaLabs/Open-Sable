"""
Zunvra social network tool implementations,  mixin for ToolRegistry.
"""

import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ZunvraToolsMixin:
    """Tool handlers for the Zunvra social skill."""

    # ── Social ────────────────────────────────────────────────────────────

    async def _zunvra_post_tool(self, params: Dict) -> str:
        """Create a post on Zunvra."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected. Set ZUNVRA_GATEWAY_URL and ZUNVRA_API_KEY in profile.env"
        content = params.get("content", "")
        if not content:
            return "⚠️ Post content is required"
        result = await self.zunvra_skill.create_post(
            content,
            media_urls=params.get("media_urls"),
            tags=params.get("tags"),
        )
        if result.get("error"):
            return f"❌ Post failed: {result['error']}"
        return f"✅ Posted on Zunvra!\n📝 {content[:120]}{'…' if len(content) > 120 else ''}"

    async def _zunvra_reply_tool(self, params: Dict) -> str:
        """Reply to a post on Zunvra."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        post_id = params.get("post_id", "")
        content = params.get("content", "")
        if not post_id or not content:
            return "⚠️ post_id and content are required"
        result = await self.zunvra_skill.reply(post_id, content)
        if result.get("error"):
            return f"❌ Reply failed: {result['error']}"
        return f"✅ Replied on Zunvra (post {post_id[:8]}…)"

    async def _zunvra_like_tool(self, params: Dict) -> str:
        """Like a post on Zunvra."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        post_id = params.get("post_id", "")
        if not post_id:
            return "⚠️ post_id is required"
        result = await self.zunvra_skill.like(post_id)
        if result.get("error"):
            return f"❌ Like failed: {result['error']}"
        return f"❤️ Liked post {post_id[:8]}… on Zunvra"

    async def _zunvra_repost_tool(self, params: Dict) -> str:
        """Repost on Zunvra."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        post_id = params.get("post_id", "")
        if not post_id:
            return "⚠️ post_id is required"
        result = await self.zunvra_skill.repost(post_id)
        if result.get("error"):
            return f"❌ Repost failed: {result['error']}"
        return f"🔁 Reposted {post_id[:8]}… on Zunvra"

    async def _zunvra_follow_tool(self, params: Dict) -> str:
        """Follow a user on Zunvra."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        user_id = params.get("user_id", "")
        if not user_id:
            return "⚠️ user_id is required"
        result = await self.zunvra_skill.follow(user_id)
        if result.get("error"):
            return f"❌ Follow failed: {result['error']}"
        return f"➕ Followed user {user_id[:8]}… on Zunvra"

    async def _zunvra_unfollow_tool(self, params: Dict) -> str:
        """Unfollow a user on Zunvra."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        user_id = params.get("user_id", "")
        if not user_id:
            return "⚠️ user_id is required"
        result = await self.zunvra_skill.unfollow(user_id)
        if result.get("error"):
            return f"❌ Unfollow failed: {result['error']}"
        return f"➖ Unfollowed user {user_id[:8]}… on Zunvra"

    # ── Feed & Discovery ──────────────────────────────────────────────────

    async def _zunvra_feed_tool(self, params: Dict) -> str:
        """Get the Zunvra feed."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        page = params.get("page", 1)
        limit = params.get("limit", 20)
        result = await self.zunvra_skill.get_feed(page=page, limit=limit)
        if result.get("error"):
            return f"❌ Feed failed: {result['error']}"
        posts = result.get("posts") or result.get("data") or []
        if not posts:
            return "📭 No posts in your Zunvra feed right now."
        lines = []
        for p in posts[:limit]:
            author = p.get("username") or p.get("user", {}).get("username", "?")
            text = (p.get("content") or "")[:140]
            likes = p.get("likes_count", p.get("likes", 0))
            pid = p.get("id", "?")
            lines.append(f"  @{author}: {text}")
            lines.append(f"    ❤️ {likes} | 🆔 {pid}")
        return f"📰 Zunvra Feed (page {page}):\n" + "\n".join(lines)

    async def _zunvra_trending_tool(self, params: Dict) -> str:
        """Get trending on Zunvra."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        result = await self.zunvra_skill.get_trending()
        if result.get("error"):
            return f"❌ Trending failed: {result['error']}"
        items = result.get("posts") or result.get("data") or result.get("trending") or []
        if not items:
            return "🔥 No trending content on Zunvra right now."
        lines = []
        for i, p in enumerate(items[:10], 1):
            text = (p.get("content") or p.get("title") or "")[:100]
            author = p.get("username") or p.get("user", {}).get("username", "?")
            lines.append(f"  {i}. @{author}: {text}")
        return "🔥 Trending on Zunvra:\n" + "\n".join(lines)

    async def _zunvra_get_user_tool(self, params: Dict) -> str:
        """Get a user profile on Zunvra."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        username = params.get("username", "")
        if not username:
            return "⚠️ username is required"
        result = await self.zunvra_skill.get_user(username)
        if result.get("error"):
            return f"❌ User lookup failed: {result['error']}"
        u = result.get("user") or result
        return (
            f"👤 @{u.get('username', username)}\n"
            f"   Name: {u.get('display_name', u.get('name', '?'))}\n"
            f"   Bio: {(u.get('bio') or ', ')[:200]}\n"
            f"   Followers: {u.get('followers_count', '?')} | Following: {u.get('following_count', '?')}\n"
            f"   Posts: {u.get('posts_count', '?')}\n"
            f"   ID: {u.get('id', '?')}"
        )

    async def _zunvra_get_post_tool(self, params: Dict) -> str:
        """Get a specific Zunvra post."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        post_id = params.get("post_id", "")
        if not post_id:
            return "⚠️ post_id is required"
        result = await self.zunvra_skill.get_post(post_id)
        if result.get("error"):
            return f"❌ Post lookup failed: {result['error']}"
        p = result.get("post") or result
        author = p.get("username") or p.get("user", {}).get("username", "?")
        content = p.get("content", "")[:500]
        likes = p.get("likes_count", p.get("likes", 0))
        replies_count = p.get("replies_count", "?")
        return (
            f"📄 Post by @{author}:\n"
            f"   {content}\n"
            f"   ❤️ {likes} | 💬 {replies_count} replies | 🆔 {post_id}"
        )

    # ── Messaging ─────────────────────────────────────────────────────────

    async def _zunvra_send_dm_tool(self, params: Dict) -> str:
        """Send a DM on Zunvra."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        receiver_id = params.get("receiver_id", "")
        content = params.get("content", "")
        if not receiver_id or not content:
            return "⚠️ receiver_id and content are required"
        result = await self.zunvra_skill.send_dm(receiver_id, content)
        if result.get("error"):
            return f"❌ DM failed: {result['error']}"
        return f"✉️ DM sent to {receiver_id[:8]}… on Zunvra"

    async def _zunvra_conversations_tool(self, params: Dict) -> str:
        """List conversations on Zunvra."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        result = await self.zunvra_skill.get_conversations()
        if result.get("error"):
            return f"❌ Conversations failed: {result['error']}"
        convos = result.get("conversations") or result.get("data") or []
        if not convos:
            return "💬 No conversations on Zunvra yet."
        lines = []
        for c in convos[:15]:
            other = c.get("other_user", {}).get("username", c.get("participant", "?"))
            last = (c.get("last_message", {}).get("content") or "")[:60]
            lines.append(f"  💬 @{other}: {last}")
        return "💬 Zunvra Conversations:\n" + "\n".join(lines)

    async def _zunvra_notifications_tool(self, params: Dict) -> str:
        """Get notifications on Zunvra."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        result = await self.zunvra_skill.get_notifications()
        if result.get("error"):
            return f"❌ Notifications failed: {result['error']}"
        notifs = result.get("notifications") or result.get("data") or []
        if not notifs:
            return "🔔 No new notifications on Zunvra."
        lines = []
        for n in notifs[:15]:
            ntype = n.get("type", "?")
            actor = n.get("actor", {}).get("username", n.get("from_username", "?"))
            msg = n.get("message") or n.get("text") or ntype
            lines.append(f"  🔔 @{actor}: {msg[:80]}")
        return "🔔 Zunvra Notifications:\n" + "\n".join(lines)

    # ── Identity ──────────────────────────────────────────────────────────

    async def _zunvra_whoami_tool(self, params: Dict) -> str:
        """Get agent identity on Zunvra."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        result = await self.zunvra_skill.whoami()
        if result.get("error"):
            return f"❌ Whoami failed: {result['error']}"
        return json.dumps(result, indent=2, default=str)

    async def _zunvra_update_profile_tool(self, params: Dict) -> str:
        """Update the agent's Zunvra profile (display name, bio, avatar, banner)."""
        if not getattr(self, "zunvra_skill", None) or not self.zunvra_skill.is_available():
            return "❌ Zunvra skill not connected"
        result = await self.zunvra_skill.update_profile(
            display_name=params.get("display_name", ""),
            bio=params.get("bio", ""),
            avatar_url=params.get("avatar_url", ""),
            banner_url=params.get("banner_url", ""),
        )
        if result.get("error"):
            return f"❌ Profile update failed: {result['error']}"
        return "✅ Zunvra profile updated"
