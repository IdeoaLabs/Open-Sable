"""
Zunvra Intelligence — Live Camera Sources

Camera system disabled.  Windy embeds removed (only showed weather maps,
not real camera feeds).  The infrastructure remains for when a real
embeddable camera source is found.

All functions return empty results so the rest of the system keeps working.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Live camera entry ────────────────────────────────────────────────

@dataclass
class LiveCamera:
    lat: float
    lng: float
    name: str
    url: str              # Embeddable URL
    platform: str         # placeholder
    tags: List[str] = field(default_factory=list)
    country: str = ""
    city: str = ""
    thumbnail: str = ""   # Optional thumbnail URL


# ═══════════════════════════════════════════════════════════════════════
# CAMERA SYSTEM DISABLED — no embeddable source available
# ═══════════════════════════════════════════════════════════════════════

def generate_camera_for_location(
    lat: float,
    lng: float,
    label: str = "",
    city: str = "",
    country: str = "",
) -> Optional[LiveCamera]:
    """Disabled — returns None. No embeddable camera source available."""
    return None


LIVE_CAMERAS: List[LiveCamera] = []


# ═══════════════════════════════════════════════════════════════════════
# SEARCH / QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in km between two points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_cameras_near(lat: float, lng: float, radius_km: float = 500.0,
                      max_results: int = 5) -> List[LiveCamera]:
    """Find live cameras within radius_km of a point, sorted by distance."""
    results: List[Tuple[float, LiveCamera]] = []
    for cam in LIVE_CAMERAS:
        d = _haversine_km(lat, lng, cam.lat, cam.lng)
        if d <= radius_km:
            results.append((d, cam))
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results[:max_results]]


def find_cameras_by_tags(tags: List[str], max_results: int = 5) -> List[LiveCamera]:
    """Find cameras whose tags match any of the given keywords."""
    scores: List[Tuple[int, LiveCamera]] = []
    tags_lower = [t.lower() for t in tags]
    for cam in LIVE_CAMERAS:
        cam_tags = [t.lower() for t in cam.tags] + [cam.country.lower(), cam.city.lower(), cam.name.lower()]
        score = sum(1 for t in tags_lower if any(t in ct for ct in cam_tags))
        if score > 0:
            scores.append((score, cam))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scores[:max_results]]


def get_random_camera() -> Optional[LiveCamera]:
    """Get a random live camera from the database."""
    if not LIVE_CAMERAS:
        return None
    return random.choice(LIVE_CAMERAS)


def get_all_cameras() -> List[LiveCamera]:
    """Return all live cameras."""
    return list(LIVE_CAMERAS)


def camera_to_dict(cam: LiveCamera) -> Dict[str, Any]:
    """Serialize a LiveCamera to a dict for JSON transport."""
    return {
        "lat": cam.lat,
        "lng": cam.lng,
        "name": cam.name,
        "url": cam.url,
        "platform": cam.platform,
        "tags": cam.tags,
        "country": cam.country,
        "city": cam.city,
        "thumbnail": cam.thumbnail,
    }
