"""
AgentMon League (Pokémon Red) tool implementations — mixin for ToolRegistry.

Includes embedded Pokémon Red strategy knowledge so the agent's LLM
can make informed gameplay decisions without needing external guides.
"""

import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  Pokémon Red Strategy Knowledge Base
# ══════════════════════════════════════════════════════════════════════════════
# This is injected into game-state responses so the LLM has tactical context.

_TYPE_CHART = {
    "fire": {"strong": ["grass", "ice", "bug"], "weak": ["water", "rock", "fire"]},
    "water": {"strong": ["fire", "rock", "ground"], "weak": ["grass", "electric", "water"]},
    "grass": {"strong": ["water", "rock", "ground"], "weak": ["fire", "ice", "poison", "flying", "bug"]},
    "electric": {"strong": ["water", "flying"], "weak": ["ground", "electric"]},
    "normal": {"strong": [], "weak": ["rock"], "immune": ["ghost"]},
    "fighting": {"strong": ["normal", "ice", "rock"], "weak": ["flying", "psychic", "poison", "bug"]},
    "poison": {"strong": ["grass", "bug"], "weak": ["ground", "psychic", "poison", "rock", "ghost"]},
    "ground": {"strong": ["fire", "electric", "poison", "rock"], "weak": ["water", "grass", "ice"], "immune_to": ["electric"]},
    "flying": {"strong": ["grass", "fighting", "bug"], "weak": ["electric", "rock", "ice"]},
    "psychic": {"strong": ["fighting", "poison"], "weak": ["bug"]},
    "bug": {"strong": ["grass", "psychic", "poison"], "weak": ["fire", "flying", "rock"]},
    "rock": {"strong": ["fire", "ice", "flying", "bug"], "weak": ["water", "grass", "fighting", "ground"]},
    "ghost": {"strong": ["ghost", "psychic"], "weak": [], "immune_to": ["normal", "fighting"]},
    "ice": {"strong": ["grass", "ground", "flying", "dragon"], "weak": ["fire", "fighting", "rock"]},
    "dragon": {"strong": ["dragon"], "weak": ["ice", "dragon"]},
}

_GYM_ORDER = [
    {"city": "Pewter", "leader": "Brock", "type": "rock", "badge": "Boulder", "tip": "Use Water/Grass. Bulbasaur or Squirtle sweep easily. If Charmander, grind to 13 for Metal Claw."},
    {"city": "Cerulean", "leader": "Misty", "type": "water", "badge": "Cascade", "tip": "Use Grass/Electric. Oddish or Pikachu from Viridian Forest. Starmie hits hard—be ready."},
    {"city": "Vermilion", "leader": "Lt. Surge", "type": "electric", "badge": "Thunder", "tip": "Use Ground types. Diglett from Diglett Cave is perfect. Need CUT to access gym."},
    {"city": "Celadon", "leader": "Erika", "type": "grass", "badge": "Rainbow", "tip": "Use Fire/Ice/Flying. Charizard, any Fire type, or Pidgeotto work well."},
    {"city": "Fuchsia", "leader": "Koga", "type": "poison", "badge": "Soul", "tip": "Use Ground/Psychic. Dugtrio or Kadabra demolish. Bring Antidotes for Toxic."},
    {"city": "Saffron", "leader": "Sabrina", "type": "psychic", "badge": "Marsh", "tip": "Use Bug types (Psychic is broken in Gen 1—Ghost doesn't work!). Jolteon with Pin Missile, or just overlevel."},
    {"city": "Cinnabar", "leader": "Blaine", "type": "fire", "badge": "Volcano", "tip": "Use Water/Ground/Rock. Surf makes this trivial."},
    {"city": "Viridian", "leader": "Giovanni", "type": "ground", "badge": "Earth", "tip": "Use Water/Grass/Ice. Surf + Ice Beam covers everything."},
]

_ROUTE_GUIDE = """PROGRESSION:
1. Pallet Town → Route 1 → Viridian City → Route 2 → Viridian Forest → Pewter City (Gym 1)
2. Route 3 → Mt. Moon → Route 4 → Cerulean City (Gym 2)
3. Route 24-25 (Bill) → Route 5-6 → Vermilion City (Gym 3, need CUT from SS Anne)
4. Route 9-10 → Rock Tunnel → Lavender Town → Route 8 → Celadon City (Gym 4)
5. Rocket Game Corner → Get Silph Scope → Pokémon Tower (Lavender) → Route 12-15 → Fuchsia City (Gym 5)
6. Saffron City (Silph Co. → Gym 6)
7. Route 19-21 (Surf) → Cinnabar Island (Pokémon Mansion → Gym 7)
8. Viridian City (Gym 8)
9. Route 22-23 → Victory Road → Indigo Plateau → Elite Four

KEY ITEMS: Buy Poké Balls early. Get the Old Rod (Vermilion). Teach HMs to HM slaves.
GRINDING: Grind your party 2-4 levels above the next gym leader.
CATCH: Try to catch every new species you encounter for Pokédex completion.
HEAL: Visit Pokémon Center before every gym battle.
SAVE: Save before gym leaders, legendary encounters, and the Elite Four."""


def _get_tactical_hint(state: Dict) -> str:
    """Generate a situational hint based on current game state."""
    hints: List[str] = []
    badges = state.get("badges", 0)
    map_name = state.get("mapName", "").lower()
    in_battle = state.get("inBattle", False)
    party_size = state.get("partySize", 0)
    
    if in_battle:
        battle_kind = state.get("battleKind", "")
        if "wild" in str(battle_kind).lower():
            hints.append("💡 Wild encounter: weaken it to catch, or flee if not useful. Throw Poké Ball at low HP.")
        elif "trainer" in str(battle_kind).lower() or "gym" in str(battle_kind).lower():
            hints.append("💡 Trainer battle: can't flee. Use type advantages. Switch if current Pokémon is weak to the opponent.")
    else:
        # Progression hints
        if badges < len(_GYM_ORDER):
            gym = _GYM_ORDER[badges]
            hints.append(f"🎯 Next goal: {gym['city']} Gym ({gym['leader']}, {gym['type']} type). {gym['tip']}")
        elif badges == 8:
            hints.append("🎯 All 8 badges! Head to Victory Road → Indigo Plateau → Elite Four!")
        
        # Party advice
        if party_size < 3 and badges < 3:
            hints.append("💡 Catch more Pokémon! You want a diverse team of 4-6 for type coverage.")
        
        # Healing reminder
        levels = state.get("levels", [])
        if levels and isinstance(levels, list):
            hp_data = state.get("partyHP", [])
            if hp_data and any(h.get("current", 100) < h.get("max", 100) * 0.3 for h in hp_data if isinstance(h, dict)):
                hints.append("⚠️ Low HP! Visit a Pokémon Center to heal before continuing.")
    
    return "\n".join(hints) if hints else ""


class AgentMonToolsMixin:
    """Tool handlers for the AgentMon League Pokémon Red skill."""

    # ── Session ───────────────────────────────────────────────────────────

    async def _agentmon_start_tool(self, params: Dict) -> str:
        """Start a Pokémon Red game session."""
        if not getattr(self, "agentmon_skill", None) or not self.agentmon_skill.is_available():
            return "❌ AgentMon skill not connected. Set AGENTMON_ENABLED=true in profile.env"
        starter = params.get("starter")
        load_save = params.get("load_session_id")
        speed = params.get("speed")
        result = await self.agentmon_skill.start_game(
            starter=starter, load_session_id=load_save, speed=speed,
        )
        if result.get("error"):
            return f"❌ Start failed: {result['error']}"
        state = result.get("state", {})
        return (
            f"🎮 Pokémon Red started!\n"
            f"📍 {state.get('mapName', '?')} ({state.get('x', '?')},{state.get('y', '?')})\n"
            f"🏅 Badges: {state.get('badges', 0)} | Party: {state.get('partySize', 0)}\n"
            f"📖 Pokédex: {state.get('pokedexOwned', 0)} owned, {state.get('pokedexSeen', 0)} seen"
        )

    async def _agentmon_stop_tool(self, params: Dict) -> str:
        """Stop the current game session."""
        if not getattr(self, "agentmon_skill", None) or not self.agentmon_skill.is_available():
            return "❌ AgentMon skill not connected"
        result = await self.agentmon_skill.stop_game()
        if result.get("error"):
            return f"❌ Stop failed: {result['error']}"
        return "🛑 Game session stopped"

    # ── Actions ───────────────────────────────────────────────────────────

    async def _agentmon_step_tool(self, params: Dict) -> str:
        """Send a single button press to the game."""
        if not getattr(self, "agentmon_skill", None) or not self.agentmon_skill.is_available():
            return "❌ AgentMon skill not connected"
        action = params.get("action", "")
        if not action:
            return "⚠️ action is required (up/down/left/right/a/b/start/select/pass)"
        result = await self.agentmon_skill.step(action)
        if result.get("error"):
            return f"❌ Step failed: {result['error']}"
        state = result.get("state", {})
        feedback = result.get("feedback", {})
        effects = feedback.get("effects", [])
        msg = feedback.get("message", "")
        screen_text = result.get("screenText", "")
        lines = [
            f"🎮 [{action.upper()}] {msg}",
            f"📍 {state.get('mapName', '?')} ({state.get('x', '?')},{state.get('y', '?')})",
        ]
        if effects:
            lines.append(f"⚡ Effects: {', '.join(effects)}")
        if state.get("inBattle"):
            lines.append(f"⚔️ In battle! ({state.get('battleKind', 'unknown')})")
        if screen_text:
            lines.append(f"💬 \"{screen_text}\"")
        lines.append(
            f"🏅 Badges: {state.get('badges', 0)} | Party: {state.get('partySize', 0)} | "
            f"Pokédex: {state.get('pokedexOwned', 0)}"
        )
        hint = _get_tactical_hint(state)
        if hint:
            lines.append(hint)
        return "\n".join(lines)

    async def _agentmon_actions_tool(self, params: Dict) -> str:
        """Send a sequence of button presses to the game."""
        if not getattr(self, "agentmon_skill", None) or not self.agentmon_skill.is_available():
            return "❌ AgentMon skill not connected"
        action_list = params.get("actions", [])
        if not action_list:
            return "⚠️ actions list is required (e.g. [\"up\", \"up\", \"a\"])"
        speed = params.get("speed")
        result = await self.agentmon_skill.actions(action_list, speed=speed)
        if result.get("error"):
            return f"❌ Actions failed: {result['error']}"
        state = result.get("state", {})
        feedback = result.get("feedback", {})
        effects = feedback.get("effects", [])
        screen_text = result.get("screenText", "")
        lines = [
            f"🎮 Sent {len(action_list)} actions: {', '.join(a.upper() for a in action_list[:10])}{'…' if len(action_list) > 10 else ''}",
            f"📍 {state.get('mapName', '?')} ({state.get('x', '?')},{state.get('y', '?')})",
        ]
        if effects:
            lines.append(f"⚡ Effects: {', '.join(effects)}")
        if state.get("inBattle"):
            lines.append(f"⚔️ In battle! ({state.get('battleKind', 'unknown')})")
        if screen_text:
            lines.append(f"💬 \"{screen_text}\"")
        lines.append(
            f"🏅 Badges: {state.get('badges', 0)} | Party: {state.get('partySize', 0)} | "
            f"Pokédex: {state.get('pokedexOwned', 0)}"
        )
        hint = _get_tactical_hint(state)
        if hint:
            lines.append(hint)
        return "\n".join(lines)

    # ── State ─────────────────────────────────────────────────────────────

    async def _agentmon_state_tool(self, params: Dict) -> str:
        """Get current game state with tactical analysis."""
        if not getattr(self, "agentmon_skill", None) or not self.agentmon_skill.is_available():
            return "❌ AgentMon skill not connected"
        result = await self.agentmon_skill.get_state()
        if result.get("error"):
            return f"❌ State failed: {result['error']}"
        state = result if "mapName" in result else result.get("state", result)
        lines = [
            f"📍 Location: {state.get('mapName', '?')} ({state.get('x', '?')},{state.get('y', '?')})",
            f"🏅 Badges: {state.get('badges', 0)}/8",
            f"👾 Party: {state.get('partySize', 0)} Pokémon (levels: {state.get('levels', '?')})",
            f"📖 Pokédex: {state.get('pokedexOwned', 0)} owned, {state.get('pokedexSeen', 0)} seen",
        ]
        if state.get("inBattle"):
            lines.append(f"⚔️ IN BATTLE — {state.get('battleKind', 'unknown')}")
        inv = state.get("inventory", {})
        if inv.get("count"):
            items = inv.get("items", [])
            item_str = ", ".join(f"{i.get('id', '?')}×{i.get('quantity', '?')}" for i in items[:8])
            lines.append(f"🎒 Inventory ({inv['count']}): {item_str}")
        local_map = state.get("localMap", {})
        if local_map:
            lines.append(f"🗺️ Tile: {local_map.get('tileUnder', '?')} | Front: {local_map.get('tileFront', '?')}")
        lines.append(f"⏱️ Session: {state.get('sessionTimeSeconds', 0)}s")
        # Tactical advice based on current situation
        hint = _get_tactical_hint(state)
        if hint:
            lines.append("")
            lines.append(hint)
        return "\n".join(lines)

    async def _agentmon_frame_tool(self, params: Dict) -> str:
        """Get the current game screen as a PNG screenshot."""
        if not getattr(self, "agentmon_skill", None) or not self.agentmon_skill.is_available():
            return "❌ AgentMon skill not connected"
        frame_bytes = await self.agentmon_skill.get_frame()
        if not frame_bytes:
            return "❌ Could not get game frame — is a session running?"
        # Save to temp file for vision processing
        import tempfile, base64
        tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="agentmon_frame_", delete=False)
        tmp.write(frame_bytes)
        tmp.close()
        b64 = base64.b64encode(frame_bytes).decode()
        return (
            f"🖼️ Game screen captured ({len(frame_bytes)} bytes)\n"
            f"📁 Saved to: {tmp.name}\n"
            f"[base64 PNG available for vision analysis]"
        )

    # ── Saves ─────────────────────────────────────────────────────────────

    async def _agentmon_save_tool(self, params: Dict) -> str:
        """Save the current game."""
        if not getattr(self, "agentmon_skill", None) or not self.agentmon_skill.is_available():
            return "❌ AgentMon skill not connected"
        label = params.get("label")
        result = await self.agentmon_skill.save_game(label=label)
        if result.get("error"):
            return f"❌ Save failed: {result['error']}"
        save_id = result.get("id", result.get("saveId", "?"))
        return f"💾 Game saved! ID: {save_id}" + (f" — \"{label}\"" if label else "")

    async def _agentmon_saves_tool(self, params: Dict) -> str:
        """List all game saves."""
        if not getattr(self, "agentmon_skill", None) or not self.agentmon_skill.is_available():
            return "❌ AgentMon skill not connected"
        result = await self.agentmon_skill.list_saves()
        if result.get("error"):
            return f"❌ List saves failed: {result['error']}"
        saves = result.get("saves", [])
        if not saves:
            return "📁 No saves found"
        lines = [f"💾 {len(saves)} save(s):"]
        for s in saves:
            label = s.get("label", "")
            lines.append(f"  • {s.get('id', '?')[:12]}… {f'— {label} ' if label else ''}({s.get('createdAt', '?')})")
        return "\n".join(lines)

    async def _agentmon_delete_save_tool(self, params: Dict) -> str:
        """Delete a game save."""
        if not getattr(self, "agentmon_skill", None) or not self.agentmon_skill.is_available():
            return "❌ AgentMon skill not connected"
        save_id = params.get("save_id", "")
        if not save_id:
            return "⚠️ save_id is required"
        result = await self.agentmon_skill.delete_save(save_id)
        if result.get("error"):
            return f"❌ Delete failed: {result['error']}"
        return f"🗑️ Save {save_id[:12]}… deleted"

    # ── Leaderboard ───────────────────────────────────────────────────────

    async def _agentmon_leaderboard_tool(self, params: Dict) -> str:
        """Get the AgentMon League leaderboard."""
        if not getattr(self, "agentmon_skill", None) or not self.agentmon_skill.is_available():
            return "❌ AgentMon skill not connected"
        result = await self.agentmon_skill.get_leaderboard()
        if result.get("error"):
            return f"❌ Leaderboard failed: {result['error']}"
        entries = result if isinstance(result, list) else result.get("leaderboard", result.get("entries", []))
        if not entries:
            return "🏆 Leaderboard is empty"
        lines = ["🏆 AgentMon League Leaderboard:"]
        for i, e in enumerate(entries[:15], 1):
            name = e.get("displayName", e.get("agentId", "?")[:8])
            badges = e.get("badges", 0)
            pokedex = e.get("pokedexOwned", 0)
            lines.append(f"  {i}. {name} — 🏅{badges} badges, 📖{pokedex} pokédex")
        return "\n".join(lines)
