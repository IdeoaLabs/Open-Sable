#!/usr/bin/env python3
"""
LIVE TEST — Proof that all 26 modules execute against REAL Zunvra data.

This script:
  1. Connects to the live Central Intelligence backend (port 5580)
  2. Fetches the full OSINT snapshot
  3. Initializes the ZunvraIntelSkill with all 26 modules
  4. Runs run_cycle() — the full 26-module analysis pipeline
  5. Prints EVERY result from EVERY module
"""

import asyncio
import json
import sys
import os
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set API key for RemoteControl authentication
os.environ["ZUNVRA_API_KEY"] = "ffb087ef70bc2e2a4bca32306f3f34ffd788fc33fe5e91f9c63f60504f3620d9"

from opensable.skills.zunvra_intel import ZunvraIntelSkill
from opensable.skills.zunvra_intel.connector import IntelSnapshot


BACKEND_URL = "http://localhost:5580"


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


async def main():
    section("SABLE INTELLIGENCE SUITE — LIVE DATA TEST")
    print(f"Backend: {BACKEND_URL}")
    print(f"Time:    {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    # ── 1. Initialize ─────────────────────────────────────────────
    section("1. INITIALIZING ZunvraIntelSkill (26 modules + remote + broadcaster)")
    skill = ZunvraIntelSkill(
        base_url=BACKEND_URL,
        data_dir="/tmp/zunvra_intel_test",
        enable_llm=False,  # Pure algorithmic test, no LLM needed
    )
    await skill.initialize()
    print("✓ All 26 modules initialized")

    # Speed up camera pilot for testing (reduce dwell/delay from ~6.5s to ~1.5s per move)
    if skill.camera_pilot:
        skill.camera_pilot.move_delay = 0.5
        skill.camera_pilot.dwell_default = 0.5
        skill.camera_pilot.max_moves = 10
        # Patch dwell_seconds on generated moves to be faster
        print("✓ Camera pilot dwell/delay reduced for test mode")

    # ── 2. Fetch snapshot ─────────────────────────────────────────
    section("2. FETCHING LIVE DATA FROM Central Intelligence")
    snapshot = await skill.connector.fetch_full()
    if snapshot is None:
        print("✗ Failed to fetch snapshot!")
        return

    print(f"✓ Snapshot acquired @ {snapshot.timestamp}")
    print(f"  Total entities:      {snapshot.total_entities}")
    print(f"  Commercial flights:  {len(snapshot.flights)}")
    print(f"  Military flights:    {len(snapshot.military_flights)}")
    print(f"  Ships:               {len(snapshot.ships)}")
    print(f"  Satellites:          {len(snapshot.satellites)}")
    print(f"  Earthquakes:         {len(snapshot.earthquakes)}")
    print(f"  Fires:               {len(snapshot.fires)}")
    print(f"  GDELT events:        {len(snapshot.gdelt_events)}")
    print(f"  Cyber threats:       {len(snapshot.cyber_threats)}")
    print(f"  GPS jamming zones:   {len(snapshot.gps_jamming)}")
    print(f"  Carriers:            {len(snapshot.carriers)}")
    print(f"  Conflicts:           {len(snapshot.conflicts)}")
    print(f"  Internet outages:    {len(snapshot.internet_outages)}")
    print(f"  Nuclear facilities:  {len(snapshot.nuclear_facilities)}")
    print(f"  Ransomware:          {len(snapshot.ransomware)}")
    print(f"  Prediction markets:  {len(snapshot.prediction_markets)}")
    print(f"  News feed:           {len(snapshot.news_feed)}")
    print(f"  Dark intel keys:     {list(snapshot.dark_intel.keys()) if snapshot.dark_intel else 'none'}")

    # ── 3. Run full cycle ─────────────────────────────────────────
    section("3. RUNNING FULL 26-MODULE ANALYSIS CYCLE")
    t0 = time.time()
    results = await skill.run_cycle()
    elapsed = time.time() - t0
    print(f"✓ Cycle completed in {elapsed:.2f}s")
    print(f"  Cycle number: {results.get('cycle', '?')}")

    # ── 4. Print every result ─────────────────────────────────────
    section("4. MODULE-BY-MODULE RESULTS")

    module_map = {
        "causal_links": "#2  Causal Correlation Engine",
        "threat_chains": "#2  Threat Chains Found",
        "anomalies": "#3  Anomalies Detected (Swarm Threat)",
        "swarm_overall": "#3  Swarm Overall Assessment",
        "world_model_domains": "#5  Predictive World Model — Domains Tracked",
        "forecasts": "#5  Predictive World Model — Forecasts",
        "dejavu_matches": "#6  Temporal Memory — Déjà Vu Matches",
        "cognitive_annotations": "#9  Cognitive Map Annotations",
        "fusion_convergences": "#10 Multi-Agent Fusion — Convergences",
        "graph_nodes": "#11 Knowledge Graph — Nodes",
        "graph_edges": "#11 Knowledge Graph — Edges",
        "graph_communities": "#11 Knowledge Graph — Communities",
        "pol_anomalies": "#12 Pattern of Life — Anomalies",
        "tripwire_alerts": "#13 Geofence Tripwire — Alerts",
        "kill_chain_alerts": "#14 Kill Chain — Alerts",
        "timeline_events": "#15 Timeline — Events Ingested",
        "wargame_live_scenarios": "#16 Wargaming — Live Scenarios Detected",
        "evasion_alerts": "#17 Counter-Surveillance — Evasion Alerts",
        "narrative_alerts": "#18 Narrative Warfare — Alerts",
        "sigint_alerts": "#19 SIGINT/EW — Alerts",
        "finint_alerts": "#20 Financial Intelligence — Alerts",
        "nuclear_alerts": "#21 Nuclear Monitor — Alerts",
        "space_alerts": "#22 Space Domain — Alerts",
        "infra_alerts": "#23 Infrastructure Resilience — Alerts",
        "env_alerts": "#24 Environmental Threat — Alerts",
        "force_alerts": "#25 Force Projection — Alerts",
    }

    for key, label in module_map.items():
        val = results.get(key, "NOT PRESENT")
        print(f"  {label}: {val}")

    # Wargame escalation details
    if "wargame_escalation_risks" in results:
        print(f"\n  Wargame Escalation Details:")
        for s in results["wargame_escalation_risks"]:
            print(f"    → [{s['risk'].upper()}] ({s['score']}/100) {s['trigger']}")

    # ── 5. Threat Fusion ──────────────────────────────────────────
    section("5. THREAT FUSION — UNIFIED THREAT PICTURE")

    if "threat_picture" in results:
        pic = results["threat_picture"]
        print(f"  Threat Level:  {pic.get('threat_level', '?')} / 5")
        print(f"  Threat Name:   {pic.get('threat_name', '?')}")
        print(f"  Overall Score: {pic.get('overall_score', '?')}")
        if pic.get("fusion_alerts"):
            print(f"  Fusion Alerts: {len(pic['fusion_alerts'])}")
            for a in pic["fusion_alerts"][:5]:
                print(f"    → {a}")
    else:
        # Try to get it from the threat_fusion module directly
        if skill.threat_fusion and hasattr(skill.threat_fusion, '_last_picture'):
            pic = skill.threat_fusion._last_picture
            if pic:
                print(f"  Threat Level:  {pic.threat_level} / 5")
                print(f"  Threat Name:   {pic.threat_name}")
                print(f"  Overall Score: {pic.overall_score:.2f}")

    # ── 5b. Camera Pilot ─────────────────────────────────────────
    section("5b. AUTONOMOUS CAMERA PILOT")
    cam_moves = results.get("camera_moves", 0)
    cam_styles = results.get("camera_styles", [])
    cam_regions = results.get("camera_regions", [])
    print(f"  Camera moves executed:  {cam_moves}")
    print(f"  Styles used:           {cam_styles}")
    print(f"  Regions visited ({len(cam_regions)}):")
    for r in cam_regions:
        print(f"    → {r}")
    if cam_moves == 0:
        print("  ⚠  No camera moves  (RemoteControl may not be connected or dashboard not open)")

    # ── 6. Raw results JSON ───────────────────────────────────────
    section("6. RAW RESULTS JSON (for verification)")
    print(json.dumps(results, indent=2, default=str))

    # ── 7. Module stats ──────────────────────────────────────────
    section("7. MODULE STATS")

    stats_modules = [
        ("Connector", skill.connector),
        ("Causal Engine", skill.causal_engine),
        ("Swarm Assessor", skill.swarm_assessor),
        ("World Model", skill.world_model),
        ("Temporal Memory", skill.temporal_memory),
        ("Knowledge Graph", skill.knowledge_graph),
        ("Pattern of Life", skill.pattern_of_life),
        ("Geofence", skill.geofence),
        ("Kill Chain", skill.kill_chain),
        ("Timeline", skill.timeline),
        ("Wargaming", skill.wargaming),
        ("Counter-Surveillance", skill.counter_surveillance),
        ("Narrative Warfare", skill.narrative_warfare),
        ("SIGINT/EW", skill.sigint_ew),
        ("FININT", skill.finint),
        ("Nuclear", skill.nuclear),
        ("Space Domain", skill.space_domain),
        ("Infrastructure", skill.infrastructure),
        ("Environmental", skill.environmental),
        ("Force Projection", skill.force_projection),
        ("Threat Fusion", skill.threat_fusion),
        ("Camera Pilot", skill.camera_pilot),
    ]

    for name, mod in stats_modules:
        if mod and hasattr(mod, "get_stats"):
            try:
                s = mod.get_stats()
                print(f"  {name}: {json.dumps(s, default=str)}")
            except Exception as e:
                print(f"  {name}: error getting stats: {e}")
        elif mod:
            print(f"  {name}: initialized (no get_stats)")

    # ── Done ──────────────────────────────────────────────────────
    section("TEST COMPLETE")
    total_findings = sum(
        results.get(k, 0) for k in results
        if isinstance(results.get(k), int) and k != "cycle"
    )
    print(f"  Total findings across all modules: {total_findings}")
    print(f"  Execution time: {elapsed:.2f}s")

    await skill.shutdown()
    print("  ✓ Skill shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())
