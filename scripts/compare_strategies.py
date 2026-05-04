#!/usr/bin/env python3
"""
BTC 5-Min Strategy Comparator,  Live Paper-Trading Tournament

Runs all 3 strategies side-by-side against REAL Polymarket + Binance
data.  No real money,  pure paper-trading comparison.

Strategies tested:
    A) Momentum        ,  chases BTC momentum + orderbook imbalance
    B) Vol-Directional ,  statistical probability from realized vol
    C) Mean-Reversion  ,  fades overreactions in share prices

Data sources:
    - Binance WebSocket (btcusdt@aggTrade),  real-time BTC prices
    - Polymarket CLOB API,  real YES/NO orderbooks & share prices

Output:
    - Live dashboard printed every round
    - JSON log at data/strategy_comparison.json (append)
    - Summary table after each round

Usage:
    # Needs: pip install websockets httpx py-clob-client
    # Polymarket key optional,  only needed for orderbooks.
    # Without key, uses public market data.

    python scripts/compare_strategies.py

    # With Polymarket key (better orderbook data):
    POLYMARKET_PRIVATE_KEY=0x... python scripts/compare_strategies.py

    # Stop with Ctrl-C,  prints final summary.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("strategy_compare")

# ═══════════════════════════════════════════════════════════════════════════
# Binance BTC WebSocket (reused from run_btc_5min_scalper.py)
# ═══════════════════════════════════════════════════════════════════════════

class BinanceBtcStream:
    WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"

    def __init__(self, maxlen: int = 2000):
        self.latest: Optional[float] = None
        self.history: deque = deque(maxlen=maxlen)
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _listen(self) -> None:
        import websockets
        while self._running:
            try:
                async with websockets.connect(
                    self.WS_URL, ping_interval=20, ping_timeout=10, close_timeout=5,
                ) as ws:
                    logger.info("Connected to Binance WS (btcusdt@aggTrade)")
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = _json.loads(raw)
                            price = float(msg["p"])
                            self.latest = price
                            self.history.append(price)
                        except (KeyError, ValueError):
                            continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Binance WS: {e},  reconnecting 2s")
                await asyncio.sleep(2)


# ═══════════════════════════════════════════════════════════════════════════
# Polymarket data fetcher (public REST,  no key needed for prices)
# ═══════════════════════════════════════════════════════════════════════════

class PolymarketFeed:
    """
    Fetches 5-min BTC market data from Polymarket.
    Works with or without a private key.
    With key: uses py-clob-client for deeper orderbooks.
    Without key: uses public REST endpoints.
    """

    def __init__(self, private_key: str = ""):
        self._client = None
        self._private_key = private_key
        self._http = None

    async def connect(self) -> None:
        if self._private_key:
            try:
                from py_clob_client.client import ClobClient
                self._client = ClobClient(
                    "https://clob.polymarket.com",
                    key=self._private_key,
                    chain_id=137,
                )
                self._client.set_api_creds(self._client.create_or_derive_api_creds())
                logger.info("Polymarket CLOB client connected")
                return
            except Exception as e:
                logger.warning(f"CLOB client failed: {e}, falling back to REST")

        import httpx
        self._http = httpx.AsyncClient(
            base_url="https://clob.polymarket.com",
            timeout=10,
        )
        logger.info("Polymarket REST feed connected")

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()

    async def discover_round(self) -> Optional[Dict[str, Any]]:
        """Find the active 5-min BTC up/down market."""
        try:
            if self._client:
                markets = self._client.get_markets()
            elif self._http:
                resp = await self._http.get("/markets")
                resp.raise_for_status()
                markets = resp.json()
            else:
                return None

            for m in markets if isinstance(markets, list) else []:
                q = (m.get("question") or "").lower()
                if (("bitcoin" in q or "btc" in q)
                        and ("5" in q or "five" in q)
                        and ("up" in q or "down" in q or "higher" in q or "lower" in q)):
                    tokens = m.get("tokens", [])
                    if len(tokens) < 2:
                        continue
                    yes_tok = next((t for t in tokens if (t.get("outcome") or "").upper() == "YES"), None)
                    no_tok = next((t for t in tokens if (t.get("outcome") or "").upper() == "NO"), None)
                    if not yes_tok or not no_tok:
                        yes_tok, no_tok = tokens[0], tokens[1]
                    return {
                        "condition_id": m.get("condition_id"),
                        "yes_token_id": yes_tok.get("token_id"),
                        "no_token_id": no_tok.get("token_id"),
                        "question": m.get("question"),
                    }
            return None
        except Exception as e:
            logger.error(f"Round discovery: {e}")
            return None

    async def get_orderbook(self, token_id: str) -> Dict[str, list]:
        """Fetch orderbook for a token."""
        try:
            if self._client:
                book = self._client.get_order_book(token_id)
            elif self._http:
                resp = await self._http.get(f"/book", params={"token_id": token_id})
                resp.raise_for_status()
                book = resp.json()
            else:
                return {"bids": [], "asks": []}

            return {
                "bids": [[b["price"], b["size"]] for b in book.get("bids", [])[:10]],
                "asks": [[a["price"], a["size"]] for a in book.get("asks", [])[:10]],
            }
        except Exception as e:
            logger.debug(f"Orderbook {token_id}: {e}")
            return {"bids": [], "asks": []}


# ═══════════════════════════════════════════════════════════════════════════
# Paper position tracker (one per strategy)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PaperPosition:
    token_id: str
    side: str       # "YES" or "NO"
    entry: Decimal
    qty: Decimal
    ts: float
    strategy: str


@dataclass
class StrategyTracker:
    """Tracks paper P&L for a single strategy."""
    name: str
    capital: float = 100.0       # starting $100 each
    position: Optional[PaperPosition] = None
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    rounds_traded: int = 0
    peak_capital: float = 100.0
    max_drawdown: float = 0.0
    trade_log: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total * 100 if total else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.trades if self.trades else 0.0

    def record_exit(self, pnl_pct: float, reason: str, entry_price: float,
                    exit_price: float, side: str) -> None:
        profit_usd = self.capital * (pnl_pct / 100) * 0.05  # 5% allocation
        self.capital += profit_usd
        self.total_pnl += pnl_pct
        if pnl_pct > 0:
            self.wins += 1
        else:
            self.losses += 1
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        dd = (self.peak_capital - self.capital) / self.peak_capital * 100
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        self.trade_log.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "side": side,
            "entry": entry_price,
            "exit": exit_price,
            "pnl_pct": round(pnl_pct, 3),
            "profit_usd": round(profit_usd, 4),
            "capital": round(self.capital, 2),
            "reason": reason,
        })

    def summary_line(self) -> str:
        return (
            f"{self.name:<22} | ${self.capital:>8.2f} | "
            f"{self.trades:>4} trades | "
            f"W/L {self.wins}/{self.losses} ({self.win_rate:>5.1f}%) | "
            f"pnl {self.total_pnl:>+7.2f}% | "
            f"dd {self.max_drawdown:>5.1f}%"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main comparison loop
# ═══════════════════════════════════════════════════════════════════════════

async def run_comparison():
    from opensable.skills.trading.strategies.polymarket_btc_5min import (
        PolymarketBtc5MinStrategy,
    )
    from opensable.skills.trading.strategies.vol_directional import (
        VolatilityDirectionalStrategy,
    )
    from opensable.skills.trading.strategies.mean_reversion_5min import (
        MeanReversionStrategy,
    )
    from opensable.skills.trading.base import PriceTick

    # -- Instantiate the 3 strategies ------------------------------------
    strat_a = PolymarketBtc5MinStrategy(config={"interval": 5, "bet_size_usd": 5})
    strat_b = VolatilityDirectionalStrategy(config={"interval": 5, "bet_size_usd": 5})
    strat_c = MeanReversionStrategy(config={"interval": 5, "bet_size_usd": 5})

    strategies = [
        (strat_a, StrategyTracker(name="A) Momentum")),
        (strat_b, StrategyTracker(name="B) Vol-Directional")),
        (strat_c, StrategyTracker(name="C) Mean-Reversion")),
    ]

    # -- Data feeds -------------------------------------------------------
    btc_stream = BinanceBtcStream(maxlen=2000)
    await btc_stream.start()

    pm_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    pm_feed = PolymarketFeed(private_key=pm_key)
    await pm_feed.connect()

    # Wait for first BTC price
    logger.info("Waiting for BTC stream…")
    for _ in range(20):
        if btc_stream.latest:
            break
        await asyncio.sleep(1)
    if not btc_stream.latest:
        logger.error("No BTC price,  check network. Exiting.")
        return

    logger.info(f"BTC stream ready: ${btc_stream.latest:,.2f}")

    # -- Log file ---------------------------------------------------------
    log_path = Path("data/strategy_comparison.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    round_num = 0

    print("\n" + "=" * 78)
    print("  BTC 5-Min Strategy Comparator,  LIVE Paper Trading")
    print("  Ctrl-C to stop and print final results")
    print("=" * 78 + "\n")

    try:
        while True:
            # -- 1. Discover current round --------------------------------
            round_info = await pm_feed.discover_round()
            if not round_info:
                logger.info("No active 5-min BTC round. Retrying in 30s…")
                await asyncio.sleep(30)
                continue

            round_num += 1
            round_start = time.time()
            yes_token = round_info["yes_token_id"]
            no_token = round_info["no_token_id"]
            logger.info(f"\n{'─'*60}")
            logger.info(f"Round #{round_num}: {round_info['question']}")
            logger.info(f"BTC: ${btc_stream.latest:,.2f}")

            # -- 2. Inner loop: scan every 5s within the round ------------
            while True:
                elapsed = time.time() - round_start
                if elapsed > 280:
                    break

                btc = btc_stream.latest
                if not btc:
                    await asyncio.sleep(2)
                    continue

                # Fetch orderbooks
                yes_book = await pm_feed.get_orderbook(yes_token)
                no_book = await pm_feed.get_orderbook(no_token)

                tick = PriceTick(
                    symbol="BTC/USDT",
                    price=Decimal(str(btc)),
                    exchange="binance",
                )
                context = {
                    "yes_token_id": yes_token,
                    "no_token_id": no_token,
                    "yes_book": yes_book,
                    "no_book": no_book,
                    "round_start_ts": round_start,
                    "btc_prices": list(btc_stream.history)[-100:],
                }

                # -- Run each strategy ------------------------------------
                for strat, tracker in strategies:
                    # Check exit on active position
                    if tracker.position:
                        pos = tracker.position
                        pos_book = yes_book if pos.token_id == yes_token else no_book
                        pos_bids = pos_book.get("bids", [])
                        current_mid = (
                            Decimal(str(pos_bids[0][0])) if pos_bids
                            else pos.entry
                        )
                        pnl_pct = (
                            (current_mid - pos.entry) / pos.entry * 100
                            if pos.entry > 0 else Decimal("0")
                        )
                        hold_secs = int(time.time() - pos.ts)

                        should_exit = await strat.should_exit(
                            pos.token_id, pos.entry, current_mid,
                            pnl_pct, hold_secs, context,
                        )
                        if should_exit:
                            pnl_f = float(pnl_pct)
                            reason = "TP" if pnl_f > 0 else ("SL" if pnl_f < -3 else "time/round")
                            tracker.record_exit(
                                pnl_f, reason,
                                float(pos.entry), float(current_mid), pos.side,
                            )
                            logger.info(
                                f"  [{tracker.name}] EXIT {pos.side} | "
                                f"pnl={pnl_f:+.2f}% | held {hold_secs}s | "
                                f"capital=${tracker.capital:.2f}"
                            )
                            tracker.position = None

                    # Generate new signals
                    if not tracker.position:
                        signals = await strat.analyze(
                            "polymarket-btc-5min", [], tick, context
                        )
                        for sig in signals:
                            if sig.confidence < 0.3:
                                continue
                            entry_price = sig.entry_price or Decimal("0.50")
                            side_label = sig.metadata.get("side", "YES")
                            bet_usd = sig.metadata.get("bet_size_usd", 5)
                            qty = (
                                Decimal(str(round(bet_usd / float(entry_price), 1)))
                                if entry_price > 0 else Decimal("10")
                            )

                            tracker.position = PaperPosition(
                                token_id=sig.symbol,
                                side=side_label,
                                entry=entry_price,
                                qty=qty,
                                ts=time.time(),
                                strategy=tracker.name,
                            )
                            tracker.trades += 1
                            tracker.rounds_traded += 1
                            logger.info(
                                f"  [{tracker.name}] BUY {side_label} @ "
                                f"{entry_price} | conf={sig.confidence:.0%} | "
                                f"{sig.reason[:80]}"
                            )
                            break

                await asyncio.sleep(5)

            # -- 3. Force exit any remaining positions --------------------
            for strat, tracker in strategies:
                if tracker.position:
                    pos = tracker.position
                    pos_book = yes_book if pos.token_id == yes_token else no_book
                    pos_bids = pos_book.get("bids", [])
                    current_mid = (
                        Decimal(str(pos_bids[0][0])) if pos_bids else pos.entry
                    )
                    pnl_pct = float(
                        (current_mid - pos.entry) / pos.entry * 100
                        if pos.entry > 0 else 0
                    )
                    tracker.record_exit(pnl_pct, "round_end",
                                        float(pos.entry), float(current_mid), pos.side)
                    tracker.position = None

            # -- 4. Print round summary -----------------------------------
            print(f"\n{'═'*78}")
            print(f"  Round #{round_num} complete | BTC: ${btc_stream.latest:,.2f}")
            print(f"{'─'*78}")
            for _, tracker in strategies:
                print(f"  {tracker.summary_line()}")
            print(f"{'═'*78}\n")

            # -- 5. Append to log file ------------------------------------
            log_entry = {
                "round": round_num,
                "time": datetime.now(timezone.utc).isoformat(),
                "btc_price": btc_stream.latest,
                "strategies": {},
            }
            for _, tracker in strategies:
                log_entry["strategies"][tracker.name] = {
                    "capital": round(tracker.capital, 2),
                    "trades": tracker.trades,
                    "wins": tracker.wins,
                    "losses": tracker.losses,
                    "win_rate": round(tracker.win_rate, 1),
                    "total_pnl": round(tracker.total_pnl, 3),
                    "max_drawdown": round(tracker.max_drawdown, 2),
                    "last_trade": tracker.trade_log[-1] if tracker.trade_log else None,
                }
            with open(log_path, "a") as f:
                f.write(_json.dumps(log_entry) + "\n")

            # Wait for next round
            await asyncio.sleep(20)

    except KeyboardInterrupt:
        pass
    finally:
        await btc_stream.stop()
        await pm_feed.close()

        # ── Final summary ──
        print("\n\n" + "=" * 78)
        print("  FINAL RESULTS")
        print("=" * 78)
        print(f"  Rounds completed: {round_num}")
        print(f"  Duration: started {datetime.now(timezone.utc).isoformat()}")
        print(f"{'─'*78}")
        for _, tracker in strategies:
            print(f"  {tracker.summary_line()}")
        print(f"{'─'*78}")

        # Declare winner
        best = max(strategies, key=lambda x: x[1].capital)
        worst = min(strategies, key=lambda x: x[1].capital)
        print(f"\n  🏆 WINNER: {best[1].name} (${best[1].capital:.2f})")
        print(f"  📉 WORST:  {worst[1].name} (${worst[1].capital:.2f})")
        print(f"\n  Full log: {log_path}")
        print("=" * 78 + "\n")

        # Save final summary
        summary_path = Path("data/strategy_comparison_summary.json")
        summary = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "rounds": round_num,
            "winner": best[1].name,
            "results": {},
        }
        for _, tracker in strategies:
            summary["results"][tracker.name] = {
                "capital": round(tracker.capital, 2),
                "return_pct": round((tracker.capital - 100) / 100 * 100, 2),
                "trades": tracker.trades,
                "wins": tracker.wins,
                "losses": tracker.losses,
                "win_rate": round(tracker.win_rate, 1),
                "total_pnl_pct": round(tracker.total_pnl, 3),
                "max_drawdown_pct": round(tracker.max_drawdown, 2),
                "avg_pnl_per_trade": round(tracker.avg_pnl, 3),
                "trades_log": tracker.trade_log,
            }
        with open(summary_path, "w") as f:
            _json.dump(summary, f, indent=2)
        print(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    asyncio.run(run_comparison())
