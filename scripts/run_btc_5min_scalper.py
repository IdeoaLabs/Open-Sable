#!/usr/bin/env python3
"""
BTC 5-Min Polymarket Scalper — Standalone Runner

This script runs the scalping strategy 24/7.  It handles:
    1.  Discovering the current 5-min BTC up/down round on Polymarket
    2.  Fetching BTC spot prices from Binance (public, no key needed)
    3.  Fetching Polymarket YES/NO order books
    4.  Feeding context to the PolymarketBtc5MinStrategy
    5.  Executing signals through the Polymarket connector
    6.  Looping across rounds forever

Usage:
    # Paper mode (default — no real money):
    python scripts/run_btc_5min_scalper.py

    # Live mode:
    python scripts/run_btc_5min_scalper.py --live

    # With the full agent framework:
    python main.py --profile btc-5min-scalper
"""

from __future__ import annotations

import argparse
import asyncio
import json as _json
import logging
import os
import sys
import time
from collections import deque
from decimal import Decimal
from typing import Any, Dict, List, Optional

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("btc_5min_scalper")

# ---------------------------------------------------------------------------
# BTC price feed — Binance WebSocket (real-time, no API key required)
# ---------------------------------------------------------------------------

class BinanceBtcStream:
    """
    Maintains a persistent WebSocket connection to Binance's
    aggTrade stream for BTCUSDT.  Exposes the latest price and
    a rolling history via simple attributes.

    Usage:
        stream = BinanceBtcStream()
        await stream.start()       # launches background task
        price = stream.latest      # always the freshest price (float | None)
        hist  = stream.history     # deque of recent prices
        await stream.stop()
    """

    WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"

    def __init__(self, maxlen: int = 500):
        self.latest: Optional[float] = None
        self.history: deque = deque(maxlen=maxlen)
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._listen())
        logger.info("Binance BTC WebSocket stream started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Binance BTC WebSocket stream stopped")

    async def _listen(self) -> None:
        import websockets

        while self._running:
            try:
                async with websockets.connect(
                    self.WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
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
                logger.warning(f"Binance WS error: {e} — reconnecting in 2s")
                await asyncio.sleep(2)

# ---------------------------------------------------------------------------
# Polymarket round discovery
# ---------------------------------------------------------------------------

async def discover_current_round(pm_client) -> Optional[Dict[str, Any]]:
    """
    Find the active 5-min BTC up/down market on Polymarket.

    Returns dict with keys:
        condition_id, yes_token_id, no_token_id,
        question, end_date_ts
    or None if no active round is found.
    """
    if pm_client is None:
        return None
    try:
        markets = pm_client.get_markets()
        for m in markets:
            q = (m.get("question") or "").lower()
            # Match common naming patterns for 5-min BTC markets
            if ("bitcoin" in q or "btc" in q) and ("5" in q or "five" in q) and ("up" in q or "down" in q or "higher" in q or "lower" in q):
                tokens = m.get("tokens", [])
                if len(tokens) < 2:
                    continue
                yes_tok = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
                no_tok = next((t for t in tokens if t.get("outcome", "").upper() == "NO"), None)
                if not yes_tok or not no_tok:
                    yes_tok, no_tok = tokens[0], tokens[1]
                return {
                    "condition_id": m.get("condition_id"),
                    "yes_token_id": yes_tok.get("token_id"),
                    "no_token_id": no_tok.get("token_id"),
                    "question": m.get("question"),
                    "end_date_ts": time.time() + 300,  # approximate
                }
        return None
    except Exception as e:
        logger.error(f"Round discovery failed: {e}")
        return None


async def fetch_orderbook(pm_client, token_id: str) -> Dict[str, list]:
    """Fetch bid/ask from Polymarket CLOB for a token."""
    if pm_client is None:
        return {"bids": [], "asks": []}
    try:
        book = pm_client.get_order_book(token_id)
        return {
            "bids": [[b["price"], b["size"]] for b in book.get("bids", [])[:10]],
            "asks": [[a["price"], a["size"]] for a in book.get("asks", [])[:10]],
        }
    except Exception as e:
        logger.debug(f"Orderbook fetch failed for {token_id}: {e}")
        return {"bids": [], "asks": []}

# ---------------------------------------------------------------------------
# Main scalping loop
# ---------------------------------------------------------------------------

async def run_scalper(live: bool = False, private_key: str = ""):
    # -- Strategy ----------------------------------------------------------
    from opensable.skills.trading.strategies.polymarket_btc_5min import (
        PolymarketBtc5MinStrategy,
    )
    from opensable.skills.trading.strategy_engine import SignalDirection

    strategy = PolymarketBtc5MinStrategy(config={
        "interval": 5,
        "take_profit_pct": 5,
        "stop_loss_pct": 5,
        "bet_size_usd": 5,
        "max_hold_seconds": 240,
    })

    # -- Polymarket client (if live) ----------------------------------------
    pm_client = None
    if live and private_key:
        try:
            from py_clob_client.client import ClobClient
            pm_client = ClobClient(
                "https://clob.polymarket.com",
                key=private_key,
                chain_id=137,
            )
            pm_client.set_api_creds(pm_client.create_or_derive_api_creds())
            logger.info("Polymarket client connected (LIVE mode)")
        except Exception as e:
            logger.error(f"Failed to connect Polymarket: {e}")
            return
    else:
        logger.info("Running in PAPER mode — no real orders will be placed")

    # -- Binance WebSocket for real-time BTC prices -------------------------
    btc_stream = BinanceBtcStream(maxlen=500)
    await btc_stream.start()
    # Give the WS a moment to receive the first trade
    await asyncio.sleep(2)
    if btc_stream.latest is None:
        logger.warning("No BTC price yet from WS — waiting up to 10 s…")
        for _ in range(8):
            await asyncio.sleep(1)
            if btc_stream.latest is not None:
                break
    logger.info(f"BTC stream ready — latest price: ${btc_stream.latest:,.2f}"
                if btc_stream.latest else "BTC stream: no price yet")

    # -- State -------------------------------------------------------------
    active_position: Optional[Dict[str, Any]] = None  # track open pos
    stats = {"rounds": 0, "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}

    logger.info("=" * 60)
    logger.info("  BTC 5-Min Polymarket Scalper — Starting")
    logger.info(f"  Mode: {'LIVE' if live else 'PAPER'}")
    logger.info("=" * 60)

    try:
      while True:
        try:
            # 1. Discover current round
            round_info = await discover_current_round(pm_client)
            if not round_info:
                logger.info("No active 5-min BTC round found. Waiting 30s…")
                await asyncio.sleep(30)
                continue

            round_start = time.time()
            yes_token = round_info["yes_token_id"]
            no_token = round_info["no_token_id"]
            logger.info(
                f"Round #{stats['rounds'] + 1}: {round_info['question']}"
            )

            # 2. Trade within this round
            while True:
                elapsed = time.time() - round_start
                if elapsed > 280:  # 4m40s — stop entering, let exits run
                    break

                # BTC price from WebSocket (real-time, no polling)
                btc = btc_stream.latest

                # Fetch orderbooks
                yes_book = await fetch_orderbook(pm_client, yes_token)
                no_book = await fetch_orderbook(pm_client, no_token)

                # Build context for strategy
                from opensable.skills.trading.base import PriceTick
                tick = PriceTick(
                    symbol="BTC/USDT",
                    price=Decimal(str(btc)) if btc else Decimal("0"),
                    exchange="binance",
                )
                context = {
                    "yes_token_id": yes_token,
                    "no_token_id": no_token,
                    "yes_book": yes_book,
                    "no_book": no_book,
                    "round_start_ts": round_start,
                    "btc_prices": list(btc_stream.history)[-60:],
                }

                # -- Check exit on active position --
                if active_position:
                    pos_token = active_position["token_id"]
                    pos_book = yes_book if pos_token == yes_token else no_book
                    pos_bids = pos_book.get("bids", [])
                    current_mid = (
                        Decimal(str(pos_bids[0][0])) if pos_bids
                        else active_position["entry"]
                    )
                    pnl_pct = (
                        (current_mid - active_position["entry"])
                        / active_position["entry"] * 100
                        if active_position["entry"] > 0 else Decimal("0")
                    )
                    hold_secs = int(time.time() - active_position["ts"])

                    should_exit = await strategy.should_exit(
                        pos_token,
                        active_position["entry"],
                        current_mid,
                        pnl_pct,
                        hold_secs,
                        context,
                    )
                    if should_exit:
                        pnl_float = float(pnl_pct)
                        stats["pnl"] += pnl_float
                        if pnl_float > 0:
                            stats["wins"] += 1
                        else:
                            stats["losses"] += 1
                        logger.info(
                            f"  EXIT {active_position['side']} | "
                            f"entry={active_position['entry']}, "
                            f"exit={current_mid}, pnl={pnl_float:+.2f}% | "
                            f"held {hold_secs}s"
                        )
                        # Place sell order in live mode
                        if live and pm_client:
                            try:
                                from py_clob_client.order_builder.constants import SELL
                                pm_client.create_and_post_order({
                                    "token_id": pos_token,
                                    "price": float(current_mid),
                                    "size": float(active_position["qty"]),
                                    "side": SELL,
                                })
                            except Exception as e:
                                logger.error(f"  Exit order failed: {e}")
                        active_position = None

                # -- Generate new signals (only if no active position) --
                if not active_position:
                    signals = await strategy.analyze(
                        "polymarket-btc-5min", [], tick, context
                    )
                    for sig in signals:
                        if sig.confidence < 0.4:
                            continue

                        entry_price = sig.entry_price or Decimal("0.50")
                        side_label = sig.metadata.get("side", "YES")
                        bet_usd = sig.metadata.get("bet_size_usd", 5)
                        qty = Decimal(str(round(bet_usd / float(entry_price), 1))) if entry_price > 0 else Decimal("10")

                        logger.info(
                            f"  SIGNAL: BUY {side_label} @ {entry_price} | "
                            f"conf={sig.confidence:.0%} | {sig.reason}"
                        )

                        # Place order in live mode
                        if live and pm_client:
                            try:
                                from py_clob_client.order_builder.constants import BUY
                                pm_client.create_and_post_order({
                                    "token_id": sig.symbol,
                                    "price": float(entry_price),
                                    "size": float(qty),
                                    "side": BUY,
                                })
                            except Exception as e:
                                logger.error(f"  Order failed: {e}")
                                continue

                        active_position = {
                            "token_id": sig.symbol,
                            "side": side_label,
                            "entry": entry_price,
                            "qty": qty,
                            "ts": time.time(),
                        }
                        stats["trades"] += 1
                        break  # one position per round

                await asyncio.sleep(5)

            # -- Force exit if still holding at round end --
            if active_position:
                logger.info("  Round ending — force exit")
                if live and pm_client:
                    try:
                        from py_clob_client.order_builder.constants import SELL
                        pm_client.create_and_post_order({
                            "token_id": active_position["token_id"],
                            "price": float(active_position["entry"]),
                            "size": float(active_position["qty"]),
                            "side": SELL,
                        })
                    except Exception:
                        pass
                active_position = None

            stats["rounds"] += 1
            win_rate = (
                stats["wins"] / max(stats["wins"] + stats["losses"], 1) * 100
            )
            logger.info(
                f"Round done | trades={stats['trades']}, "
                f"W/L={stats['wins']}/{stats['losses']} "
                f"({win_rate:.0f}%), cum_pnl={stats['pnl']:+.2f}%"
            )

            # Wait for the next round to start
            await asyncio.sleep(20)

        except KeyboardInterrupt:
            logger.info("Shutting down…")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}", exc_info=True)
            await asyncio.sleep(10)
    finally:
        await btc_stream.stop()


def main():
    parser = argparse.ArgumentParser(
        description="BTC 5-Min Polymarket Scalper"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable LIVE trading (requires POLYMARKET_PRIVATE_KEY env var)",
    )
    args = parser.parse_args()

    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    if args.live and not private_key:
        print("ERROR: --live requires POLYMARKET_PRIVATE_KEY env var")
        sys.exit(1)

    asyncio.run(run_scalper(live=args.live, private_key=private_key))


if __name__ == "__main__":
    main()
