"""
Polymarket BTC 5-Min — Mean-Reversion Counter-Trade Strategy

Fades the crowd:  when YES or NO shares spike after a BTC move,
bet that the market has overreacted and prices will revert.

Edge:  Short-term binary-market participants systematically overshoot.
       A 0.1% BTC move often pushes YES from $0.50 → $0.65 even though
       the actual probability barely changed.  Buy the cheap side.

Signal logic:
    1.  Track the YES/NO mid-price over the round.
    2.  When YES jumps > threshold above its round mean → overreaction.
        Buy NO (counter-trade).
    3.  When NO jumps > threshold above its round mean → overreaction.
        Buy YES (counter-trade).
    4.  Exit when the share price reverts toward the mean or on TP/SL.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..base import OHLCV, OrderType, PriceTick
from ..strategy_engine import Signal, SignalDirection, Strategy

logger = logging.getLogger(__name__)


def _orderbook_imbalance(bids: list, asks: list) -> float:
    bid_vol = sum(float(b[1]) for b in bids[:5]) if bids else 0
    ask_vol = sum(float(a[1]) for a in asks[:5]) if asks else 0
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


class MeanReversionStrategy(Strategy):
    """
    Mean-reversion counter-trade on Polymarket 5-min BTC up/down.

    Buys the opposite side when the market overreacts to BTC moves.

    Config:
        overshoot_threshold  – how far YES/NO must deviate from round
                               mean to trigger (default 0.08 = 8 cents)
        min_round_ticks      – minimum data points in the round before
                               trading (default 10)
        take_profit_pct      – exit % (default 5)
        stop_loss_pct        – stop % (default 6)
        max_hold_seconds     – forced exit (default 200)
        bet_size_usd         – per-trade (default 5)
    """

    name = "mean_reversion_5min"
    description = "Counter-trades Polymarket 5-min overreactions"
    version = "1.0.0"
    supported_markets = ["prediction"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.run_interval_seconds = self.config.get("interval", 5)
        self.overshoot_threshold = float(self.config.get("overshoot_threshold", 0.08))
        self.min_round_ticks = int(self.config.get("min_round_ticks", 10))
        self.take_profit_pct = Decimal(str(self.config.get("take_profit_pct", "5")))
        self.stop_loss_pct = Decimal(str(self.config.get("stop_loss_pct", "6")))
        self.max_hold_seconds = int(self.config.get("max_hold_seconds", 200))
        self.bet_size_usd = float(self.config.get("bet_size_usd", 5))

        # Per-round tracking (reset each round)
        self._yes_prices: deque = deque(maxlen=200)
        self._no_prices: deque = deque(maxlen=200)
        self._current_round_start: float = 0
        self._last_signal_ts: float = 0
        self._cooldown_seconds = float(self.config.get("cooldown_seconds", 25))

    def _reset_round(self, round_start: float) -> None:
        if round_start != self._current_round_start:
            self._yes_prices.clear()
            self._no_prices.clear()
            self._current_round_start = round_start

    async def analyze(
        self,
        symbol: str,
        candles: List[OHLCV],
        current_price: PriceTick,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Signal]:
        if not context:
            return []

        yes_token = context.get("yes_token_id")
        no_token = context.get("no_token_id")
        if not yes_token or not no_token:
            return []

        round_start = context.get("round_start_ts", 0)
        elapsed = time.time() - round_start if round_start else 999
        remaining = max(300 - elapsed, 0)

        # Need at least 30s of data and 60s remaining
        if remaining < 60 or elapsed < 30:
            return []

        if time.time() - self._last_signal_ts < self._cooldown_seconds:
            return []

        # Reset tracking when a new round starts
        self._reset_round(round_start)

        # -- Get current share prices -------------------------------------
        yes_book = context.get("yes_book", {})
        no_book = context.get("no_book", {})
        yes_bids = yes_book.get("bids", [])
        yes_asks = yes_book.get("asks", [])
        no_bids = no_book.get("bids", [])
        no_asks = no_book.get("asks", [])

        yes_mid = self._mid(yes_bids, yes_asks)
        no_mid = self._mid(no_bids, no_asks)
        if yes_mid is None or no_mid is None:
            return []

        self._yes_prices.append(float(yes_mid))
        self._no_prices.append(float(no_mid))

        if len(self._yes_prices) < self.min_round_ticks:
            return []

        # -- Compute round averages ---------------------------------------
        yes_mean = sum(self._yes_prices) / len(self._yes_prices)
        no_mean = sum(self._no_prices) / len(self._no_prices)

        yes_now = float(yes_mid)
        no_now = float(no_mid)

        yes_deviation = yes_now - yes_mean  # positive = YES spiked up
        no_deviation = no_now - no_mean     # positive = NO spiked up

        signals: List[Signal] = []

        # -- YES overreacted upward → fade it → BUY NO -------------------
        if yes_deviation > self.overshoot_threshold and no_mid < Decimal("0.85"):
            # YES spiked → crowd thinks BTC going up → counter-trade: buy NO
            confidence = min(abs(yes_deviation) / 0.20, 1.0)
            best_ask = Decimal(str(no_asks[0][0])) if no_asks else no_mid
            signals.append(Signal(
                symbol=no_token,
                direction=SignalDirection.LONG,
                confidence=round(confidence, 3),
                entry_price=best_ask,
                take_profit=min(best_ask + Decimal("0.05"), Decimal("0.98")),
                stop_loss=max(best_ask - Decimal("0.05"), Decimal("0.01")),
                order_type=OrderType.LIMIT,
                reason=(
                    f"MeanRev: YES overshoot={yes_deviation:+.3f} "
                    f"(mean={yes_mean:.2f}, now={yes_now:.2f}), "
                    f"BUY NO@{no_mid:.2f}"
                ),
                metadata={
                    "strategy_type": "mean_reversion_5min",
                    "side": "NO",
                    "yes_deviation": yes_deviation,
                    "no_deviation": no_deviation,
                    "yes_mean": yes_mean,
                    "no_mean": no_mean,
                    "remaining_seconds": remaining,
                    "bet_size_usd": self.bet_size_usd,
                },
            ))

        # -- NO overreacted upward → fade it → BUY YES -------------------
        elif no_deviation > self.overshoot_threshold and yes_mid < Decimal("0.85"):
            confidence = min(abs(no_deviation) / 0.20, 1.0)
            best_ask = Decimal(str(yes_asks[0][0])) if yes_asks else yes_mid
            signals.append(Signal(
                symbol=yes_token,
                direction=SignalDirection.LONG,
                confidence=round(confidence, 3),
                entry_price=best_ask,
                take_profit=min(best_ask + Decimal("0.05"), Decimal("0.98")),
                stop_loss=max(best_ask - Decimal("0.05"), Decimal("0.01")),
                order_type=OrderType.LIMIT,
                reason=(
                    f"MeanRev: NO overshoot={no_deviation:+.3f} "
                    f"(mean={no_mean:.2f}, now={no_now:.2f}), "
                    f"BUY YES@{yes_mid:.2f}"
                ),
                metadata={
                    "strategy_type": "mean_reversion_5min",
                    "side": "YES",
                    "yes_deviation": yes_deviation,
                    "no_deviation": no_deviation,
                    "yes_mean": yes_mean,
                    "no_mean": no_mean,
                    "remaining_seconds": remaining,
                    "bet_size_usd": self.bet_size_usd,
                },
            ))

        if signals:
            self._last_signal_ts = time.time()
        return signals

    async def should_exit(
        self,
        symbol: str,
        entry_price: Decimal,
        current_price: Decimal,
        pnl_pct: Decimal,
        holding_seconds: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if pnl_pct >= self.take_profit_pct:
            return True
        if pnl_pct <= -self.stop_loss_pct:
            return True
        if holding_seconds >= self.max_hold_seconds:
            return True
        if context:
            round_start = context.get("round_start_ts", 0)
            if round_start:
                remaining = max(300 - (time.time() - round_start), 0)
                if remaining < 30:
                    return True
        return False

    @staticmethod
    def _mid(bids: list, asks: list) -> Optional[Decimal]:
        if not bids and not asks:
            return None
        best_bid = Decimal(str(bids[0][0])) if bids else Decimal("0")
        best_ask = Decimal(str(asks[0][0])) if asks else Decimal("1")
        return (best_bid + best_ask) / 2
