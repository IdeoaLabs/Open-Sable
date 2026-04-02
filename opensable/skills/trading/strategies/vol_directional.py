"""
Polymarket BTC 5-Min — Volatility-Weighted Directional Strategy

Instead of chasing momentum, this strategy uses realized BTC volatility
to estimate the *statistical probability* that BTC will be higher/lower
after 5 minutes, then buys shares only when the Polymarket price is
cheaper than the fair value implied by that probability.

Edge:  stat_prob(BTC_up) vs YES_share_price
       Only trades when expected value is positive.

Example:
    BTC 5-min realized vol → stdev = $30
    Current BTC move this round = +$15  (0.5 sigma)
    stat_prob(finishing up)  ≈ 69%     (from normal CDF)
    YES share trading at     $0.55
    Edge = 69% - 55% = +14%           → BUY YES
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..base import OHLCV, OrderType, PriceTick
from ..strategy_engine import Signal, SignalDirection, Strategy

logger = logging.getLogger(__name__)


def _rolling_stdev(prices: List[float], window: int = 60) -> float:
    """Standard deviation of price *changes* over the last *window* ticks."""
    if len(prices) < window + 1:
        return 0.0
    changes = [prices[i] - prices[i - 1] for i in range(len(prices) - window, len(prices))]
    if not changes:
        return 0.0
    mean = sum(changes) / len(changes)
    var = sum((c - mean) ** 2 for c in changes) / len(changes)
    return math.sqrt(var)


def _normal_cdf(z: float) -> float:
    """Approximate standard normal CDF (Abramowitz & Stegun)."""
    if z > 6:
        return 1.0
    if z < -6:
        return 0.0
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1 if z >= 0 else -1
    z_abs = abs(z)
    t = 1.0 / (1.0 + p * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z_abs * z_abs / 2)
    return 0.5 * (1.0 + sign * y)


def _orderbook_imbalance(bids: list, asks: list) -> float:
    bid_vol = sum(float(b[1]) for b in bids[:5]) if bids else 0
    ask_vol = sum(float(a[1]) for a in asks[:5]) if asks else 0
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


class VolatilityDirectionalStrategy(Strategy):
    """
    Volatility-weighted directional — buys shares only when statistical
    probability exceeds market price (positive expected value).

    Config:
        min_edge_pct         – minimum edge to trigger (default 8)
        vol_window           – ticks for realized vol (default 60)
        take_profit_pct      – exit % (default 6)
        stop_loss_pct        – stop % (default 4)
        max_hold_seconds     – forced exit (default 240)
        bet_size_usd         – per-trade (default 5)
    """

    name = "vol_directional"
    description = "Volatility-weighted directional — positive EV only"
    version = "1.0.0"
    supported_markets = ["prediction"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.run_interval_seconds = self.config.get("interval", 5)
        self.min_edge_pct = float(self.config.get("min_edge_pct", 8))
        self.vol_window = int(self.config.get("vol_window", 60))
        self.take_profit_pct = Decimal(str(self.config.get("take_profit_pct", "6")))
        self.stop_loss_pct = Decimal(str(self.config.get("stop_loss_pct", "4")))
        self.max_hold_seconds = int(self.config.get("max_hold_seconds", 240))
        self.bet_size_usd = float(self.config.get("bet_size_usd", 5))

        self._btc_history: deque = deque(maxlen=500)
        self._last_signal_ts: float = 0
        self._cooldown_seconds = float(self.config.get("cooldown_seconds", 20))

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

        # -- Collect BTC prices -------------------------------------------
        btc_prices: List[float] = context.get("btc_prices", [])
        if current_price and current_price.price:
            btc_prices.append(float(current_price.price))
        if btc_prices:
            self._btc_history.extend(btc_prices[-5:])
        btc_series = list(self._btc_history)

        if len(btc_series) < self.vol_window + 5:
            return []  # need enough data for vol calc

        # -- Realized volatility ------------------------------------------
        stdev = _rolling_stdev(btc_series, self.vol_window)
        if stdev <= 0:
            return []

        # -- Current round movement ---------------------------------------
        round_start = context.get("round_start_ts", 0)
        elapsed = time.time() - round_start if round_start else 999
        remaining = max(300 - elapsed, 0)

        if remaining < 60:
            return []

        if time.time() - self._last_signal_ts < self._cooldown_seconds:
            return []

        # BTC price at round start (approximate: first price in this round)
        # We use the price from ~elapsed seconds ago
        lookback_idx = min(int(elapsed * 2), len(btc_series) - 1)  # ~2 ticks/sec from WS
        btc_at_start = btc_series[-(lookback_idx + 1)] if lookback_idx < len(btc_series) else btc_series[0]
        btc_now = btc_series[-1]
        move = btc_now - btc_at_start

        # -- Statistical probability (normal model) -----------------------
        # Scale vol for remaining time: vol * sqrt(remaining/300)
        time_frac = remaining / 300.0
        projected_stdev = stdev * math.sqrt(max(time_frac, 0.01)) * math.sqrt(self.vol_window)

        if projected_stdev <= 0:
            return []

        # z-score: how far BTC has already moved relative to expected vol
        z = move / projected_stdev
        prob_up = _normal_cdf(z)  # prob of finishing positive given current move

        # -- Get share prices ---------------------------------------------
        yes_book = context.get("yes_book", {})
        no_book = context.get("no_book", {})
        yes_asks = yes_book.get("asks", [])
        no_asks = no_book.get("asks", [])
        yes_bids = yes_book.get("bids", [])
        no_bids = no_book.get("bids", [])

        yes_mid = self._mid(yes_bids, yes_asks)
        no_mid = self._mid(no_bids, no_asks)
        if yes_mid is None or no_mid is None:
            return []

        signals: List[Signal] = []

        # -- Check for edge -----------------------------------------------
        yes_implied = float(yes_mid) * 100  # share price → implied prob %
        no_implied = float(no_mid) * 100

        edge_yes = prob_up * 100 - yes_implied   # positive → YES underpriced
        edge_no = (1 - prob_up) * 100 - no_implied  # positive → NO underpriced

        if edge_yes >= self.min_edge_pct and yes_mid < Decimal("0.88"):
            confidence = min(edge_yes / 25.0, 1.0)
            best_ask = Decimal(str(yes_asks[0][0])) if yes_asks else yes_mid
            signals.append(Signal(
                symbol=yes_token,
                direction=SignalDirection.LONG,
                confidence=round(confidence, 3),
                entry_price=best_ask,
                take_profit=min(best_ask + Decimal("0.06"), Decimal("0.98")),
                stop_loss=max(best_ask - Decimal("0.04"), Decimal("0.01")),
                order_type=OrderType.LIMIT,
                reason=(
                    f"VolDir: prob_up={prob_up:.0%}, YES@{yes_mid:.2f} "
                    f"(impl={yes_implied:.0f}%), edge={edge_yes:+.1f}%, "
                    f"z={z:+.2f}, stdev=${stdev:.1f}"
                ),
                metadata={
                    "strategy_type": "vol_directional",
                    "side": "YES",
                    "prob_up": prob_up,
                    "z_score": z,
                    "stdev": stdev,
                    "edge_pct": edge_yes,
                    "remaining_seconds": remaining,
                    "bet_size_usd": self.bet_size_usd,
                },
            ))
        elif edge_no >= self.min_edge_pct and no_mid < Decimal("0.88"):
            confidence = min(edge_no / 25.0, 1.0)
            best_ask = Decimal(str(no_asks[0][0])) if no_asks else no_mid
            signals.append(Signal(
                symbol=no_token,
                direction=SignalDirection.LONG,
                confidence=round(confidence, 3),
                entry_price=best_ask,
                take_profit=min(best_ask + Decimal("0.06"), Decimal("0.98")),
                stop_loss=max(best_ask - Decimal("0.04"), Decimal("0.01")),
                order_type=OrderType.LIMIT,
                reason=(
                    f"VolDir: prob_down={1-prob_up:.0%}, NO@{no_mid:.2f} "
                    f"(impl={no_implied:.0f}%), edge={edge_no:+.1f}%, "
                    f"z={z:+.2f}, stdev=${stdev:.1f}"
                ),
                metadata={
                    "strategy_type": "vol_directional",
                    "side": "NO",
                    "prob_up": prob_up,
                    "z_score": z,
                    "stdev": stdev,
                    "edge_pct": edge_no,
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
