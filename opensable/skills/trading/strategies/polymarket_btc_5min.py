"""
Polymarket BTC 5-Min Up/Down Scalper

Trades the Polymarket 5-minute BTC up/down binary market by buying
shares (YES or NO) when they're cheap and selling before the 5-min
window resolves.  Does NOT wait for resolution — it scalps the
order-book spread and price momentum within each round.

Strategy:
    1.  Continuously poll the BTC spot price (Binance/Coinbase)
        and the Polymarket 5-min YES/NO order books.
    2.  When BTC is strongly trending up → YES shares become more
        valuable.  Buy YES early (when cheap) and sell once the
        share price moves up.  Vice-versa for downtrends.
    3.  Use order-book imbalance + BTC momentum as the signal.
    4.  Exit quickly — take small profits (3-8 %) per round and
        cut losses at -5 %.  Never hold through resolution unless
        the position is already deep in profit.

Requires:
    - POLYMARKET_PRIVATE_KEY (for placing orders)
    - A BTC price feed (Binance or any crypto connector)
    - The token_ids for the current 5-min YES and NO outcomes
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from decimal import Decimal
from typing import Any, Callable, Coroutine, Dict, List, Optional

from ..base import OHLCV, OrderType, PriceTick
from ..strategy_engine import Signal, SignalDirection, Strategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _momentum(prices: List[float], lookback: int = 10) -> float:
    """Simple price momentum: percent change over *lookback* ticks."""
    if len(prices) < lookback + 1:
        return 0.0
    old = prices[-(lookback + 1)]
    if old == 0:
        return 0.0
    return (prices[-1] - old) / old * 100


def _ema(values: List[float], span: int) -> float:
    """Exponential moving average of the last *span* values."""
    if not values:
        return 0.0
    k = 2 / (span + 1)
    ema_val = values[0]
    for v in values[1:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val


def _orderbook_imbalance(bids: list, asks: list) -> float:
    """
    Return a value in [-1, 1].
    +1 = all pressure on the bid side (bullish)
    -1 = all pressure on the ask side (bearish)
    """
    bid_vol = sum(float(b[1]) for b in bids[:5]) if bids else 0
    ask_vol = sum(float(a[1]) for a in asks[:5]) if asks else 0
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


class PolymarketBtc5MinStrategy(Strategy):
    """
    Scalps the Polymarket 5-min BTC up/down market.

    Config keys (all optional, sensible defaults provided):
        interval             – seconds between scans (default 5)
        btc_momentum_window  – BTC price ticks for momentum (default 12)
        momentum_threshold   – % move to trigger signal (default 0.04)
        imbalance_threshold  – order-book imbalance threshold (default 0.20)
        take_profit_pct      – exit at this % gain (default 5)
        stop_loss_pct        – exit at this % loss (default 5)
        max_hold_seconds     – max hold time before forced exit (default 240)
        bet_size_usd         – flat bet per trade in USD (default 5)

    Context keys (must be supplied by the scan loop):
        yes_token_id         – Polymarket YES token id for current round
        no_token_id          – Polymarket NO token id for current round
        yes_book             – {"bids": [...], "asks": [...]}
        no_book              – {"bids": [...], "asks": [...]}
        round_start_ts       – unix timestamp when the current 5-min round started
        btc_prices           – list[float] of recent BTC spot prices
    """

    name = "polymarket_btc_5min"
    description = "Scalps Polymarket 5-min BTC up/down — buys low, sells high before resolution"
    version = "1.0.0"
    supported_markets = ["prediction"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.run_interval_seconds = self.config.get("interval", 5)

        # Signal thresholds
        self.btc_momentum_window = int(self.config.get("btc_momentum_window", 12))
        self.momentum_threshold = float(self.config.get("momentum_threshold", 0.04))
        self.imbalance_threshold = float(self.config.get("imbalance_threshold", 0.20))

        # Risk / position management
        self.take_profit_pct = Decimal(str(self.config.get("take_profit_pct", "5")))
        self.stop_loss_pct = Decimal(str(self.config.get("stop_loss_pct", "5")))
        self.max_hold_seconds = int(self.config.get("max_hold_seconds", 240))
        self.bet_size_usd = float(self.config.get("bet_size_usd", 5))

        # Internal state
        self._btc_history: deque = deque(maxlen=200)
        self._last_signal_ts: float = 0
        self._cooldown_seconds = float(self.config.get("cooldown_seconds", 15))

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

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
            logger.debug("polymarket_btc_5min: missing token ids in context")
            return []

        # -- BTC momentum --------------------------------------------------
        btc_prices: List[float] = context.get("btc_prices", [])
        if current_price and current_price.price:
            btc_prices.append(float(current_price.price))
        # Also maintain internal history
        if btc_prices:
            self._btc_history.extend(btc_prices[-5:])
        btc_series = list(self._btc_history)

        mom = _momentum(btc_series, self.btc_momentum_window)
        short_mom = _momentum(btc_series, 3)  # very short-term kick

        # -- Order-book imbalance for YES/NO shares ------------------------
        yes_book = context.get("yes_book", {})
        no_book = context.get("no_book", {})
        yes_imb = _orderbook_imbalance(
            yes_book.get("bids", []), yes_book.get("asks", [])
        )
        no_imb = _orderbook_imbalance(
            no_book.get("bids", []), no_book.get("asks", [])
        )

        # -- Current share prices -----------------------------------------
        yes_bids = yes_book.get("bids", [])
        yes_asks = yes_book.get("asks", [])
        no_bids = no_book.get("bids", [])
        no_asks = no_book.get("asks", [])

        yes_mid = self._mid(yes_bids, yes_asks)
        no_mid = self._mid(no_bids, no_asks)

        if yes_mid is None or no_mid is None:
            return []

        # -- Time remaining in round ---------------------------------------
        round_start = context.get("round_start_ts", 0)
        elapsed = time.time() - round_start if round_start else 999
        remaining = max(300 - elapsed, 0)

        # Don't enter new positions in the last 60 s — too risky
        if remaining < 60:
            return []

        # -- Cooldown: avoid rapid-fire signals ----------------------------
        if time.time() - self._last_signal_ts < self._cooldown_seconds:
            return []

        # ------------------------------------------------------------------
        # Composite score
        # ------------------------------------------------------------------
        # Positive score → BTC trending up → buy YES
        # Negative score → BTC trending down → buy NO
        score = 0.0

        # Momentum component (weighted 60 %)
        if abs(mom) >= self.momentum_threshold:
            score += mom * 15  # scale it up

        # Short-term kick (weighted 20 %)
        score += short_mom * 5

        # Order-book imbalance component (weighted 20 %)
        # If YES book is bullish (lots of bids) → score up
        score += yes_imb * 2
        # If NO book is bullish → score down
        score -= no_imb * 2

        signals: List[Signal] = []

        # --- BUY YES (BTC going UP) ---
        if score > 1.0 and yes_mid < Decimal("0.85"):
            confidence = min(abs(score) / 5.0, 1.0)
            best_ask = Decimal(str(yes_asks[0][0])) if yes_asks else yes_mid
            signals.append(Signal(
                symbol=yes_token,
                direction=SignalDirection.LONG,
                confidence=round(confidence, 3),
                entry_price=best_ask,
                take_profit=min(best_ask + Decimal("0.05"), Decimal("0.98")),
                stop_loss=max(best_ask - Decimal("0.04"), Decimal("0.01")),
                order_type=OrderType.LIMIT,
                reason=(
                    f"BTC 5min UP: mom={mom:+.3f}%, ob_imb={yes_imb:+.2f}, "
                    f"YES@{yes_mid}, score={score:.1f}"
                ),
                metadata={
                    "strategy_type": "polymarket_btc_5min",
                    "side": "YES",
                    "btc_momentum": mom,
                    "short_momentum": short_mom,
                    "yes_imbalance": yes_imb,
                    "no_imbalance": no_imb,
                    "score": score,
                    "remaining_seconds": remaining,
                    "bet_size_usd": self.bet_size_usd,
                },
            ))

        # --- BUY NO (BTC going DOWN) ---
        elif score < -1.0 and no_mid < Decimal("0.85"):
            confidence = min(abs(score) / 5.0, 1.0)
            best_ask = Decimal(str(no_asks[0][0])) if no_asks else no_mid
            signals.append(Signal(
                symbol=no_token,
                direction=SignalDirection.LONG,
                confidence=round(confidence, 3),
                entry_price=best_ask,
                take_profit=min(best_ask + Decimal("0.05"), Decimal("0.98")),
                stop_loss=max(best_ask - Decimal("0.04"), Decimal("0.01")),
                order_type=OrderType.LIMIT,
                reason=(
                    f"BTC 5min DOWN: mom={mom:+.3f}%, ob_imb={no_imb:+.2f}, "
                    f"NO@{no_mid}, score={score:.1f}"
                ),
                metadata={
                    "strategy_type": "polymarket_btc_5min",
                    "side": "NO",
                    "btc_momentum": mom,
                    "short_momentum": short_mom,
                    "yes_imbalance": yes_imb,
                    "no_imbalance": no_imb,
                    "score": score,
                    "remaining_seconds": remaining,
                    "bet_size_usd": self.bet_size_usd,
                },
            ))

        if signals:
            self._last_signal_ts = time.time()

        return signals

    # ------------------------------------------------------------------
    # Exit logic — fast exits, never hold through resolution
    # ------------------------------------------------------------------

    async def should_exit(
        self,
        symbol: str,
        entry_price: Decimal,
        current_price: Decimal,
        pnl_pct: Decimal,
        holding_seconds: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        # Take profit
        if pnl_pct >= self.take_profit_pct:
            return True

        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return True

        # Time-based exit: if we've held longer than max, get out
        if holding_seconds >= self.max_hold_seconds:
            return True

        # If the round is about to resolve (< 30 s left) get out
        if context:
            round_start = context.get("round_start_ts", 0)
            if round_start:
                remaining = max(300 - (time.time() - round_start), 0)
                if remaining < 30:
                    return True

        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mid(bids: list, asks: list) -> Optional[Decimal]:
        if not bids and not asks:
            return None
        best_bid = Decimal(str(bids[0][0])) if bids else Decimal("0")
        best_ask = Decimal(str(asks[0][0])) if asks else Decimal("1")
        return (best_bid + best_ask) / 2
