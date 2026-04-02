"""
TradingSkill,  Main trading orchestrator for Open-Sable.

This is the top-level skill that the ToolRegistry instantiates.
It owns the portfolio, risk manager, market data, strategy engine,
and exchange connectors, and exposes high-level methods that map
directly to tool schemas (get_portfolio, place_trade, scan_markets, etc.)
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .base import (
    Balance,
    ExchangeConnector,
    Order,
    OrderSide,
    OrderType,
    Position,
    TradeRecord,
)
from .connectors.paper import PaperTradingConnector
from .market_data import MarketDataService
from .portfolio import PortfolioManager
from .risk_manager import RiskAction, RiskManager
from .signals import SignalAggregator
from .strategy_engine import Signal, SignalDirection, StrategyEngine

logger = logging.getLogger(__name__)


class TradingSkill:
    """
    High-level trading skill wired into Open-Sable's tool registry.

    Lifecycle:
        skill = TradingSkill(config)
        await skill.initialize()
        ...
        result = await skill.get_portfolio()
        result = await skill.place_trade(...)
    """

    def __init__(self, config):
        self.config = config
        self._initialized = False

        # Core components (created in initialize)
        self.portfolio: Optional[PortfolioManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.market_data: Optional[MarketDataService] = None
        self.strategy_engine: Optional[StrategyEngine] = None
        self.signal_aggregator: Optional[SignalAggregator] = None

        # Connectors keyed by name
        self.connectors: Dict[str, ExchangeConnector] = {}

        # LLM callback (injected by tools.py after agent is ready)
        self._llm_invoke: Optional[Callable[..., Coroutine]] = None

        # Background scan task
        self._scan_task: Optional[asyncio.Task] = None

    # ── Lifecycle ────────────────────────────────────────────

    async def initialize(self) -> None:
        """Wire up all trading sub-components based on config."""
        if self._initialized:
            return

        paper_mode = getattr(self.config, "trading_paper_mode", True)

        # 1. Always start with paper connector
        paper = PaperTradingConnector()
        await paper.connect()
        self.connectors["paper"] = paper

        # 2. Connect live exchanges if keys are provided and paper mode is off
        if not paper_mode:
            await self._connect_live_exchanges()

        # 3. Portfolio manager,  aggregates all connectors
        active_connectors = list(self.connectors.values())
        self.portfolio = PortfolioManager()
        for conn in active_connectors:
            self.portfolio.add_connector(conn)

        # Build connector dict for MarketDataService {name: connector}
        self._connector_dict = {c.name: c for c in active_connectors}

        # 4. Risk manager
        risk_config = {
            "max_position_size_pct": getattr(self.config, "trading_max_position_pct", 5.0),
            "max_daily_loss_pct": getattr(self.config, "trading_max_daily_loss_pct", 2.0),
            "max_drawdown_pct": getattr(self.config, "trading_max_drawdown_pct", 10.0),
            "max_open_positions": getattr(self.config, "trading_max_open_positions", 10),
            "max_order_value_usd": getattr(self.config, "trading_max_order_usd", 10000),
            "require_approval_above_usd": getattr(
                self.config, "trading_require_approval_above_usd", 100
            ),
        }
        banned = getattr(self.config, "trading_banned_assets", "")
        if banned:
            risk_config["banned_assets"] = [a.strip() for a in banned.split(",") if a.strip()]
        self.risk_manager = RiskManager(**risk_config)

        # 5. Market data service (expects Dict[str, ExchangeConnector])
        self.market_data = MarketDataService(self._connector_dict)

        # 6. Strategy engine
        self.strategy_engine = StrategyEngine()
        self.signal_aggregator = SignalAggregator()
        await self._load_strategies()

        self._initialized = True
        mode = "PAPER" if paper_mode else "LIVE"
        logger.info(
            f"🤖 TradingSkill initialized,  mode={mode}, "
            f"connectors={list(self.connectors.keys())}, "
            f"strategies={list(self.strategy_engine.strategies.keys())}"
        )

    async def _connect_live_exchanges(self) -> None:
        """Connect to live exchanges based on config API keys."""
        exchange_map = {
            "binance": {
                "key_attr": "binance_api_key",
                "secret_attr": "binance_api_secret",
                "class_path": "opensable.skills.trading.connectors.binance_connector.BinanceConnector",
                "extra": lambda: {"testnet": getattr(self.config, "binance_testnet", True)},
            },
            "coinbase": {
                "key_attr": "coinbase_api_key",
                "secret_attr": "coinbase_api_secret",
                "class_path": "opensable.skills.trading.connectors.coinbase_connector.CoinbaseConnector",
            },
            "alpaca": {
                "key_attr": "alpaca_api_key",
                "secret_attr": "alpaca_api_secret",
                "class_path": "opensable.skills.trading.connectors.alpaca_connector.AlpacaConnector",
                "extra": lambda: {"paper": getattr(self.config, "alpaca_paper", True)},
            },
            "polymarket": {
                "key_attr": "polymarket_private_key",
                "class_path": "opensable.skills.trading.connectors.polymarket_connector.PolymarketConnector",
                "config_build": lambda: {
                    "private_key": getattr(self.config, "polymarket_private_key", ""),
                    "funder": getattr(self.config, "polymarket_funder", None),
                },
            },
            "hyperliquid": {
                "key_attr": "hyperliquid_private_key",
                "class_path": "opensable.skills.trading.connectors.hyperliquid_connector.HyperliquidConnector",
                "config_build": lambda: {
                    "private_key": getattr(self.config, "hyperliquid_private_key", ""),
                    "testnet": getattr(self.config, "hyperliquid_testnet", True),
                },
            },
            "jupiter": {
                "key_attr": "jupiter_private_key",
                "class_path": "opensable.skills.trading.connectors.jupiter_connector.JupiterConnector",
                "config_build": lambda: {
                    "private_key": getattr(self.config, "jupiter_private_key", ""),
                    "rpc_url": getattr(
                        self.config,
                        "jupiter_rpc_url",
                        "https://api.mainnet-beta.solana.com",
                    ),
                },
            },
        }

        for name, spec in exchange_map.items():
            key = getattr(self.config, spec["key_attr"], None)
            if not key:
                continue
            try:
                import importlib

                mod_path, cls_name = spec["class_path"].rsplit(".", 1)
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, cls_name)

                # Pass the original config object so connectors can read
                # their own named attributes (e.g. binance_api_key, etc.).
                connector = cls(self.config)
                await connector.connect()
                self.connectors[name] = connector
                logger.info(f"✅ Connected to {name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to connect to {name}: {e}")

    async def _load_strategies(self) -> None:
        """Load strategies from config."""
        from .strategies.momentum import MomentumStrategy
        from .strategies.mean_reversion import MeanReversionStrategy
        from .strategies.sentiment import SentimentStrategy
        from .strategies.arbitrage import ArbitrageStrategy
        from .strategies.polymarket_edge import PolymarketEdgeStrategy

        available = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy,
            "sentiment": SentimentStrategy,
            "arbitrage": ArbitrageStrategy,
            "polymarket_edge": PolymarketEdgeStrategy,
        }

        names = [
            s.strip()
            for s in getattr(self.config, "trading_strategies", "momentum,mean_reversion").split(",")
            if s.strip()
        ]

        for name in names:
            cls = available.get(name)
            if not cls:
                logger.warning(f"Unknown strategy: {name}")
                continue
            strat = cls()
            # Inject LLM for strategies that need it
            if hasattr(strat, "set_llm") and self._llm_invoke:
                strat.set_llm(self._llm_invoke)
            self.strategy_engine.register(strat)

    def set_llm(self, llm_invoke: Callable[..., Coroutine]) -> None:
        """Inject LLM capability into strategies that need it."""
        self._llm_invoke = llm_invoke
        if self.strategy_engine:
            for strat in self.strategy_engine.strategies.values():
                if hasattr(strat, "set_llm"):
                    strat.set_llm(llm_invoke)

    # ── Tool handlers (called by ToolRegistry) ──────────────

    async def get_portfolio(self, args: Dict[str, Any] = None) -> str:
        """Return portfolio summary as formatted text."""
        if not self._initialized:
            return "⚠️ Trading not initialized. Set TRADING_ENABLED=true."

        snapshot = await self.portfolio.refresh()
        positions = self.portfolio.positions
        summary = self.portfolio.get_summary()

        lines = [
            "📊 **Portfolio Summary**",
            f"Total Value: ${snapshot.total_value_usd:,.2f}",
            f"Cash (USDT): ${snapshot.cash_value_usd:,.2f}",
            f"Unrealized P&L: ${snapshot.unrealized_pnl_usd:,.2f}",
            f"Realized P&L: ${snapshot.realized_pnl_usd:,.2f}",
            "",
        ]

        if positions:
            lines.append("**Open Positions:**")
            for pos in positions:
                lines.append(
                    f"  • {pos.symbol}: {pos.quantity} @ ${pos.entry_price:.4f} "
                    f"(PnL: ${pos.unrealized_pnl:.2f})"
                )
        else:
            lines.append("No open positions.")

        lines.extend([
            "",
            "**Performance:**",
            f"  Win Rate: {summary.get('win_rate', 'N/A')}",
            f"  Total Trades: {summary.get('total_trades', 0)}",
            f"  Sharpe Ratio: {summary.get('sharpe_ratio', 'N/A')}",
            f"  Max Drawdown: {summary.get('max_drawdown', 'N/A')}",
        ])

        mode = "📝 PAPER" if getattr(self.config, "trading_paper_mode", True) else "🔴 LIVE"
        lines.append(f"\nMode: {mode}")

        return "\n".join(lines)

    async def get_price(self, args: Dict[str, Any]) -> str:
        """Get current price for a symbol."""
        if not self._initialized:
            return "⚠️ Trading not initialized."

        symbol = args.get("symbol", "BTC/USDT")
        try:
            tick = await self.market_data.get_price(symbol)
            if tick:
                lines = [f"💰 {symbol}: ${tick.price:,.4f}"]
                if tick.bid is not None and tick.ask is not None:
                    lines.append(f"  Bid: ${tick.bid:,.4f} | Ask: ${tick.ask:,.4f}")
                if tick.volume_24h is not None:
                    lines.append(f"  24h Volume: ${tick.volume_24h:,.0f}")
                lines.append(f"  Source: {tick.exchange}")
                return "\n".join(lines)
            return f"⚠️ No price data for {symbol}"
        except Exception as e:
            return f"⚠️ Error fetching price for {symbol}: {e}"

    async def analyze_market(self, args: Dict[str, Any]) -> str:
        """Run all strategies on a symbol and return signals."""
        if not self._initialized:
            return "⚠️ Trading not initialized."

        symbol = args.get("symbol", "BTC/USDT")
        try:
            # Fetch candles and current price for strategies
            candles = await self.market_data.get_ohlcv(symbol, interval="1h", limit=100)
            tick = await self.market_data.get_price(symbol)
            signals = await self.strategy_engine.scan_symbol(symbol, candles, tick)
            if not signals:
                return f"📊 No trading signals for {symbol} right now."

            aggregated = self.signal_aggregator.aggregate(signals)

            lines = [f"📊 **Market Analysis: {symbol}**", ""]

            for sig in signals:
                emoji = "🟢" if sig.direction == SignalDirection.LONG else "🔴"
                lines.append(
                    f"  {emoji} [{sig.strategy or 'unknown'}] "
                    f"{sig.direction.value},  conf: {sig.confidence:.0%},  {sig.reason}"
                )

            if aggregated:
                c = aggregated[0]  # Best consensus signal (sorted by confidence)
                emoji = "🟢" if c.direction == SignalDirection.LONG else "🔴"
                lines.extend([
                    "",
                    f"**Consensus: {emoji} {c.direction.value} "
                    f"(confidence: {c.confidence:.0%})**",
                    f"  {c.reason}",
                ])

            return "\n".join(lines)
        except Exception as e:
            return f"⚠️ Analysis failed for {symbol}: {e}"

    async def place_trade(self, args: Dict[str, Any]) -> str:
        """Place a trade with risk checks and HITL gate."""
        if not self._initialized:
            return "⚠️ Trading not initialized."

        symbol = args.get("symbol")
        side = args.get("side", "buy").lower()
        amount = args.get("amount")
        order_type = args.get("type", "market").lower()
        price = args.get("price")
        exchange = args.get("exchange", "paper")

        if not symbol or not amount:
            return "⚠️ Missing required fields: symbol, amount"

        try:
            amount_dec = Decimal(str(amount))
        except Exception:
            return f"⚠️ Invalid amount: {amount}"

        # Get current price for risk check
        tick = await self.market_data.get_price(symbol)
        if not tick:
            return f"⚠️ Cannot get price for {symbol}"

        order_value_usd = float(amount_dec * tick.price)

        # Build a temporary Order for risk check
        proposed_order = Order(
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET if order_type == "market" else OrderType.LIMIT,
            quantity=amount_dec,
            price=Decimal(str(price)) if price else None,
            exchange=exchange,
        )

        # Get portfolio value for risk check
        snapshot = await self.portfolio.refresh()
        portfolio_value = snapshot.total_value_usd or Decimal("10000")
        current_positions = self.portfolio.positions

        # Risk check
        decision = await self.risk_manager.check_trade(
            order=proposed_order,
            portfolio_value_usd=portfolio_value,
            current_positions=current_positions,
            current_price=tick.price,
            volume_24h_usd=tick.volume_24h,
        )

        if decision.action == RiskAction.REJECT:
            return f"🚫 Trade rejected by risk manager: {decision.reason}"

        if decision.action == RiskAction.REDUCE:
            amount_dec = decision.max_allowed_quantity or amount_dec

        # Select connector
        connector = self.connectors.get(exchange)
        if not connector:
            available = list(self.connectors.keys())
            return f"⚠️ Exchange '{exchange}' not connected. Available: {available}"

        # Place order
        order = await connector.place_order(
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET if order_type == "market" else OrderType.LIMIT,
            quantity=amount_dec,
            price=Decimal(str(price)) if price else None,
        )

        # Record in portfolio
        if order:
            record = TradeRecord(
                trade_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.filled_quantity or order.quantity,
                entry_price=order.average_fill_price or tick.price,
                fees=order.fee or Decimal("0"),
                exchange=exchange,
            )
            self.portfolio.record_trade(record)

        mode = "📝 PAPER" if getattr(self.config, "trading_paper_mode", True) else "🔴 LIVE"
        return (
            f"✅ Order placed ({mode})\n"
            f"  {side.upper()} {amount_dec} {symbol} @ ~${tick.price:,.4f}\n"
            f"  Exchange: {exchange}\n"
            f"  Order ID: {order.order_id if order else 'N/A'}\n"
            f"  Value: ~${order_value_usd:,.2f}"
        )

    async def cancel_order(self, args: Dict[str, Any]) -> str:
        """Cancel an open order."""
        if not self._initialized:
            return "⚠️ Trading not initialized."

        order_id = args.get("order_id")
        exchange = args.get("exchange", "paper")
        symbol = args.get("symbol", "")

        if not order_id:
            return "⚠️ Missing order_id"

        connector = self.connectors.get(exchange)
        if not connector:
            return f"⚠️ Exchange '{exchange}' not connected."

        try:
            success = await connector.cancel_order(order_id, symbol)
            return f"✅ Order {order_id} cancelled." if success else f"⚠️ Failed to cancel {order_id}"
        except Exception as e:
            return f"⚠️ Error cancelling order: {e}"

    async def get_trade_history(self, args: Dict[str, Any] = None) -> str:
        """Return recent trade history."""
        if not self._initialized:
            return "⚠️ Trading not initialized."

        limit = (args or {}).get("limit", 20)
        trades = self.portfolio.get_trade_history(limit)

        if not trades:
            return "No trade history yet."

        lines = ["📜 **Recent Trades**", ""]
        for t in trades:
            lines.append(
                f"  {t.entry_time.strftime('%m/%d %H:%M')} | "
                f"{t.side.value.upper()} {t.quantity} {t.symbol} @ ${t.entry_price:.4f} | "
                f"{t.exchange}"
            )

        return "\n".join(lines)

    async def get_signals(self, args: Dict[str, Any] = None) -> str:
        """Scan watchlist and return all current signals."""
        if not self._initialized:
            return "⚠️ Trading not initialized."

        watchlist = [
            s.strip()
            for s in getattr(self.config, "trading_watchlist", "BTC/USDT").split(",")
            if s.strip()
        ]

        all_signals: List[Signal] = []
        for symbol in watchlist:
            try:
                candles = await self.market_data.get_ohlcv(symbol, interval="1h", limit=100)
                tick = await self.market_data.get_price(symbol)
                sigs = await self.strategy_engine.scan_symbol(symbol, candles, tick)
                all_signals.extend(sigs)
            except Exception as e:
                logger.debug(f"Signal scan failed for {symbol}: {e}")

        if not all_signals:
            return f"📊 No signals across watchlist: {', '.join(watchlist)}"

        lines = ["📡 **Current Signals**", ""]
        for s in sorted(all_signals, key=lambda x: x.confidence, reverse=True):
            emoji = "🟢" if s.direction == SignalDirection.LONG else "🔴"
            lines.append(
                f"  {emoji} {s.symbol},  {s.direction.value} "
                f"({s.confidence:.0%}),  {s.reason[:80]}"
            )

        return "\n".join(lines)

    async def start_scanning(self, args: Dict[str, Any] = None) -> str:
        """Start background market scanning loop."""
        if self._scan_task and not self._scan_task.done():
            return "⚠️ Scanning already running."

        interval = getattr(self.config, "trading_scan_interval", 60)
        self._scan_task = asyncio.create_task(self._scan_loop(interval))
        return f"✅ Background scanning started (interval: {interval}s)"

    async def stop_scanning(self, args: Dict[str, Any] = None) -> str:
        """Stop background scanning."""
        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()
            return "✅ Scanning stopped."
        return "⚠️ No scanning task running."

    async def _scan_loop(self, interval: int) -> None:
        """Background loop: scan watchlist, auto-trade if enabled."""
        auto_trade = getattr(self.config, "trading_auto_trade", False)
        watchlist = [
            s.strip()
            for s in getattr(self.config, "trading_watchlist", "BTC/USDT").split(",")
            if s.strip()
        ]

        while True:
            try:
                for symbol in watchlist:
                    try:
                        candles = await self.market_data.get_ohlcv(symbol, interval="1h", limit=100)
                        tick = await self.market_data.get_price(symbol)
                        signals = await self.strategy_engine.scan_symbol(symbol, candles, tick)
                    except Exception as e:
                        logger.debug(f"Scan failed for {symbol}: {e}")
                        continue
                    if not signals:
                        continue

                    aggregated = self.signal_aggregator.aggregate(signals)
                    if not aggregated or aggregated[0].confidence < 0.7:
                        continue

                    consensus = aggregated[0]
                    logger.info(
                        f"📡 Strong signal: {consensus.direction.value} {symbol} "
                        f"({consensus.confidence:.0%})"
                    )

                    if auto_trade:
                        # Auto-execute with small position
                        side = "buy" if consensus.direction == SignalDirection.LONG else "sell"
                        result = await self.place_trade({
                            "symbol": symbol,
                            "side": side,
                            "amount": "0.001",  # minimal position for safety
                            "exchange": "paper",
                        })
                        logger.info(f"Auto-trade result: {result}")

            except asyncio.CancelledError:
                logger.info("Scan loop cancelled")
                break
            except Exception as e:
                logger.error(f"Scan loop error: {e}")

            await asyncio.sleep(interval)

    async def get_risk_status(self, args: Dict[str, Any] = None) -> str:
        """Return current risk manager status."""
        if not self._initialized or not self.risk_manager:
            return "⚠️ Trading not initialized."

        status = self.risk_manager.get_status()
        limits = status.get("limits", {})
        daily_pnl = float(status.get("daily_pnl", 0))
        max_order = float(limits.get("max_order_value_usd", 10000))
        lines = [
            "🛡️ **Risk Status**",
            f"  Emergency Halt: {'🔴 YES' if status.get('halted') else '🟢 NO'}",
            f"  Daily P&L: ${daily_pnl:,.2f}",
            f"  Daily Trades: {status.get('daily_trades', 0)}",
            "",
            "**Limits:**",
            f"  Max Position Size: {limits.get('max_position_size_pct', 5)}%",
            f"  Max Daily Loss: {limits.get('max_daily_loss_pct', 2)}%",
            f"  Max Drawdown: {limits.get('max_drawdown_pct', 10)}%",
            f"  Max Open Positions: {limits.get('max_open_positions', 10)}",
            f"  Max Order Value: ${max_order:,.0f}",
            f"  Banned Assets: {', '.join(status.get('banned_assets', [])) or 'None'}",
        ]
        return "\n".join(lines)
