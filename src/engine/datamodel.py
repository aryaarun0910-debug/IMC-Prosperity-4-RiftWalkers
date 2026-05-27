"""
IMC Prosperity Datamodel -- Offline Compatible Version
Mirrors the real IMC datamodel.py so strategies work both locally and in competition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Type alias
Symbol = str


@dataclass
class Order:
    symbol: Symbol
    price: int
    quantity: int  # positive = buy, negative = sell

    def __repr__(self):
        side = "BUY" if self.quantity > 0 else "SELL"
        return f"Order({side} {abs(self.quantity)}x {self.symbol} @ {self.price})"


@dataclass
class OrderDepth:
    buy_orders:  Dict[int, int] = field(default_factory=dict)   # {price: +volume}
    sell_orders: Dict[int, int] = field(default_factory=dict)   # {price: -volume}

    def best_bid(self) -> Optional[int]:
        return max(self.buy_orders) if self.buy_orders else None

    def best_ask(self) -> Optional[int]:
        return min(self.sell_orders) if self.sell_orders else None

    def mid_price(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        return (b + a) / 2 if b is not None and a is not None else None

    def spread(self) -> Optional[int]:
        b, a = self.best_bid(), self.best_ask()
        return a - b if b is not None and a is not None else None


@dataclass
class Trade:
    symbol: Symbol
    price: int
    quantity: int
    buyer: str = ""
    seller: str = ""
    timestamp: int = 0


@dataclass
class ConversionObservation:
    bidPrice: float = 0.0
    askPrice: float = 0.0
    transportFees: float = 0.0
    exportTariff: float = 0.0
    importTariff: float = 0.0
    sunshineFraction: float = 0.0
    humidity: float = 0.0


@dataclass
class Observation:
    plainValueObservations: Dict[Symbol, float] = field(default_factory=dict)
    conversionObservations: Dict[Symbol, ConversionObservation] = field(default_factory=dict)


@dataclass
class TradingState:
    timestamp: int
    traderData: str
    listings: Dict[Symbol, Any]
    order_depths: Dict[Symbol, OrderDepth]
    own_trades: Dict[Symbol, List[Trade]]
    market_trades: Dict[Symbol, List[Trade]]
    position: Dict[Symbol, int]
    observations: Observation
