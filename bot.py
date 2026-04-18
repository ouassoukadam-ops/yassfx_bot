from io import BytesIO
import logging
import os
from typing import Any

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8289829407:AAEhp1T3M_jL6mWeVVOrFHn4lIAm86QSOVA")

BINANCE_BASE_URL = "https://api.binance.com"
GOLD_API_URL = "https://api.gold-api.com/price/XAU"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"

KLINE_INTERVAL = "5m"
KLINE_LIMIT = 200
RISK_REWARD_RATIO = 1.5
ATR_SL_MULTIPLIER = 1.2
POSITION_SIZE = 0.01
TRADE_HORIZON = "~1 hour"

session: aiohttp.ClientSession | None = None

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def post_init(app: Application) -> None:
    global session
    timeout = aiohttp.ClientTimeout(total=15)
    session = aiohttp.ClientSession(timeout=timeout)


async def post_shutdown(app: Application) -> None:
    global session
    if session and not session.closed:
        await session.close()


async def http_get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    if session is None:
        raise RuntimeError("HTTP session is not initialized")

    async with session.get(url, params=params) as response:
        response.raise_for_status()
        return await response.json()


async def fetch_binance_klines(
    symbol: str,
    interval: str = KLINE_INTERVAL,
    limit: int = KLINE_LIMIT,
) -> pd.DataFrame:
    data = await http_get_json(
        f"{BINANCE_BASE_URL}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )

    if not isinstance(data, list) or not data:
        raise ValueError(f"No Binance kline data returned for {symbol}")

    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df


async def fetch_order_book_pressure(symbol: str, limit: int = 20) -> dict[str, float]:
    data = await http_get_json(
        f"{BINANCE_BASE_URL}/api/v3/depth",
        params={"symbol": symbol, "limit": limit},
    )

    bids = sum(float(level[1]) for level in data.get("bids", []))
    asks = sum(float(level[1]) for level in data.get("asks", []))
    total = bids + asks
    buy_pressure = (bids / total) if total else 0.5

    return {
        "bids_volume": bids,
        "asks_volume": asks,
        "buy_pressure": buy_pressure,
    }


async def fetch_ticker_stats(symbol: str) -> dict[str, Any]:
    return await http_get_json(
        f"{BINANCE_BASE_URL}/api/v3/ticker/24hr",
        params={"symbol": symbol},
    )


async def fetch_fear_and_greed() -> dict[str, Any] | None:
    try:
        data = await http_get_json(FEAR_GREED_URL)
        values = data.get("data", [])
        if not values:
            return None
        current = values[0]
        return {
            "value": int(current["value"]),
            "classification": current["value_classification"],
        }
    except Exception as exc:
        logger.warning("Fear & Greed API unavailable: %s", exc)
        return None


async def fetch_gold_price() -> float:
    data = await http_get_json(GOLD_API_URL)
    if "price" not in data:
        raise ValueError("Gold API did not return a price")
    return float(data["price"])


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)


def macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, 12)
    ema_slow = ema(series, 26)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=period).mean()


def support_resistance(df: pd.DataFrame, window: int = 12) -> tuple[float, float]:
    support = float(df["low"].rolling(window=window).min().iloc[-1])
    resistance = float(df["high"].rolling(window=window).max().iloc[-1])
    return support, resistance


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma20"] = sma(df["close"], 20)
    df["ma50"] = sma(df["close"], 50)
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["ema20"] = ema(df["close"], 20)
    df["rsi14"] = rsi(df["close"], 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = bollinger_bands(df["close"], 20, 2.0)
    df["atr14"] = atr(df, 14)
    df["volume_ma20"] = sma(df["volume"], 20)
    df["volume_ma10"] = sma(df["volume"], 10)
    return df


def compute_trade_levels(price: float, atr_value: float, side: str) -> tuple[float, float]:
    sl_distance = atr_value * ATR_SL_MULTIPLIER
    tp_distance = sl_distance * RISK_REWARD_RATIO

    if side == "BUY":
        stop_loss = price - sl_distance
        take_profit = price + tp_distance
    else:
        stop_loss = price + sl_distance
        take_profit = price - tp_distance

    return round(stop_loss, 4), round(take_profit, 4)


def build_indicator_groups(df: pd.DataFrame) -> dict[str, Any]:
    last = df.iloc[-1]
    price = float(last["close"])

    trend_score = 0.0
    momentum_score = 0.0
    volatility_score = 0.0
    volume_score = 0.0

    trend_reasons: list[str] = []
    momentum_reasons: list[str] = []
    volatility_reasons: list[str] = []
    volume_reasons: list[str] = []

    if price > last["ema9"] > last["ema21"]:
        trend_score += 2
        trend_reasons.append("Price > EMA9 > EMA21")
    elif price < last["ema9"] < last["ema21"]:
        trend_score -= 2
        trend_reasons.append("Price < EMA9 < EMA21")
    else:
        trend_reasons.append("Trend alignment is mixed")

    if price > last["ma20"]:
        trend_score += 1
        trend_reasons.append("Price above MA20")
    elif price < last["ma20"]:
        trend_score -= 1
        trend_reasons.append("Price below MA20")

    if last["macd"] > last["macd_signal"] and last["macd_hist"] > 0:
        momentum_score += 2
        momentum_reasons.append("MACD bullish crossover")
    elif last["macd"] < last["macd_signal"] and last["macd_hist"] < 0:
        momentum_score -= 2
        momentum_reasons.append("MACD bearish crossover")

    if 50 <= last["rsi14"] <= 65:
        momentum_score += 1
        momentum_reasons.append("RSI supports continuation")
    elif 35 <= last["rsi14"] < 50:
        momentum_score -= 0.5
        momentum_reasons.append("RSI below neutral")
    elif last["rsi14"] > 75:
        momentum_score -= 1
        momentum_reasons.append("RSI overbought")
    elif last["rsi14"] < 25:
        momentum_score += 0.5
        momentum_reasons.append("RSI oversold bounce zone")

    atr_value = float(last["atr14"]) if pd.notna(last["atr14"]) else 0.0
    bb_width = 0.0
    if pd.notna(last["bb_upper"]) and pd.notna(last["bb_lower"]):
        bb_width = float(last["bb_upper"] - last["bb_lower"])

    if atr_value > 0:
        volatility_score += 1
        volatility_reasons.append(f"ATR active ({atr_value:.4f})")
    else:
        volatility_reasons.append("ATR unavailable")

    if bb_width > 0:
        volatility_score += 1
        volatility_reasons.append(f"Bollinger width active ({bb_width:.4f})")
    else:
        volatility_reasons.append("Bollinger width unavailable")

    if last["volume"] > last["volume_ma10"]:
        volume_score += 1.5
        volume_reasons.append("Volume above MA10")
    else:
        volume_score -= 0.5
        volume_reasons.append("Volume below MA10")

    if last["volume"] > last["volume_ma20"]:
        volume_score += 1
        volume_reasons.append("Volume above MA20")

    return {
        "trend": {"score": round(trend_score, 2), "reasons": trend_reasons},
        "momentum": {"score": round(momentum_score, 2), "reasons": momentum_reasons},
        "volatility": {"score": round(volatility_score, 2), "reasons": volatility_reasons},
        "volume": {"score": round(volume_score, 2), "reasons": volume_reasons},
    }


def build_market_sentiment(
    book_pressure: dict[str, float],
    ticker_stats: dict[str, Any],
    fear_greed: dict[str, Any] | None,
    one_hour_change_pct: float,
) -> dict[str, Any]:
    score = 0.0
    reasons: list[str] = []

    buy_pressure = book_pressure.get("buy_pressure", 0.5)
    if buy_pressure > 0.56:
        score += 2
        reasons.append("Order book buyers dominate")
    elif buy_pressure < 0.44:
        score -= 2
        reasons.append("Order book sellers dominate")
    else:
        reasons.append("Order book balanced")

    try:
        price_change_pct = float(ticker_stats.get("priceChangePercent", 0))
        if price_change_pct > 1:
            score += 1
            reasons.append("24h change positive")
        elif price_change_pct < -1:
            score -= 1
            reasons.append("24h change negative")
    except Exception:
        reasons.append("24h change unavailable")

    if one_hour_change_pct > 0.35:
        score += 1
        reasons.append("1h momentum positive")
    elif one_hour_change_pct < -0.35:
        score -= 1
        reasons.append("1h momentum negative")

    if fear_greed:
        fg_value = fear_greed["value"]
        if fg_value >= 60:
            score += 1
            reasons.append(f"Fear & Greed bullish ({fg_value})")
        elif fg_value <= 40:
            score -= 1
            reasons.append(f"Fear & Greed bearish ({fg_value})")
        else:
            reasons.append(f"Fear & Greed neutral ({fg_value})")
    else:
        reasons.append("Fear & Greed unavailable")

    if score >= 2:
        label = "BULLISH"
    elif score <= -2:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return {
        "label": label,
        "score": round(score, 2),
        "reasons": reasons[:4],
    }


def build_health_score(
    indicator_groups: dict[str, Any],
    market_sentiment: dict[str, Any],
) -> dict[str, Any]:
    raw_score = (
        indicator_groups["trend"]["score"] * 8
        + indicator_groups["momentum"]["score"] * 7
        + indicator_groups["volume"]["score"] * 5
        + indicator_groups["volatility"]["score"] * 3
        + market_sentiment["score"] * 6
    )

    normalized = max(0, min(100, 50 + raw_score))

    if normalized >= 75:
        status = "STRONG"
    elif normalized >= 60:
        status = "HEALTHY"
    elif normalized >= 40:
        status = "MIXED"
    else:
        status = "WEAK"

    return {
        "score": round(normalized, 2),
        "status": status,
    }


def score_signal(
    df: pd.DataFrame,
    book_pressure: dict[str, float],
    ticker_stats: dict[str, Any],
    fear_greed: dict[str, Any] | None,
) -> dict[str, Any]:
    last = df.iloc[-1]
    support, resistance = support_resistance(df)

    recent_12 = df["close"].tail(12)
    one_hour_change_pct = 0.0
    if len(recent_12) >= 12 and recent_12.iloc[0] != 0:
        one_hour_change_pct = ((recent_12.iloc[-1] - recent_12.iloc[0]) / recent_12.iloc[0]) * 100

    indicator_groups = build_indicator_groups(df)
    market_sentiment = build_market_sentiment(
        book_pressure=book_pressure,
        ticker_stats=ticker_stats,
        fear_greed=fear_greed,
        one_hour_change_pct=one_hour_change_pct,
    )
    health = build_health_score(indicator_groups, market_sentiment)

    final_score = (
        indicator_groups["trend"]["score"]
        + indicator_groups["momentum"]["score"]
        + indicator_groups["volume"]["score"]
        + indicator_groups["volatility"]["score"]
        + market_sentiment["score"]
    )

    reasons = (
        indicator_groups["trend"]["reasons"][:2]
        + indicator_groups["momentum"]["reasons"][:2]
        + indicator_groups["volume"]["reasons"][:1]
        + market_sentiment["reasons"][:2]
    )[:6]

    if final_score >= 2.5:
        side = "BUY"
    elif final_score <= -2.5:
        side = "SELL"
    else:
        side = "NEUTRAL"

    confidence = min(95, max(5, int(50 + abs(final_score) * 8)))

    return {
        "side": side,
        "score": round(final_score, 2),
        "confidence": confidence,
        "support": round(support, 4),
        "resistance": round(resistance, 4),
        "one_hour_change_pct": round(one_hour_change_pct, 2),
        "reasons": reasons,
        "indicator_groups": indicator_groups,
        "market_sentiment": market_sentiment,
        "health": health,
    }


async def make_chart(symbol: str, df: pd.DataFrame) -> BytesIO:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["open_time"], df["close"], label="Close", linewidth=1.8)
    ax.plot(df["open_time"], df["ema9"], label="EMA9", linewidth=1.2)
    ax.plot(df["open_time"], df["ema21"], label="EMA21", linewidth=1.2)
    ax.plot(df["open_time"], df["bb_upper"], label="BB Upper", linestyle="--", linewidth=1)
    ax.plot(df["open_time"], df["bb_lower"], label="BB Lower", linestyle="--", linewidth=1)
    ax.set_title(f"{symbol} - Intraday Price / EMA / Bollinger Bands")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    buffer.seek(0)
    plt.close(fig)
    return buffer


def build_market_message(
    symbol: str,
    df: pd.DataFrame,
    signal: dict[str, Any],
    ticker_stats: dict[str, Any],
    book_pressure: dict[str, float],
    fear_greed: dict[str, Any] | None,
) -> str:
    last = df.iloc[-1]
    price = float(last["close"])
    atr_value = float(last["atr14"]) if pd.notna(last["atr14"]) else np.nan
    side = signal["side"]

    if side in {"BUY", "SELL"} and not np.isnan(atr_value):
        stop_loss, take_profit = compute_trade_levels(price, atr_value, side)
        risk_block = (
            f"🛑 Stop Loss: {stop_loss}\n"
            f"🎯 Take Profit: {take_profit}\n"
            f"📏 ATR(14): {atr_value:.4f}\n"
        )
        summary_sl = stop_loss
        summary_tp = take_profit
    else:
        risk_block = (
            "🛑 Stop Loss: N/A\n"
            "🎯 Take Profit: N/A\n"
            "📏 ATR(14): N/A\n"
        )
        summary_sl = "N/A"
        summary_tp = "N/A"

    indicator_groups = signal["indicator_groups"]
    market_sentiment = signal["market_sentiment"]
    health = signal["health"]

    fg_text = (
        f"{fear_greed['value']} ({fear_greed['classification']})"
        if fear_greed else "N/A"
    )

    reasons_text = "\n".join(f"• {reason}" for reason in signal["reasons"]) or "• No strong reason"
    signal_emoji = "🟢" if side == "BUY" else "🔴" if side == "SELL" else "🟡"

    return (
        f"📊 {symbol}\n"
        f"⏱ Trade Horizon: {TRADE_HORIZON}\n"
        f"💵 Price: {price:.4f}\n"
        f"⚡ EMA9: {last['ema9']:.4f}\n"
        f"⚡ EMA21: {last['ema21']:.4f}\n"
        f"📍 RSI(14): {last['rsi14']:.2f}\n"
        f"📊 MACD: {last['macd']:.4f}\n"
        f"📶 MACD Signal: {last['macd_signal']:.4f}\n"
        f"📦 Volume: {last['volume']:.2f}\n"
        f"📦 Volume MA10: {last['volume_ma10']:.2f}\n"
        f"☁️ Bollinger Upper: {last['bb_upper']:.4f}\n"
        f"☁️ Bollinger Lower: {last['bb_lower']:.4f}\n"
        f"🟢 Support: {signal['support']}\n"
        f"🔴 Resistance: {signal['resistance']}\n"
        f"⚖️ Order Book Buy Pressure: {book_pressure['buy_pressure']:.2%}\n"
        f"🚀 1h Momentum: {signal['one_hour_change_pct']:.2f}%\n"
        f"😶‍🌫️ Fear & Greed: {fg_text}\n"
        f"🌍 Market Sentiment: {market_sentiment['label']} ({market_sentiment['score']})\n"
        f"❤️ Health Score: {health['score']}/100 [{health['status']}]\n\n"
        "📚 Indicator Groups:\n"
        f"• Trend: {indicator_groups['trend']['score']}\n"
        f"• Momentum: {indicator_groups['momentum']['score']}\n"
        f"• Volume: {indicator_groups['volume']['score']}\n"
        f"• Volatility: {indicator_groups['volatility']['score']}\n\n"
        f"{signal_emoji} Signal: {side}\n"
        f"🧠 Score: {signal['score']}\n"
        f"✅ Confidence: {signal['confidence']:.2f}\n"
        f"{risk_block}\n"
        f"✨ Reasons:\n{reasons_text}\n\n"
        "⚠️ Intraday/scalping signal only. The bot's buy/sell signals are not 100% accurate. "
        "Always confirm with your own analysis before entering a trade.\n\n"
        "━━━━━━━━━━━━━━\n"
        f"💵 Price: {price:.4f}\n"
        f"{signal_emoji} Signal: {side}\n"
        f"✅ Confidence: {signal['confidence']:.2f}\n"
        f"❤️ Health: {health['score']}/100\n\n"
        f"💰 Position Size: {POSITION_SIZE}\n"
        f"🛑 SL: {summary_sl}\n"
        f"🎯 TP: {summary_tp}\n"
        f"⏱ Duration: {TRADE_HORIZON}"
    )


async def analyze_symbol(
    symbol: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, float], dict[str, Any] | None]:
    df = await fetch_binance_klines(symbol)
    df = add_indicators(df)
    ticker_stats = await fetch_ticker_stats(symbol)
    book_pressure = await fetch_order_book_pressure(symbol)
    fear_greed = await fetch_fear_and_greed()
    signal = score_signal(df, book_pressure, ticker_stats, fear_greed)
    return df, signal, ticker_stats, book_pressure, fear_greed


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("BTC/USDT", callback_data="BTCUSDT")],
        [InlineKeyboardButton("ETH/USDT", callback_data="ETHUSDT")],
        [InlineKeyboardButton("BNB/USDT", callback_data="BNBUSDT")],
        [InlineKeyboardButton("GOLD", callback_data="GOLD")],
    ]

    if update.message:
        await update.message.reply_text(
            "Trading bot with Binance intraday indicators, grouped signals, market sentiment, health score, risk management, and charting.",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )


async def handle_gold(query) -> None:
    price = await fetch_gold_price()
    signal = "BUY" if price > 2000 else "SELL"
    stop_loss, take_profit = compute_trade_levels(price, 25, signal)
    signal_emoji = "🟢" if signal == "BUY" else "🔴"

    await query.message.reply_text(
        f"🟡 GOLD\n"
        f"⏱ Trade Horizon: {TRADE_HORIZON}\n"
        f"💵 Price: {price:.2f}\n"
        f"{signal_emoji} Signal: {signal}\n"
        f"🌍 Market Sentiment: SIMPLE\n"
        f"❤️ Health Score: 50.00/100 [MIXED]\n"
        f"🛑 Stop Loss: {stop_loss}\n"
        f"🎯 Take Profit: {take_profit}\n\n"
        "⚠️ Gold here uses a simpler external price feed. "
        "This is an intraday indication only and signals are not 100% accurate.\n\n"
        "━━━━━━━━━━━━━━\n"
        f"💵 Price: {price:.2f}\n"
        f"{signal_emoji} Signal: {signal}\n"
        f"✅ Confidence: 50.00\n\n"
        f"💰 Position Size: {POSITION_SIZE}\n"
        f"🛑 SL: {stop_loss}\n"
        f"🎯 TP: {take_profit}\n"
        f"⏱ Duration: {TRADE_HORIZON}"
    )


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    await query.answer()
    data = query.data or ""

    try:
        if data == "GOLD":
            await handle_gold(query)
            return

        if data.startswith("chart_"):
            symbol = data.replace("chart_", "", 1)
            df = add_indicators(await fetch_binance_klines(symbol))
            chart = await make_chart(symbol, df)
            await query.message.reply_photo(photo=chart)
            return

        df, signal, ticker_stats, book_pressure, fear_greed = await analyze_symbol(data)
        message = build_market_message(data, df, signal, ticker_stats, book_pressure, fear_greed)

        keyboard = [
            [InlineKeyboardButton("Live Chart", callback_data=f"chart_{data}")],
            [InlineKeyboardButton("Refresh", callback_data=data)],
        ]

        await query.message.reply_text(
            message,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
    except aiohttp.ClientResponseError as exc:
        logger.exception("HTTP error while handling callback")
        await query.message.reply_text(f"API error: {exc.status} - {exc.message}")
    except Exception as exc:
        logger.exception("Unexpected error while handling callback")
        await query.message.reply_text(f"Unexpected error: {exc}")


def main() -> None:
    if TOKEN == "PUT_YOUR_TELEGRAM_TOKEN_HERE":
        raise ValueError("Please set TELEGRAM_BOT_TOKEN before running the bot.")

    application = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))

    print("Bot running...")
    application.run_polling()


if __name__ == "__main__":
    main()
