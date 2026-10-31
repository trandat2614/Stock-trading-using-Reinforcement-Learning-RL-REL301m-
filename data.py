"""Market data, causal features, configuration and portfolio metrics."""
from __future__ import annotations
import copy
import hashlib
import json
import random
import warnings
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parent
OHLCV=["Open","High","Low","Close","Volume"]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Đọc và kiểm tra cấu hình mà không sửa bản gốc."""
    with open(path or ROOT / "config.yaml", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Từ chối cấu hình không bảo đảm luật giao dịch."""
    assert cfg["SELL_PRICE_MODE"] in {"open", "close", "avg", "high"}
    if cfg.get('FIXED_SELL_PRICE'):
        assert cfg['FIXED_SELL_PRICE']=='high' and cfg.get('DYNAMIC_SELL_PRICES')
        assert cfg.get('BUY_PRICE_MODE','open') in ('open','low')
    if cfg.get('BUY_PRICE_MODE')=='low':assert cfg.get('FIXED_SELL_PRICE')=='high'
    assert len(cfg["BUY_FRACTIONS"]) == len(cfg["SELL_FRACTIONS"]) == 5
    assert all(cfg["BUY_MIN"] <= x <= cfg["BUY_MAX"] for x in cfg["BUY_FRACTIONS"])
    assert all(cfg["SELL_MIN"] <= x <= cfg["SELL_MAX"] for x in cfg["SELL_FRACTIONS"])
    assert max(cfg["BUY_FRACTIONS"]) * (1 + cfg["FEE"]) <= 1
    assert cfg["TRANSACTION_SESSION"] > 0 and cfg["WINDOW"] > 0
    assert cfg["MIN_TRADE_VALUE"] >= 0
    assert pd.Timestamp(cfg["DATA"]["TRAIN_END"]) < pd.Timestamp(cfg["DATA"]["TEST_START"])
    assert cfg["TRAIN"]["ARCHITECTURE"] in {"mlp", "gru", "tcn", "state_mlp"}
    assert 0 <= cfg["TRAIN"].get("BC_CLASS_POWER", 0) <= 1
    assert cfg["TRAIN"].get("BC_TRAJECTORIES", 1) >= 1
    assert cfg["TRAIN"].get("GUIDANCE_INTERVAL", 1) > 0
    assert cfg["TRAIN"].get("BC_TEACHER", "oracle") in {"oracle", "buy_hold", "flexible_expert"}
    if cfg.get("DYNAMIC_SELL_PRICES", False):
        assert cfg["TRAIN"]["PORTFOLIO_PRIOR"] and cfg["SELL_PRICE_MODE"] in {"close", "avg", "high"}
    if cfg["TRAIN"].get("PORTFOLIO_PRIOR", False):
        assert cfg["TRAIN"]["ARCHITECTURE"] == "state_mlp" and cfg["TRAIN"]["ALGORITHM"] == "maskable"
        assert 0 < cfg["TRAIN"]["CASH_PRIOR_SCALE"] <= 1
        assert cfg["TRAIN"]["CASH_PRIOR_STRENGTH"] >= 0
    if cfg["DATA"]["SYMBOL"] != "AMZN":
        raise ValueError("Dự án này dành riêng cho mã AMZN")
    expected = {"INIT_CASH": 100000, "WIN": 1000000, "BUY_MIN": 0.49, "BUY_MAX": 0.83,
                "SELL_MIN": 0.21, "SELL_MAX": 0.69, "FEE": 0.04, "TRANSACTION_SESSION": 19,
                "TRANSACTION_PENALTY": 3751, "TOTAL_ASSETS_FLOOR": 1000, "CASH_FLOOR": -5000}
    if cfg.get("ENFORCE_AMZN_RULES", True):
        for name, value in expected.items():
            if cfg[name] != value:
                raise ValueError(f"Sai tham số AMZN trong đề: {name} phải bằng {value}")


def clone_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Sao chép sâu cấu hình cho từng thí nghiệm."""
    return copy.deepcopy(cfg)


def write_json(path: str | Path, value: Any) -> None:
    """Ghi JSON UTF-8 với giá trị numpy được chuyển an toàn."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default,
                                 allow_nan=False), encoding="utf-8")


def _json_default(value: Any) -> Any:
    """Chuyển kiểu numpy và ngày tháng thành kiểu JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def file_hash(path: str | Path) -> str:
    """Tính SHA256 để kiểm tra dữ liệu và model không bị thay đổi."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_hash(cfg: dict[str, Any]) -> str:
    """Tạo chữ ký cấu hình xác định."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def training_source_hash() -> str:
    """Chặn dùng lại model/cache từ phiên bản thuật toán cũ."""
    names = ("data", "environment", "baseline", "agent")
    return hashlib.sha256("".join(file_hash(ROOT / f"{name}.py") for name in names).encode()).hexdigest()


def seed_everything(seed: int) -> None:
    """Cố định các bộ sinh ngẫu nhiên; hỗ trợ cả máy chưa cài torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def metrics(history: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, float | int]:
    """Tính chỉ số từ sổ lịch sử có một dòng vốn đầu kỳ."""
    assets = history["total_assets"].to_numpy(dtype=float)
    eps = cfg["FEATURES"]["EPSILON"]
    returns = assets[1:] / np.maximum(assets[:-1], eps) - 1
    days = len(returns)
    final = float(assets[-1])
    years = days / cfg["DATA"]["TRADING_DAYS"]
    dd = 1 - assets / np.maximum.accumulate(assets)
    std = float(returns.std(ddof=1)) if days > 1 else 0.0
    steps = history.loc[history["event"] == "step"]
    requested = steps.get("requested_action", steps["action"])
    guard = steps.get("guard_applied", pd.Series(False, index=steps.index)).eq(True)
    notional = steps["shares_delta"].abs() * steps["price"]
    model_trade = steps["executed"].astype(bool) & ~guard
    exposure = (steps["total_assets"] - steps["cash"]) / steps["total_assets"].clip(lower=eps)
    return {
        "final_assets_close": final,
        "final_assets_prev_high": float(history.iloc[-1]["total_assets_prev_high"]),
        "peak_assets": float(assets.max()),
        "score": final / cfg["WIN"],
        "cagr": float(max(final / assets[0], eps) ** (1 / years) - 1) if years else 0.0,
        "sharpe": float(np.sqrt(cfg["DATA"]["TRADING_DAYS"]) * returns.mean() / std) if std else 0.0,
        "max_drawdown": float(dd.max()),
        "trades": int(history["executed"].sum()),
        "penalties": int((history["penalty"] > 0).sum()),
        "total_fees": float(history["fee"].sum()),
        "cumulative_reward": float(history.iloc[-1]["cum_reward"]),
        "sessions": days,
        "score_prev_high": float(history.iloc[-1]["total_assets_prev_high"]) / cfg["WIN"],
        "score_peak": float(assets.max()) / cfg["WIN"],
        "target_reached": bool((assets >= cfg["WIN"]).any()),
        "model_buy_requests": int(requested.between(1, 5).sum()),
        "model_sell_requests": int(requested.between(6, 20 if cfg.get("DYNAMIC_SELL_PRICES", False) else 10).sum()),
        "model_hold_requests": int(requested.eq(0).sum()),
        "guard_overrides": int(guard.sum()),
        "meaningful_model_trades": int((model_trade & (notional >= cfg["INIT_CASH"] * 0.001)).sum()),
        "mean_exposure": float(exposure.mean()) if len(steps) else 0.0,
        "turnover": float(notional.sum() / cfg["INIT_CASH"]),
    }


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    """Đọc CSV thông dụng, xử lý ký hiệu tiền, dấu phẩy và volume K/M/B."""
    frame = pd.read_csv(path)
    frame.columns = [str(x).strip().title().replace("Adj Close", "Adj Close") for x in frame.columns]
    if "Date" not in frame:
        raise ValueError(f"{path}: thiếu Date")
    frame["Date"] = pd.to_datetime(frame["Date"], format="mixed", utc=True).dt.tz_localize(None).dt.normalize()
    frame = frame.set_index("Date").sort_index()
    for col in frame.columns:
        if frame[col].dtype == object:
            text = frame[col].astype(str).str.replace(r"[$,%]", "", regex=True).str.strip()
            multipliers = text.str.extract(r"([KMB])$", expand=False).map({"K": 1e3, "M": 1e6, "B": 1e9}).fillna(1)
            frame[col] = pd.to_numeric(text.str.replace(r"[KMB]$", "", regex=True), errors="coerce") * multipliers
    if required and not set(OHLCV).issubset(frame.columns):
        raise ValueError(f"{path}: cần Date,{','.join(OHLCV)}")
    if "Adj Close" in frame and set(OHLCV).issubset(frame):
        factor = frame["Adj Close"] / frame["Close"]
        frame.loc[:, OHLCV[:4]] = frame[OHLCV[:4]].mul(factor, axis=0)
    return frame


def validate_prices(frame: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Kiểm tra OHLCV, giá điều chỉnh và tính duy nhất theo ngày."""
    if frame.empty or not set(OHLCV).issubset(frame):
        raise ValueError("OHLCV trống hoặc thiếu cột")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Ngày phải tăng dần, không được trùng")
    values = frame[OHLCV].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OHLCV chứa NaN/inf")
    if (frame[OHLCV[:4]] <= 0).any().any() or (frame["Volume"] < 0).any():
        raise ValueError("Giá phải dương, volume không âm")
    if (frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError("High không bao phủ OHLC")
    if (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError("Low không bao phủ OHLC")
    returns = frame["Close"].pct_change()
    jumps = returns[returns.abs() > cfg["DATA"]["RETURN_WARNING"]]
    if not jumps.empty:
        warnings.warn(f"Có |return| lớn: {jumps.to_dict()}; kiểm tra split", stacklevel=2)
    split = returns.loc["2022-06-01":"2022-06-10"]
    if (split < -0.8).any():
        raise ValueError("Phát hiện split 20:1 chưa điều chỉnh quanh 06/2022")


def raw_features(prices: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Tính chỉ báo quá khứ rồi dịch toàn bộ đúng một phiên."""
    fcfg = cfg["FEATURES"]
    eps = fcfg["EPSILON"]
    close = prices["Close"]
    previous = close.shift(1)
    f = pd.DataFrame(index=prices.index)
    for col in ("Open", "High", "Low", "Close"):
        f[f"log_{col.lower()}"] = np.log(prices[col] / prices[col].shift(1))
    f["range"] = (prices["High"] - prices["Low"]) / close
    f["gap"] = np.log(prices["Open"] / previous)
    vol = np.log1p(prices["Volume"])
    f["volume_z"] = (vol - vol.rolling(fcfg["Z_WINDOW"], min_periods=fcfg["Z_MIN_PERIODS"]).mean()) / vol.rolling(fcfg["Z_WINDOW"], min_periods=fcfg["Z_MIN_PERIODS"]).std(ddof=0).clip(lower=eps)
    if fcfg["INCLUDE_RSI"]:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / fcfg["RSI_PERIOD"], adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / fcfg["RSI_PERIOD"], adjust=False).mean()
        f["rsi"] = 1 - 1 / (1 + gain / loss.clip(lower=eps))
    if fcfg["INCLUDE_MACD"]:
        f["macd"] = (close.ewm(span=fcfg["MACD_FAST"], adjust=False).mean() - close.ewm(span=fcfg["MACD_SLOW"], adjust=False).mean()) / close
    center = close.rolling(fcfg["BOLLINGER_PERIOD"]).mean()
    std = close.rolling(fcfg["BOLLINGER_PERIOD"]).std(ddof=0)
    f["bollinger_b"] = (close - center + fcfg["BOLLINGER_STD"] * std) / (2 * fcfg["BOLLINGER_STD"] * std).clip(lower=eps)
    true_range = pd.concat([prices["High"] - prices["Low"], (prices["High"] - previous).abs(), (prices["Low"] - previous).abs()], axis=1).max(axis=1)
    f["atr_close"] = true_range.ewm(alpha=1 / fcfg["ATR_PERIOD"], adjust=False).mean() / close
    if fcfg["INCLUDE_SMA"]:
        for period in fcfg["SMA_PERIODS"]:
            f[f"price_sma_{period}"] = close / close.rolling(period).mean() - 1
    f["volatility"] = np.log(close / previous).rolling(fcfg["VOL_PERIOD"]).std(ddof=0)
    for period in fcfg.get("MOMENTUM_PERIODS", []):
        f[f"momentum_{period}"] = np.log(close / close.shift(period))
    downside_period = fcfg.get("DOWNSIDE_PERIOD")
    if downside_period:
        downside = np.log(close / previous).clip(upper=0)
        f["downside_volatility"] = np.sqrt(downside.pow(2).rolling(downside_period).mean())
    return f.shift(1)


def build_features(prices: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Chuẩn hóa cuộn; không ước lượng thống kê từ tương lai hoặc toàn test."""
    features = raw_features(prices, cfg)
    settings = cfg["FEATURES"]
    rolling = features.rolling(settings["Z_WINDOW"], min_periods=settings["Z_MIN_PERIODS"])
    normalized = (features - rolling.mean()) / rolling.std(ddof=0).clip(lower=settings["EPSILON"])
    normalized = normalized.clip(-settings["CLIP"], settings["CLIP"])
    if settings.get("REGIME_FEATURES", False):
        # raw_features đã shift(1); giữ dấu xu hướng thay vì trừ rolling mean lần nữa.
        scales = {"price_sma_50": 5, "price_sma_200": 5, "momentum_20": 3,
                  "momentum_60": 2, "macd": 20, "rsi": 2, "bollinger_b": 2}
        for name, scale in scales.items():
            center = 0.5 if name in {"rsi", "bollinger_b"} else 0
            normalized[f"regime_{name}"] = ((features[name] - center) * scale).clip(-settings["CLIP"], settings["CLIP"])
    return normalized.astype(np.float32)
