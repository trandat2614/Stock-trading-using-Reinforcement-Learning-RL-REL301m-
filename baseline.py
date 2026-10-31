"""Causal baselines and the teacher oracle restricted to fit data."""
from __future__ import annotations
from typing import Any,Callable
import numpy as np
import pandas as pd
from data import clone_config
from environment import TradingEnv,PriceExecutionEnv,PenaltyGuard,cheapest_action,sell_price


def baseline_action(name: str, env: TradingEnv, rng: np.random.Generator) -> int:
    """Chọn hành động chỉ từ quá khứ và trạng thái danh mục hiện tại."""
    cfg = env.cfg
    history = env.prices.iloc[:env.idx]
    close = float(history.iloc[-1]["Close"])
    if name == "random":
        return int(rng.choice(np.flatnonzero(env.action_masks())))
    if name == "buy_hold":
        if env.steps < cfg["BASELINES"]["BUY_HOLD_INITIAL_DAYS"]:
            return len(cfg["BUY_FRACTIONS"])
        return cheapest_action(env.cash, env.shares, close, cfg) if env.idle == cfg["TRANSACTION_SESSION"] - 1 else 0
    if name == "sma":
        fast, slow = cfg["BASELINES"]["SMA_FAST"], cfg["BASELINES"]["SMA_SLOW"]
        bullish = history["Close"].tail(fast).mean() > history["Close"].tail(slow).mean()
    elif name == "momentum":
        period = cfg["BASELINES"]["MOMENTUM_DAYS"]
        bullish = close > float(history.iloc[-period - 1]["Close"])
    else:
        raise ValueError(name)
    assets = env.cash + env.shares * close
    weight = env.shares * close / max(assets, cfg["FEATURES"]["EPSILON"])
    # Chỉ tái cân bằng khi tín hiệu đổi phía; tránh mua/bán mọi ngày vì phí cao.
    if bullish and weight < cfg["BUY_MIN"] and env.cash > 0:
        return len(cfg["BUY_FRACTIONS"])
    if not bullish and weight > cfg["SELL_MIN"] and env.shares > 0:
        return 5 + len(cfg["SELL_FRACTIONS"])
    return 0


def rollout(env: TradingEnv | PenaltyGuard, policy: Callable[[np.ndarray, TradingEnv], int],
            seed: int = 0) -> pd.DataFrame:
    """Ghi trực tiếp reset/step, luôn giữ snapshot vốn ban đầu."""
    obs, info = env.reset(seed=seed)
    records = [info]
    while True:
        action = policy(obs, env.unwrapped)
        obs, _, terminated, truncated, info = env.step(action)
        records.append(info)
        if terminated or truncated:
            break
    return pd.DataFrame(records)


def run_baseline(name: str, prices: pd.DataFrame, cfg: dict[str, Any], start: str,
                 end: str, seed: int = 0) -> pd.DataFrame:
    """Chạy baseline với cùng vốn, khớp lệnh, phí và phạt như agent."""
    if cfg.get('FIXED_SELL_PRICE')=='high':
        base=PriceExecutionEnv(prices,cfg,start_date=start,end_date=end,eval_mode=True)
    else:
        base = TradingEnv(prices, cfg, start_date=start, end_date=end, eval_mode=True)
    # Mọi baseline đều tránh idle bằng guard, kể cả khi ablation tắt guard của agent.
    env = PenaltyGuard(base)
    rng = np.random.default_rng(seed)
    return rollout(env, lambda obs, state: baseline_action(name, state, rng), seed)


def continuation_factors(prices: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Ước lượng giá trị tiếp diễn biết tương lai, CHỈ cho xếp hạng oracle."""
    n = len(prices)
    cash_factor = np.ones(n + 1)
    stock_factor = np.full(n + 1, float(prices.iloc[-1]["Close"]))
    opening = prices["Open"].to_numpy(dtype=float)
    sale = np.array([sell_price(row, cfg) for _, row in prices.iterrows()])
    for day in range(n - 1, -1, -1):
        cash_factor[day] = max(cash_factor[day + 1], stock_factor[day + 1] / (opening[day] * (1 + cfg["FEE"])))
        stock_factor[day] = max(stock_factor[day + 1], cash_factor[day + 1] * sale[day] * (1 - cfg["FEE"]))
    return cash_factor, stock_factor


def beam_search(prices: pd.DataFrame, cfg: dict[str, Any], width: int | None = None) -> tuple[list[int], float]:
    """Tìm nghiệm khả thi top-K; lưu parent để truy vết, không giả là tối ưu tuyệt đối."""
    width = width or cfg["ORACLE"]["BEAM_WIDTH"]
    cash, shares, idle = np.array([float(cfg["INIT_CASH"])]), np.array([0.0]), np.array([0], dtype=np.int32)
    finished = np.array([False])
    scores = cash.copy()
    parents, actions = [], []
    buy_fractions = np.asarray(cfg["BUY_FRACTIONS"])
    sell_fractions = np.asarray(cfg["SELL_FRACTIONS"])
    cash_factor, stock_factor = continuation_factors(prices, cfg)
    for day, (_, row) in enumerate(prices.iterrows()):
        count = len(cash)
        p = np.repeat(np.arange(count), 11)
        a = np.tile(np.arange(11), count)
        c, s, i = cash[p].copy(), shares[p].copy(), idle[p].copy()
        was_done = finished[p]
        buy = (a >= 1) & (a <= 5) & ~was_done & (c > 0)
        sale = (a >= 6) & ~was_done & (s > 0)
        delta = np.zeros_like(c)
        delta[buy] = buy_fractions[a[buy] - 1] * c[buy] / float(row["Open"])
        delta[sale] = -sell_fractions[a[sale] - 6] * s[sale]
        if not cfg["FRACTIONAL_SHARES"]:
            delta = np.sign(delta) * np.floor(np.abs(delta))
        price = np.where(buy, float(row["Open"]), sell_price(row, cfg))
        value = np.abs(delta) * price
        delta[value < cfg["MIN_TRADE_VALUE"]] = 0
        value = np.abs(delta) * price
        fee = value * cfg["FEE"]
        c -= delta * price + fee
        s += delta
        executed = delta != 0
        i = np.where(was_done, i, np.where(executed, 0, i + 1))
        penalized = (i == cfg["TRANSACTION_SESSION"]) & ~was_done
        c -= penalized * cfg["TRANSACTION_PENALTY"]
        i[penalized] = 0
        score = np.where(was_done, scores[p], c + s * float(row["Close"]))
        loss = (score < cfg["TOTAL_ASSETS_FLOOR"]) | (c <= cfg["CASH_FLOOR"])
        done = was_done | loss | (score >= cfg["WIN"])
        # Loại lệnh không khả thi; Hold luôn tồn tại, terminal chỉ mang tiếp một node.
        valid = ((a == 0) | executed) & (~was_done | (a == 0))
        ids = np.flatnonzero(valid)
        frozen = np.where(done[ids], score[ids], 0.0)
        order = np.lexsort((a[ids], p[ids], frozen, done[ids], i[ids], s[ids], c[ids]))
        ids, frozen = ids[order], frozen[order]
        unique = np.ones(len(ids), dtype=bool)
        unique[1:] = (np.diff(c[ids]) != 0) | (np.diff(s[ids]) != 0) | (np.diff(i[ids]) != 0) | (np.diff(done[ids].astype(int)) != 0) | (np.diff(frozen) != 0)
        ids = ids[unique]
        ranking = np.where(done, score, c * cash_factor[day + 1] + s * stock_factor[day + 1]) if cfg["ORACLE"].get("RANKING", "assets") == "lookahead" else score
        if len(ids) > width:
            threshold = np.partition(ranking[ids], len(ids) - width)[len(ids) - width]
            strong = ids[ranking[ids] > threshold]
            tied = ids[ranking[ids] == threshold]
            tied = tied[np.lexsort((a[tied], p[tied]))[:width - len(strong)]]
            ids = np.concatenate((strong, tied))
        chosen = ids[np.lexsort((a[ids], p[ids], -ranking[ids]))]
        parents.append(p[chosen].astype(np.int32))
        actions.append(np.where(was_done[chosen], -1, a[chosen]).astype(np.int8))
        cash, shares, idle, finished, scores = c[chosen], s[chosen], i[chosen], done[chosen], score[chosen]
    best = int(np.argmax(scores))
    final = float(scores[best])
    sequence = []
    for day in range(len(parents) - 1, -1, -1):
        action = int(actions[day][best])
        if action >= 0:
            sequence.append(action)
        best = int(parents[day][best])
    return sequence[::-1], final


def constrained_oracle(prices: pd.DataFrame, cfg: dict[str, Any], start: str,
                       end: str) -> tuple[pd.DataFrame, list[int]]:
    """Sinh nhãn beam và replay qua TradingEnv để kiểm chứng kế toán."""
    env = TradingEnv(prices, cfg, start_date=start, end_date=end, eval_mode=True)
    subset = prices.iloc[env.first_idx:env.last_idx + 1]
    actions, expected = beam_search(subset, cfg)
    cursor = iter(actions)
    history = rollout(env, lambda obs, base: next(cursor), cfg["SEEDS"][0])
    if not np.isclose(history.iloc[-1]["total_assets"], expected, rtol=1e-9, atol=1e-6):
        raise AssertionError("Beam và TradingEnv không thống nhất luật/kế toán")
    return history, actions
