"""Trading rules, Low/High execution, Guard, asset log and independent audit."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from data import build_features,clone_config


def sell_price(row: pd.Series | dict[str, float], cfg: dict[str, Any]) -> float:
    """Chọn giá khớp bán theo chế độ đã cố định trước thí nghiệm."""
    mode = cfg["SELL_PRICE_MODE"]
    if mode in {"open", "high", "close"}:
        return float(row[mode.title()])
    if mode == "avg":
        return float((row["Open"] + row["High"] + row["Low"] + row["Close"]) / 4)
    raise ValueError(f"Chế độ giá bán không hợp lệ: {mode}")


def quote_trade(cash: float, shares: float, action: int, buy_price: float,
                sale_price: float, cfg: dict[str, Any]) -> tuple[float, float, float]:
    """Tính biến động cash, shares, phí; lệnh nhỏ/không đủ vị thế thành Hold."""
    if action == 0:
        return 0.0, 0.0, 0.0
    if 1 <= action <= len(cfg["BUY_FRACTIONS"]):
        if cash <= 0:
            return 0.0, 0.0, 0.0
        qty = cfg["BUY_FRACTIONS"][action - 1] * cash / buy_price
        if not cfg["FRACTIONAL_SHARES"]:
            qty = float(np.floor(qty))
        value = qty * buy_price
        if qty <= 0 or value < cfg["MIN_TRADE_VALUE"]:
            return 0.0, 0.0, 0.0
        fee = cfg["FEE"] * value
        return -value - fee, qty, fee
    if 6 <= action <= 10:
        if shares <= 0:
            return 0.0, 0.0, 0.0
        qty = cfg["SELL_FRACTIONS"][action - 6] * shares
        if not cfg["FRACTIONAL_SHARES"]:
            qty = float(np.floor(qty))
        value = qty * sale_price
        if qty <= 0 or value < cfg["MIN_TRADE_VALUE"]:
            return 0.0, 0.0, 0.0
        fee = cfg["FEE"] * value
        return value - fee, -qty, fee
    return 0.0, 0.0, 0.0


def cheapest_action(cash: float, shares: float, previous_close: float, cfg: dict[str, Any]) -> int:
    """Tối thiểu hóa phí dự kiến bằng thông tin t−1; hòa chọn mã nhỏ nhất."""
    choices = []
    for action in range(1, 11):
        _, delta, fee = quote_trade(cash, shares, action, previous_close, previous_close, cfg)
        if delta != 0:
            choices.append((fee, action))
    return min(choices)[1] if choices else 0


class TradingEnv(gym.Env[np.ndarray, int]):
    """Mô phỏng giao dịch hợp lệ; quan sát gồm cửa sổ feature và portfolio."""

    metadata = {"render_modes": []}

    def __init__(self, prices: pd.DataFrame, cfg: dict[str, Any], *,
                 features: pd.DataFrame | None = None, start_date: str | None = None,
                 end_date: str | None = None, eval_mode: bool = False) -> None:
        """Giữ ngữ cảnh trước kỳ giao dịch để tính chỉ báo không bị mất warm-up."""
        super().__init__()
        self.cfg = clone_config(cfg)
        self.prices = prices.copy()
        self._rows = self.prices.to_dict("records")
        self._closes = self.prices["Close"].to_numpy(dtype=float)
        self._highs = self.prices["High"].to_numpy(dtype=float)
        self._dates = self.prices.index.strftime("%Y-%m-%d").to_numpy()
        self.features = (features if features is not None else build_features(prices, cfg)).reindex(prices.index)
        self.feature_array = np.ascontiguousarray(self.features.to_numpy(dtype=np.float32))
        self.window = cfg["WINDOW"]
        self.eval_mode = eval_mode
        finite = np.isfinite(self.feature_array).all(axis=1)
        valid = pd.Series(finite).rolling(self.window).sum().eq(self.window).to_numpy()
        eligible = valid.copy()
        if start_date:
            eligible &= prices.index >= pd.Timestamp(start_date)
        if end_date:
            eligible &= prices.index <= pd.Timestamp(end_date)
        indices = np.flatnonzero(eligible)
        if not len(indices):
            raise ValueError("Không đủ warm-up/cửa sổ hoặc không có ngày giao dịch")
        self.first_idx, self.last_idx = int(indices[0]), int(indices[-1])
        if not eligible[self.first_idx:self.last_idx + 1].all():
            raise ValueError("Feature có NaN/inf giữa kỳ giao dịch")
        if self.first_idx < 1:
            raise ValueError("Cần ít nhất một ngày giá trước kỳ giao dịch")
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.window * self.feature_array.shape[1] + 6,), np.float32)
        self._finished = True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Khởi tạo vốn; train chọn đoạn ngẫu nhiên, eval chạy tuần tự toàn kỳ."""
        super().reset(seed=seed)
        self.action_space.seed(seed)
        options = options or {}
        available = self.last_idx - self.first_idx + 1
        default_length = available if self.eval_mode else min(self.cfg["EPISODE_LEN"], available)
        self.length = int(options.get("length", default_length))
        if not 1 <= self.length <= available:
            raise ValueError("length vượt dữ liệu cho phép")
        latest = self.last_idx - self.length + 1
        default_start = self.first_idx if self.eval_mode else int(self.np_random.integers(self.first_idx, latest + 1))
        self.start_idx = int(options.get("start_idx", default_start))
        if not self.first_idx <= self.start_idx <= latest:
            raise ValueError("start_idx vượt biên kỳ đã chọn")
        self.idx = self.start_idx
        self.end_idx = self.start_idx + self.length - 1
        self.last_valued_idx = self.start_idx - 1
        self.cash = float(self.cfg["INIT_CASH"])
        self.shares = 0.0
        self.cost_basis = 0.0
        self.idle = 0
        self.steps = 0
        self.value = self.cash
        self.peak = self.value
        self.drawdown = 0.0
        self.cum_reward = 0.0
        self.bh_cash, self.bh_shares, self.bh_idle, self.bh_value = self.cash, 0.0, 0, self.cash
        self._finished = False
        info = self._info(0, 0.0, 0.0, 0.0, 0.0, False, False, "initial")
        info["date"] = self._dates[self.start_idx]
        return self._observation(), info

    def _observation(self) -> np.ndarray:
        """Portfolio dùng giá cuối phiên đã biết, không dùng OHLC phiên sắp tới."""
        idx = min(self.idx, self.last_idx)
        window = self.feature_array[idx - self.window + 1:idx + 1].reshape(-1)
        previous_close = float(self._closes[self.idx - 1])
        value = self.cash + self.shares * previous_close
        eps = self.cfg["FEATURES"]["EPSILON"]
        denominator = max(abs(value), eps)
        unrealized = previous_close / self.cost_basis - 1 if self.cost_basis > 0 and self.shares > 0 else 0.0
        portfolio = np.array([self.cash / denominator, self.shares * previous_close / denominator,
                              np.log(max(value, eps) / self.cfg["INIT_CASH"]), unrealized,
                              self.idle / self.cfg["TRANSACTION_SESSION"], self.steps / self.length], np.float32)
        return np.concatenate((window, portfolio)).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        """Mask chỉ dựa trên trạng thái đã biết và giá t−1, luôn cho phép Hold."""
        if self.cfg["FRACTIONAL_SHARES"] and self.cfg["MIN_TRADE_VALUE"] == 0:
            return np.array([True] + [self.cash > 0] * 5 + [self.shares > 0] * 5, dtype=bool)
        price = float(self._closes[min(self.idx - 1, self.last_idx)])
        mask = np.ones(self.action_space.n, dtype=bool)
        for action in range(1, self.action_space.n):
            _, delta, _ = quote_trade(self.cash, self.shares, action, price, price, self.cfg)
            mask[action] = delta != 0
        return mask

    def total_assets_prev_high(self) -> float:
        """Định giá tại High của phiên trước phiên định giá cuối cùng."""
        previous = max(self.last_valued_idx - 1, 0)
        return float(self.cash + self.shares * self._highs[previous])

    def _benchmark_step(self, row: dict[str, float]) -> float:
        """Cập nhật B&H hợp lệ cùng kỳ dùng làm mốc excess reward."""
        if self.steps < self.cfg["BASELINES"]["BUY_HOLD_INITIAL_DAYS"]:
            action = len(self.cfg["BUY_FRACTIONS"])
        elif self.bh_idle == self.cfg["TRANSACTION_SESSION"] - 1:
            action = cheapest_action(self.bh_cash, self.bh_shares, float(self._closes[self.idx - 1]), self.cfg)
        else:
            action = 0
        dcash, dshares, _ = quote_trade(self.bh_cash, self.bh_shares, action, float(row["Open"]), sell_price(row, self.cfg), self.cfg)
        self.bh_cash += dcash
        self.bh_shares += dshares
        self.bh_idle = 0 if dshares else self.bh_idle + 1
        if self.bh_idle == self.cfg["TRANSACTION_SESSION"]:
            self.bh_cash -= self.cfg["TRANSACTION_PENALTY"]
            self.bh_idle = 0
        result = self.bh_cash + self.bh_shares * float(row["Close"])
        return float(result)

    def _info(self, action: int, price: float, delta: float, fee: float, penalty: float,
              invalid: bool, executed: bool, event: str = "step") -> dict[str, Any]:
        """Tạo bản ghi nguyên bản dùng trực tiếp cho sổ lịch sử."""
        return {"date": self._dates[self.last_valued_idx], "event": event,
                "action": action, "price": price, "shares_delta": delta, "fee": fee,
                "penalty_applied": bool(penalty), "penalty": penalty, "cash": self.cash,
                "shares": self.shares, "total_assets": self.value,
                "total_assets_prev_high": self.total_assets_prev_high(), "drawdown": self.drawdown,
                "idle": self.idle, "reward": 0.0, "cum_reward": self.cum_reward,
                "invalid_action": invalid, "executed": executed, "outcome": "ongoing",
                "initial_assets": self.cfg["INIT_CASH"]}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Khớp → idle/phạt → định giá → reward → điều kiện dừng."""
        if self._finished:
            raise RuntimeError("Cần reset trước khi step sau khi episode đã kết thúc")
        action = int(action)
        row = self._rows[self.idx]
        previous_value = self.value
        old_drawdown = self.drawdown
        invalid = not self.action_space.contains(action) or (1 <= action <= 5 and self.cash <= 0) or (6 <= action <= 10 and self.shares <= 0)
        dcash, delta, fee = quote_trade(self.cash, self.shares, action, float(row["Open"]), sell_price(row, self.cfg), self.cfg) if not invalid else (0.0, 0.0, 0.0)
        if delta > 0:
            self.cost_basis = (self.cost_basis * self.shares + delta * float(row["Open"])) / (self.shares + delta)
        self.cash += dcash
        self.shares += delta
        executed = delta != 0
        self.idle = 0 if executed else self.idle + 1
        penalty = 0.0
        if self.idle == self.cfg["TRANSACTION_SESSION"]:
            penalty = float(self.cfg["TRANSACTION_PENALTY"])
            self.cash -= penalty
            self.idle = 0
        self.value = float(self.cash + self.shares * float(row["Close"]))
        self.peak = max(self.peak, self.value)
        self.drawdown = 1 - self.value / self.peak
        eps = self.cfg["FEATURES"]["EPSILON"]
        log_return = np.log(max(self.value, eps) / max(previous_value, eps))
        benchmark_value = self._benchmark_step(row)
        bh_log_return = np.log(max(benchmark_value, eps) / max(self.bh_value, eps))
        self.bh_value = benchmark_value
        r = self.cfg["REWARD"]
        reward = r["LOG_SCALE"] * log_return + r["EXCESS"] * (log_return - bh_log_return) - r["DRAWDOWN"] * max(0.0, self.drawdown - old_drawdown) - r["TRADE"] * executed - r["INVALID"] * invalid
        loss = self.value < self.cfg["TOTAL_ASSETS_FLOOR"] or self.cash <= self.cfg["CASH_FLOOR"]
        win = not loss and self.value >= self.cfg["WIN"]
        reward += r["LOSS"] if loss else r["WIN"] if win else 0.0
        self.steps += 1
        terminated = bool(loss or win)
        truncated = bool(not terminated and (self.steps >= self.length or self.idx >= self.end_idx))
        self.cum_reward += float(reward)
        self.last_valued_idx = self.idx
        price = float(row["Open"]) if delta > 0 else sell_price(row, self.cfg) if delta < 0 else 0.0
        info = self._info(action, price, delta, fee, penalty, invalid, executed)
        info.update(reward=float(reward), cum_reward=self.cum_reward, outcome="loss" if loss else "win" if win else "ongoing",
                    terminated=terminated, truncated=truncated,
                    valuation_prev_high_date=self._dates[max(self.last_valued_idx - 1, 0)],
                    valuation_prev_high_price=float(self._highs[max(self.last_valued_idx - 1, 0)]))
        self.idx += 1
        self._finished = terminated or truncated
        return self._observation(), float(reward), terminated, truncated, info


SELL_MODES = ("close", "avg", "high")


def decode_action(action: int) -> tuple[int, str | None]:
    if 0 <= action <= 5:
        return action, None
    if 6 <= action <= 20:
        return 6 + (action - 6) % 5, SELL_MODES[(action - 6) // 5]
    return action, None


class FlexibleTradingEnv(TradingEnv):
    """Discrete(21): 0 Hold, 1..5 Buy, 6..10 Close, 11..15 OHLC4, 16..20 High."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Discrete(21)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(**kwargs)
        info.update(base_action=0, sell_price_mode="none", requested_action=0, guard_applied=False)
        return obs, info

    def action_masks(self) -> np.ndarray:
        ordinary = super().action_masks()[:11]
        return np.concatenate((ordinary[:6], np.tile(ordinary[6:11], 3)))

    def _benchmark_step(self, row: dict[str, float]) -> float:
        original = self.cfg["SELL_PRICE_MODE"]
        self.cfg["SELL_PRICE_MODE"] = self.cfg.get("BENCHMARK_SELL_PRICE_MODE", "high")
        try:
            return super()._benchmark_step(row)
        finally:
            self.cfg["SELL_PRICE_MODE"] = original

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        base_action, mode = decode_action(action)
        original = self.cfg["SELL_PRICE_MODE"]
        if mode is not None:
            self.cfg["SELL_PRICE_MODE"] = mode
        try:
            result = super().step(base_action)
        finally:
            self.cfg["SELL_PRICE_MODE"] = original
        obs, reward, terminated, truncated, info = result
        info.update(action=action, base_action=base_action,
                    sell_price_mode=mode if mode is not None else "none")
        return obs, reward, terminated, truncated, info


class PriceExecutionEnv(FlexibleTradingEnv):
    """Use the original accounting engine with a separate execution-row view."""

    def __init__(self,*args:Any,**kwargs:Any)->None:
        super().__init__(*args,**kwargs)
        mode=self.cfg.get('BUY_PRICE_MODE','open')
        if mode not in ('open','low') or self.cfg.get('FIXED_SELL_PRICE')!='high':
            raise ValueError('Execution profile requires Open/Low purchases and High sales')
        self._market_rows=self._rows
        if mode=='low':
            # Only the accounting rows change. prices/features/observations retain actual OHLC.
            self._rows=[dict(row,Open=row['Low']) for row in self._market_rows]

    def action_masks(self)->np.ndarray:
        mask=super().action_masks();mask[6:16]=False
        return mask

    def reset(self,**kwargs:Any):
        obs,info=super().reset(**kwargs)
        info.update(buy_price_mode=self.cfg.get('BUY_PRICE_MODE','open'),execution_price_mapping=False)
        return obs,info

    def step(self,action:int):
        original=int(action)
        # Canonical Guard/baseline Sell actions use the same fraction at High.
        if 6<=original<=20:action=16+(original-6)%5
        obs,reward,terminated,truncated,info=super().step(action)
        row=self._market_rows[self.last_valued_idx]
        info.update(buy_price_mode=self.cfg.get('BUY_PRICE_MODE','open'),
                    market_open=row['Open'],market_low=row['Low'],market_high=row['High'],
                    execution_price_mapping=int(action)!=original)
        return obs,reward,terminated,truncated,info


class PenaltyGuard(gym.Wrapper):
    """Thay Hold ở phiên sắp bị phạt bằng lệnh có phí dự kiến thấp nhất."""

    def action_masks(self) -> np.ndarray:
        """Chuyển tiếp mask cho MaskablePPO và VecEnv."""
        return self.env.unwrapped.action_masks()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Ghi cả lệnh agent đề nghị và lệnh thực thi để audit."""
        base: TradingEnv = self.env.unwrapped
        requested = int(action)
        if requested == 0 and base.idle == base.cfg["TRANSACTION_SESSION"] - 1:
            price = float(base._closes[base.idx - 1])
            action = cheapest_action(base.cash, base.shares, price, base.cfg)
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["requested_action"] = requested
        info["guard_applied"] = int(action) != requested
        return obs, reward, terminated, truncated, info


def action_masks(env: gym.Env) -> np.ndarray:
    """Trả mask qua các wrapper Gymnasium."""
    return env.unwrapped.action_masks()


class TrainingPeakLedger(gym.Wrapper):
    FIELDS=('rank','episode','local_transition','global_transition','episode_step','date','event',
            'total_assets','cash','shares','action','requested_action','guard_applied','base_action',
            'buy_price_mode','sell_price_mode','price','fee','penalty','reward','cum_reward','new_rank_peak',
            'stage','seed','close','low','high','asset_change','close_change_pct','new_episode_peak',
            'drawdown','shares_delta','terminated','truncated','idle','outcome')

    def __init__(self,env:gym.Env,path:Path,rank:int,n_envs:int,*,all_transitions:bool=False,
                 seed:int=0,stage:str='optimization')->None:
        super().__init__(env)
        path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=path.open('w',encoding='utf-8',newline='')
        self.writer=csv.DictWriter(self.handle,fieldnames=self.FIELDS,extrasaction='ignore')
        self.writer.writeheader()
        self.rank,self.n_envs=rank,n_envs
        self.all_transitions,self.seed,self.stage=all_transitions,seed,stage
        self.previous_value=0.0
        self.episode,self.transitions=0,0
        self.episode_peak,self.rank_peak=-float('inf'),-float('inf')

    def _record(self,info:dict[str,Any],initial:bool=False)->None:
        value=float(info['total_assets'])
        new_episode=value>self.episode_peak
        if initial or new_episode or self.all_transitions:
            new=value>self.rank_peak
            self.episode_peak=max(value,self.episode_peak);self.rank_peak=max(value,self.rank_peak)
            row={**info,'rank':self.rank,'episode':self.episode,'local_transition':self.transitions,
                 'global_transition':0 if initial else (self.transitions-1)*self.n_envs+self.rank+1,
                 'episode_step':self.env.unwrapped.steps,'new_rank_peak':new,
                 'stage':self.stage,'seed':self.seed,'asset_change':0.0 if initial else value-self.previous_value,
                 'new_episode_peak':new_episode}
            base=self.env.unwrapped
            if hasattr(base,'prices'):
                index=base.idx-1 if initial else base.last_valued_idx
                mark=base.prices.iloc[index]
                previous=base.prices.iloc[max(0,index-1)]
                row.update(close=float(mark.Close),low=float(mark.Low),high=float(mark.High),
                           close_change_pct=float(mark.Close/previous.Close-1)*100)
            self.writer.writerow(row);self.handle.flush()
        self.previous_value=value

    def reset(self,**kwargs:Any):
        obs,info=self.env.reset(**kwargs)
        self.episode+=1;self.episode_peak=-float('inf');self._record(info,True)
        return obs,info

    def step(self,action:int):
        result=self.env.step(action);self.transitions+=1;self._record(result[-1]);return result

    def close(self)->None:
        if not self.handle.closed:self.handle.close()
        self.env.close()


def audit_flexible(history: pd.DataFrame, market: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    def close(actual: float, expected: float) -> None:
        if not np.isclose(actual, expected, rtol=1e-9, atol=1e-6):
            raise AssertionError(f"Ledger mismatch: {actual} != {expected}")

    cash, shares, idle = float(cfg["INIT_CASH"]), 0.0, 0
    previous_value, peak, old_dd, cumulative = cash, cash, 0.0, 0.0
    first = history.iloc[0]
    assert first["event"] == "initial" and first["cash"] == cash and first["shares"] == 0
    close(first["total_assets"], cash)
    steps = history.loc[history["event"] == "step"]
    dates = market.loc[cfg["DATA"]["TEST_START"]:cfg["DATA"]["TEST_END"]].index
    assert steps["date"].tolist() == [str(date.date()) for date in dates[:len(steps)]]
    sales = {mode: 0 for mode in ("close", "avg", "high")}
    for row in steps.to_dict("records"):
        date = pd.Timestamp(row["date"])
        price_row = market.loc[date]
        previous = market.iloc[market.index.get_loc(date) - 1]
        action, requested = int(row["action"]), int(row["requested_action"])
        if 0 <= action <= 5:
            base_action, mode = action, "none"
        elif 6 <= action <= 20:
            base_action, mode = 6 + (action - 6) % 5, ("close", "avg", "high")[(action - 6) // 5]
        else:
            base_action, mode = action, "none"
        assert int(row["base_action"]) == base_action and row["sell_price_mode"] == mode
        assert bool(row["guard_applied"]) == (action != requested)
        if action != requested:
            assert requested == 0 and idle == cfg["TRANSACTION_SESSION"] - 1
            options = []
            for number, fraction in enumerate(cfg["BUY_FRACTIONS"], 1):
                if cash > 0 and fraction * cash / previous["Close"] > 0:
                    options.append((fraction * cash * cfg["FEE"], number))
            for number, fraction in enumerate(cfg["SELL_FRACTIONS"], 6):
                if shares > 0 and fraction * shares > 0:
                    options.append((fraction * shares * previous["Close"] * cfg["FEE"], number))
            expected=min(options)[1] if options else 0
            if cfg.get('FIXED_SELL_PRICE')=='high' and 6<=expected<=10:expected+=10
            assert action == expected
        invalid = not 0 <= action <= 20 or (1 <= action <= 5 and cash <= 0) or (6 <= action <= 20 and shares <= 0)
        qty, price = 0.0, 0.0
        if 1 <= action <= 5 and cash > 0:
            buy_key='Low' if cfg.get('BUY_PRICE_MODE')=='low' else 'Open'
            qty = cfg["BUY_FRACTIONS"][action - 1] * cash / price_row[buy_key]
            price = float(price_row[buy_key]) if qty else 0.0
        elif 6 <= action <= 20 and shares > 0:
            if cfg.get('FIXED_SELL_PRICE')=='high':assert mode=='high'
            qty = -cfg["SELL_FRACTIONS"][(action - 6) % 5] * shares
            price = float(price_row[["Open", "High", "Low", "Close"]].mean()) if mode == "avg" else float(price_row[mode.title()])
            if qty:
                sales[mode] += 1
            else:
                price = 0.0
        executed = qty != 0
        fee = abs(qty) * price * cfg["FEE"]
        if executed:
            cash -= qty * price + fee
            shares += qty
            idle = 0
        else:
            idle += 1
        penalty = cfg["TRANSACTION_PENALTY"] if idle == cfg["TRANSACTION_SESSION"] else 0
        if penalty:
            cash -= penalty
            idle = 0
        value = cash + shares * price_row["Close"]
        peak = max(peak, value)
        dd = 1 - value / peak
        eps, reward_cfg = cfg["FEATURES"]["EPSILON"], cfg["REWARD"]
        assert reward_cfg["EXCESS"] == 0
        reward = reward_cfg["LOG_SCALE"] * np.log(max(value, eps) / max(previous_value, eps)) - reward_cfg["DRAWDOWN"] * max(0, dd - old_dd) - reward_cfg["TRADE"] * executed - reward_cfg["INVALID"] * invalid
        loss = value < cfg["TOTAL_ASSETS_FLOOR"] or cash <= cfg["CASH_FLOOR"]
        win = not loss and value >= cfg["WIN"]
        reward += reward_cfg["LOSS"] if loss else reward_cfg["WIN"] if win else 0
        cumulative += reward
        for name, expected in (("cash", cash), ("shares", shares), ("shares_delta", qty), ("price", price),
                               ("fee", fee), ("penalty", penalty), ("total_assets", value), ("reward", reward),
                               ("cum_reward", cumulative), ("total_assets_prev_high", cash + shares * previous["High"]),
                               ("valuation_prev_high_price", previous["High"])):
            close(row[name], expected)
        assert int(row["idle"]) == idle and bool(row["executed"]) == executed and bool(row["invalid_action"]) == invalid
        assert bool(row["terminated"]) == (loss or win)
        assert row["outcome"] == ("loss" if loss else "win" if win else "ongoing")
        if loss or win:
            assert date == pd.Timestamp(steps.iloc[-1]["date"])
        previous_value, old_dd = value, dd
    assert len(steps) == len(dates) or steps.iloc[-1]["outcome"] in {"win", "loss"}
    return {"status": "passed", "sessions": len(steps), "final_assets": previous_value, "executed_sells": sales,
            "checks": "21 actions, modes, fractions, causal Guard, fees, penalties, cash/shares, previous High, rewards, done"}


def environment_class(cfg):
    if cfg.get('BUY_PRICE_MODE') not in ('open','low') or cfg.get('FIXED_SELL_PRICE')!='high':
        raise ValueError('This project requires fixed High sales')
    return PriceExecutionEnv
