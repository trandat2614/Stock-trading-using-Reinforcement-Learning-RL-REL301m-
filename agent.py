"""Temporal-state PPO, balanced imitation and validation checkpoint selection.

GuidedMaskablePPO adapts sb3-contrib 2.7.1 (MIT).
Copyright (c) 2020 Stable-Baselines Team. See THIRD_PARTY_NOTICES.md."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor,MlpExtractor
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.utils import explained_variance
from data import ROOT,build_features,clone_config,config_hash,file_hash,metrics,seed_everything,write_json,training_source_hash
from environment import TradingEnv,FlexibleTradingEnv,PriceExecutionEnv,PenaltyGuard,TrainingPeakLedger,environment_class
from baseline import rollout,run_baseline,baseline_action,constrained_oracle


class MarketStateExtractor(BaseFeaturesExtractor):
    """Mã hóa trạng thái thị trường gọn, giữ portfolio ở một nhánh trực tiếp."""

    def __init__(self, observation_space: Any, window: int, hidden_size: int = 64) -> None:
        super().__init__(observation_space, hidden_size + 6)
        self.window = window
        market_dim = (observation_space.shape[0] - 6) // window
        self.encoder = torch.nn.Sequential(torch.nn.Linear(market_dim * 5, hidden_size), torch.nn.Tanh(),
                                           torch.nn.Linear(hidden_size, hidden_size), torch.nn.Tanh())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        market = observations[:, :-6].reshape(-1, self.window, (observations.shape[-1] - 6) // self.window)
        # Keep chronology: recent mean and direction distinguish windows whose
        # last value, overall mean and variance are otherwise identical.
        recent=market[:, -min(5,self.window):].mean(dim=1)
        direction=market[:, -1]-market[:, 0]
        summary = torch.cat((market[:, -1], market.mean(dim=1), market.std(dim=1, unbiased=False),
                             recent,direction), dim=1)
        return torch.cat((self.encoder(summary), observations[:, -6:]), dim=1)


class PortfolioVecNormalize(VecNormalize):
    """Chuẩn hóa thị trường, giữ sáu biến danh mục ở đơn vị gốc đã chuẩn hóa tỷ lệ."""

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        result = super().normalize_obs(obs)
        result[..., -6:] = obs[..., -6:]
        return result

    def unnormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        result = super().unnormalize_obs(obs)
        result[..., -6:] = obs[..., -6:]
        return result


class PortfolioMlpExtractor(nn.Module):
    """Đưa trực tiếp portfolio vào actor, đồng thời giữ nhánh value thông thường."""

    def __init__(self, feature_dim: int, net_arch: Any, activation_fn: type[nn.Module], device: Any) -> None:
        super().__init__()
        self.base = MlpExtractor(feature_dim, net_arch, activation_fn, device)
        self.latent_dim_pi = self.base.latent_dim_pi + 6
        self.latent_dim_vf = self.base.latent_dim_vf

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.base.forward_actor(features), features[..., -6:]), dim=-1)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.base.forward_critic(features)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class PortfolioPriorPolicy(MaskableActorCriticPolicy):
    """PPO học residual logits trên prior mềm tránh mua lặp khi gần hết cash.

    Mọi hành động hợp lệ vẫn có xác suất dương. Không đặt minimum trade,
    không bỏ phí/phạt và không quyết định thay model khi predict.
    """

    def __init__(self, *args: Any, cash_prior_scale: float = 0.02,
                 cash_prior_strength: float = 1.5, cost_prior_strength: float = 0.0,
                 transaction_fee: float = 0.04,
                 buy_fractions: tuple[float, ...] = (0.49, 0.58, 0.66, 0.75, 0.83),
                 sell_fractions: tuple[float, ...] = (0.21, 0.33, 0.45, 0.57, 0.69), **kwargs: Any) -> None:
        if not 0 < cash_prior_scale <= 1 or cash_prior_strength < 0:
            raise ValueError("Prior scale phải thuộc (0,1], strength >= 0")
        self.cash_prior_scale, self.cash_prior_strength = cash_prior_scale, cash_prior_strength
        if cost_prior_strength < 0:
            raise ValueError("Cost prior strength phải >= 0")
        self.cost_prior_strength, self.transaction_fee = cost_prior_strength, transaction_fee
        self.buy_fractions, self.sell_fractions = tuple(buy_fractions), tuple(sell_fractions)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PortfolioMlpExtractor(self.features_dim, self.net_arch, self.activation_fn, self.device)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Any:
        logits = self.action_net(latent_pi)
        cash_ratio = latent_pi[..., -6]
        # Chỉ phụ thuộc cash/V đã biết ở t-1. Clamp giữ prior hữu hạn.
        penalty = self.cash_prior_strength * torch.log((cash_ratio / self.cash_prior_scale).clamp(1e-12, 1.0))
        prior = torch.zeros_like(logits)
        prior[..., 1:6] = penalty.unsqueeze(-1)
        if self.cost_prior_strength:
            buy = torch.as_tensor(self.buy_fractions, device=logits.device, dtype=logits.dtype)
            sell = torch.as_tensor(self.sell_fractions, device=logits.device, dtype=logits.dtype)
            coefficient = self.cost_prior_strength * self.transaction_fee
            prior[..., 1:6] -= coefficient * cash_ratio.clamp(min=0).unsqueeze(-1) * buy
            prior[..., 6:11] -= coefficient * latent_pi[..., -5].clamp(min=0).unsqueeze(-1) * sell
        return self.action_dist.proba_distribution(action_logits=logits + prior)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        result = super()._get_constructor_parameters()
        result.update(cash_prior_scale=self.cash_prior_scale, cash_prior_strength=self.cash_prior_strength)
        result.update(cost_prior_strength=self.cost_prior_strength, transaction_fee=self.transaction_fee,
                      buy_fractions=self.buy_fractions, sell_fractions=self.sell_fractions)
        return result


class FactorizedActionHead(nn.Module):
    def __init__(self, latent_dim: int, fixed_sell_price: str | None = None) -> None:
        super().__init__()
        self.fixed_sell_price=fixed_sell_price
        self.trade_head = nn.Linear(latent_dim, 11)
        self.price_head = nn.Linear(latent_dim, 3)
        for head in (self.trade_head, self.price_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    @property
    def bias(self) -> torch.Tensor:
        return self.trade_head.bias

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        trade = self.trade_head(latent)
        if self.fixed_sell_price=='high':
            price=trade.new_full((*trade.shape[:-1],3),-1e8)
            price[...,2]=0
        else:
            price = torch.log_softmax(self.price_head(latent), dim=-1)
        sell = (price.unsqueeze(-1) + trade[..., 6:11].unsqueeze(-2)).flatten(-2)
        return torch.cat((trade[..., :6], sell), dim=-1)


class FlexiblePortfolioPolicy(PortfolioPriorPolicy):
    def __init__(self,*args:Any,fixed_sell_price:str|None=None,
                 risk_hold_penalty:float=0.0,risk_sell_boost:float=0.0,**kwargs:Any)->None:
        if fixed_sell_price not in (None,'high'):raise ValueError('Unsupported fixed sell price')
        self.fixed_sell_price=fixed_sell_price
        self.risk_hold_penalty=float(risk_hold_penalty)
        self.risk_sell_boost=float(risk_sell_boost)
        super().__init__(*args,**kwargs)

    def _get_constructor_parameters(self)->dict[str,Any]:
        result=super()._get_constructor_parameters()
        result['fixed_sell_price']=self.fixed_sell_price
        result.update(risk_hold_penalty=self.risk_hold_penalty,risk_sell_boost=self.risk_sell_boost)
        return result

    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        if self.action_space.n != 21:
            raise ValueError("Flexible policy cần Discrete(21)")
        self.action_net = FactorizedActionHead(self.mlp_extractor.latent_dim_pi,self.fixed_sell_price).to(self.device)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Any:
        logits = self.action_net(latent_pi)
        cash, stock = latent_pi[..., -6], latent_pi[..., -5]
        prior = torch.zeros_like(logits)
        # A soft preference only: HOLD remains legal and can win after learning.
        # Portfolio ratios are from the previous completed session.
        prior[..., 0] -= self.risk_hold_penalty
        prior[..., 6:21] += self.risk_sell_boost * stock.clamp(0,1).unsqueeze(-1)
        prior[..., 1:6] = (self.cash_prior_strength * torch.log((cash / self.cash_prior_scale).clamp(1e-12, 1))).unsqueeze(-1)
        buys = torch.as_tensor(self.buy_fractions, device=logits.device, dtype=logits.dtype)
        sells = torch.as_tensor(self.sell_fractions * 3, device=logits.device, dtype=logits.dtype)
        coefficient = self.cost_prior_strength * self.transaction_fee
        prior[..., 1:6] -= coefficient * cash.clamp(min=0).unsqueeze(-1) * buys
        prior[..., 6:21] -= coefficient * stock.clamp(min=0).unsqueeze(-1) * sells
        return self.action_dist.proba_distribution(action_logits=logits + prior)


def action_class_weights(actions: np.ndarray, action_count: int, power: float = 0.0) -> np.ndarray:
    """Giảm lệch lớp BC; power=0 giữ nguyên cách học cũ, không đổi luật action."""
    if not 0 <= power <= 1:
        raise ValueError("BC_CLASS_POWER phải nằm trong [0,1]")
    counts = np.bincount(actions, minlength=action_count)
    weights = np.ones(action_count, dtype=np.float32)
    present = counts > 0
    weights[present] = np.clip((len(actions) / (present.sum() * counts[present])) ** power, 0.25, 4.0)
    return weights


def expert_sampling_probabilities(actions: np.ndarray, power: float) -> np.ndarray:
    """Mix natural sampling with equal HOLD/BUY/SELL groups, using existing BC power."""
    actions=np.asarray(actions,dtype=int)
    if not len(actions) or not 0<=power<=1:raise ValueError('Invalid expert sampling inputs')
    groups=np.where(actions==0,0,np.where(actions<=5,1,2))
    counts=np.bincount(groups,minlength=3)
    balanced=1.0/(np.count_nonzero(counts)*counts[groups])
    probabilities=(1-power)/len(actions)+power*balanced
    return probabilities/probabilities.sum()


class GuidedMaskablePPO(MaskablePPO):
    """Tối ưu cùng lúc PPO và imitation, có clip gradient và chặn KL quá lớn."""

    def __init__(self, *args: Any, guidance_start: float = 0.0, guidance_end: float = 0.0,
                 guidance_batch_size: int = 128, guidance_interval: int = 4,
                 sampling_power: float = 0.0, **kwargs: Any) -> None:
        """Không gắn expert tự động khi load: inference chỉ dùng trọng số policy."""
        self.guidance_start, self.guidance_end = guidance_start, guidance_end
        self.guidance_batch_size, self.guidance_interval = guidance_batch_size, guidance_interval
        self._sampling_power=float(sampling_power)
        self.expert_observations: np.ndarray | None = None
        self.expert_actions: np.ndarray | None = None
        self.expert_masks: np.ndarray | None = None
        self.expert_normalizer: Any = None
        self.expert_class_weights: np.ndarray | None = None
        self._expert_rng = np.random.default_rng(kwargs.get("seed"))
        super().__init__(*args, **kwargs)

    def set_expert(self, observations: np.ndarray, actions: np.ndarray, masks: np.ndarray,
                   normalizer: Any, class_weights: np.ndarray | None = None) -> None:
        """Nhận quan sát THÔ và nhãn oracle từ phần fit do train.py kiểm soát."""
        if not len(actions) or len(observations) != len(actions) or len(masks) != len(actions):
            raise ValueError("Expert phải có quan sát, nhãn và mask cùng số dòng")
        if not masks[np.arange(len(actions)), actions].all():
            raise ValueError("Expert chứa hành động không hợp lệ")
        self.expert_observations = observations.astype(np.float32, copy=True)
        self.expert_actions = actions.astype(np.int64, copy=True)
        self.expert_masks = masks.astype(bool, copy=True)
        self.expert_normalizer = normalizer
        self.expert_class_weights = class_weights
        self._expert_sampling=expert_sampling_probabilities(actions,self._sampling_power)

    def _excluded_save_params(self) -> list[str]:
        """Không nhúng dữ liệu oracle hoặc env train vào model dùng để predict."""
        return super()._excluded_save_params() + ["expert_observations", "expert_actions", "expert_masks", "expert_normalizer", "expert_class_weights", "_expert_rng", "_expert_sampling"]

    def _imitation_loss(self) -> torch.Tensor:
        """Lấy batch expert và chuẩn hóa bằng obs_rms train hiện tại, không cập nhật rms."""
        if self.expert_observations is None:
            return torch.zeros((), device=self.device)
        ids = self._expert_rng.choice(len(self.expert_actions),self.guidance_batch_size,replace=True,p=self._expert_sampling)
        normalized = self.expert_normalizer.normalize_obs(self.expert_observations[ids])
        observations = torch.as_tensor(normalized, device=self.device)
        labels = torch.as_tensor(self.expert_actions[ids], device=self.device)
        distribution = self.policy.get_distribution(observations, action_masks=self.expert_masks[ids])
        losses = -distribution.log_prob(labels)
        if self.expert_class_weights is None:
            return losses.mean()
        weights = torch.as_tensor(self.expert_class_weights[self.expert_actions[ids]], device=self.device)
        return (losses * weights).sum() / weights.sum()

    def train(self) -> None:
        """Cập nhật PPO với cùng mask rollout; thêm imitation theo lịch giảm dần."""
        if self.expert_observations is None or max(self.guidance_start, self.guidance_end) <= 0:
            return super().train()
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip = self.clip_range(self._current_progress_remaining)
        value_clip = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf else None
        coefficient = self.guidance_end + (self.guidance_start - self.guidance_end) * self._current_progress_remaining
        rows: list[tuple[float, ...]] = []
        updates, batches = 0, 0
        stop = False
        for _ in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                actions = data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(data.observations, actions, action_masks=data.action_masks)
                values = values.flatten()
                advantages = data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - data.old_log_prob)
                policy_loss = -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
                predictions = values if value_clip is None else data.old_values + torch.clamp(values - data.old_values, -value_clip, value_clip)
                value_loss = F.mse_loss(data.returns, predictions)
                entropy_loss = -entropy.mean() if entropy is not None else log_prob.mean()
                with torch.no_grad():
                    log_ratio = log_prob - data.old_log_prob
                    kl = float(((torch.exp(log_ratio) - 1) - log_ratio).mean().cpu())
                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    stop = True
                    break
                imitation = self._imitation_loss() if batches % self.guidance_interval == 0 else torch.zeros((), device=self.device)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + coefficient * imitation
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                clip_fraction = float(((ratio - 1).abs() > clip).float().mean().detach().cpu())
                rows.append((float(policy_loss.detach()), float(value_loss.detach()), float(entropy_loss.detach()), kl,
                             float(imitation.detach()), float(loss.detach()), clip_fraction))
                batches += 1
            updates += 1
            if stop:
                break
        self._n_updates += updates
        if rows:
            names = ("policy_gradient_loss", "value_loss", "entropy_loss", "approx_kl", "imitation_loss", "loss", "clip_fraction")
            for name, value in zip(names, np.mean(rows, axis=0)):
                self.logger.record(f"train/{name}", value)
        self.logger.record("train/guidance_coefficient", coefficient)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip)
        self.logger.record("train/explained_variance", explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))


TEACHERS = (("oracle", "high"), ("buy_hold", "close"), ("oracle", "high"),
            ("sma", "close"), ("oracle", "high"), ("momentum", "avg"),
            ("oracle", "high"), ("sma", "high"))


def teacher_actions(prices: pd.DataFrame, cfg: dict[str, Any], start: str, end: str,
                    trajectory: int) -> list[int]:
    name, mode = TEACHERS[trajectory % len(TEACHERS)]
    if prices.index.max() > pd.Timestamp(end) or pd.Timestamp(end) > pd.Timestamp(cfg["DATA"]["TRAIN_END"]):
        raise ValueError("Teacher chỉ được dùng prefix fit")
    if name == "oracle":
        ordinary = clone_config(cfg)
        ordinary["DYNAMIC_SELL_PRICES"], ordinary["SELL_PRICE_MODE"] = False, "high"
        execution_prices=prices
        if cfg.get('BUY_PRICE_MODE')=='low':
            execution_prices=prices.copy()
            execution_prices['Open']=execution_prices['Low']
        _, actions = constrained_oracle(execution_prices, ordinary, start, end)
        return [a + 10 if 6 <= a <= 10 else a for a in actions]
    if cfg.get('FIXED_SELL_PRICE')=='high':
        env=PriceExecutionEnv(prices,cfg,start_date=start,end_date=end,eval_mode=True)
        mode='high'
    else:
        env = FlexibleTradingEnv(prices, cfg, start_date=start, end_date=end, eval_mode=True)
    rng = np.random.default_rng(0)
    offset = {"close": 0, "avg": 5, "high": 10}[mode]

    def policy(obs: np.ndarray, state: TradingEnv) -> int:
        action = baseline_action(name, state, rng)
        return action + offset if 6 <= action <= 10 else action

    history = rollout(env, policy)
    env.close()
    return history.loc[history["event"] == "step", "action"].astype(int).tolist()


def make_env(prices: pd.DataFrame, cfg: dict[str, Any], start: str, end: str,
             *, evaluation: bool = False, features: pd.DataFrame | None = None) -> Any:
    """Dựng đúng wrapper đã cố định trong cấu hình thí nghiệm."""
    env = environment_class(cfg)(prices, cfg, features=features, start_date=start, end_date=end, eval_mode=evaluation)
    return PenaltyGuard(env) if cfg["PENALTY_GUARD"] else env


class TrainingCallback(BaseCallback):
    """Select frozen checkpoints on a strictly later, pre-test validation range."""
    def __init__(self,cfg,budget,directory,validation=None,validation_prices=None):
        super().__init__()
        self.cfg,self.budget,self.directory=cfg,budget,directory
        self.curve=[];self.best_steps=0;self.best_result={}
        self.validation,self.validation_prices=validation,validation_prices
        self.validation_rows=[]
    def _on_step(self):
        progress=min(1.0,self.num_timesteps/self.budget)
        train=self.cfg['TRAIN']
        self.model.ent_coef=(train['ENT_START']+progress*(train['ENT_END']-train['ENT_START']))*self.cfg.get('RISK',{}).get('entropy_multiplier',1.0)
        for info in self.locals.get('infos',[]):
            if 'episode' in info:
                ep=info['episode'];self.curve.append({'timesteps':self.num_timesteps,'episode_reward':ep['r'],'total_assets':ep['total_assets'],'length':ep['l']})
        return True
    def _on_rollout_end(self):
        write_json(self.directory/'progress.json',{'timesteps':self.num_timesteps,'target_timesteps':self.budget})
    def _evaluate(self):
        if self.validation is None:return
        normalizer=self.model.get_env()
        before=np.array(normalizer.obs_rms.mean,copy=True)
        env=make_env(self.validation_prices,self.cfg,*self.validation,evaluation=True)
        was_training=self.model.policy.training
        def predict(obs,state):
            action,_=self.model.predict(normalizer.normalize_obs(obs[None,:]),deterministic=True,
                action_masks=state.action_masks()[None,:])
            return int(np.asarray(action).item())
        try:
            h=rollout(env,predict,self.model.seed)
            result=metrics(h,self.cfg)
        finally:
            env.close();self.model.policy.set_training_mode(was_training)
        assert np.array_equal(before,normalizer.obs_rms.mean)
        key=lambda m:(m['final_assets_close'],-m['max_drawdown'])
        if not self.best_result or key(result)>key(self.best_result):
            self.best_result=result;self.best_steps=self.num_timesteps
            self.model.save(self.directory/'best.zip')
            normalizer.save(self.directory/'best_vecnormalize.pkl')
        self.validation_rows.append({'timesteps':self.num_timesteps,'metrics':result})
        write_json(self.directory/'validation.json',self.validation_rows)
        print(f'VALIDATION steps={self.num_timesteps} assets={result["final_assets_close"]:.2f}',flush=True)
    def _on_rollout_start(self):
        # This hook runs after the previous PPO update, unlike rollout_end.
        if self.num_timesteps>0 and self.num_timesteps%self.cfg['TRAIN']['EVAL_FREQ']==0:self._evaluate()
    def _on_training_end(self):
        self._evaluate()
        pd.DataFrame(self.curve,columns=['timesteps','episode_reward','total_assets','length']).to_csv(self.directory/'learning_curve.csv',index=False)



def behavior_clone(model: PPO | MaskablePPO, normalizer: VecNormalize, prices: pd.DataFrame,
                   cfg: dict[str, Any], start: str, end: str, seed: int, directory: Path,
                   oracle_cache: Path | None = None) -> None:
    """Dùng oracle chỉ trong phần fit; validation/test không được sinh nhãn BC."""
    if pd.Timestamp(end) > pd.Timestamp(cfg["DATA"]["TRAIN_END"]):
        raise ValueError("BC không được truy cập test")
    fit_prices = prices.loc[:end].copy()
    keys = ("INIT_CASH", "WIN", "TOTAL_ASSETS_FLOOR", "CASH_FLOOR", "BUY_FRACTIONS", "SELL_FRACTIONS",
            "FEE", "TRANSACTION_SESSION", "TRANSACTION_PENALTY", "SELL_PRICE_MODE", "FRACTIONAL_SHARES", "MIN_TRADE_VALUE")
    ranges = [(start, end)]
    count = cfg["TRAIN"].get("BC_TRAJECTORIES", 1)
    eligible = fit_prices.loc[start:end].index
    length = min(cfg["TRAIN"].get("BC_EPISODE_LEN", cfg["EPISODE_LEN"]), len(eligible))
    if count > 1:
        for offset in np.linspace(0, len(eligible) - length, count - 1, dtype=int):
            interval = (str(eligible[offset].date()), str(eligible[offset + length - 1].date()))
            if interval not in ranges:
                ranges.append(interval)
    observations, masks, actions = [], [], []
    records = []
    for trajectory, (episode_start, episode_end) in enumerate(ranges):
        prefix = fit_prices.loc[:episode_end]
        env = environment_class(cfg)(prefix, cfg, start_date=episode_start, end_date=episode_end, eval_mode=True)
        identity = {key: cfg[key] for key in keys}
        teacher = cfg["TRAIN"].get("BC_TEACHER", "oracle")
        identity.update(first=str(prefix.index[env.first_idx]), end=episode_end, oracle=cfg["ORACLE"],
                        teacher=teacher, baselines=cfg["BASELINES"])
        identity.update(buy_price_mode=cfg.get('BUY_PRICE_MODE','open'),fixed_sell_price=cfg.get('FIXED_SELL_PRICE'))
        if teacher == "flexible_expert":
            identity.update(trajectory=trajectory, teacher_source_hash=training_source_hash())
        digest = hashlib.sha256(pd.util.hash_pandas_object(prefix, index=True).values.tobytes() + json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache_file = (oracle_cache or directory / "oracle_cache") / f"{digest}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached["split"] != "train_only" or cached["identity"] != identity:
                raise RuntimeError("Cache oracle không đúng provenance train")
            expert_actions = cached["actions"]
        else:
            if teacher == "buy_hold":
                causal = run_baseline("buy_hold", prefix, cfg, episode_start, episode_end, seed)
                expert_actions = causal.loc[causal["event"] == "step", "action"].astype(int).tolist()
            elif teacher == "oracle":
                _, expert_actions = constrained_oracle(prefix, cfg, episode_start, episode_end)
            elif teacher == "flexible_expert":
                expert_actions = teacher_actions(prefix, cfg, episode_start, episode_end, trajectory)
            else:
                raise ValueError(f"BC_TEACHER không hỗ trợ: {teacher}")
            write_json(cache_file, {"split": "train_only", "identity": identity, "actions": expert_actions})
        obs, _ = env.reset(seed=seed)
        for action in expert_actions:
            observations.append(obs.copy())
            masks.append(env.action_masks())
            actions.append(action)
            obs, _, terminated, truncated, info = env.step(action)
            records.append({**info, "trajectory": trajectory})
            if terminated or truncated:
                break
        env.close()
    x = np.asarray(observations, dtype=np.float32)
    raw_observations = x.copy()
    normalizer.obs_rms.update(x)
    x = normalizer.normalize_obs(x)
    device = model.device
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=cfg["TRAIN"]["BC_LR"])
    rng = np.random.default_rng(seed)
    losses = []
    class_weights = action_class_weights(np.asarray(actions), model.action_space.n, cfg["TRAIN"].get("BC_CLASS_POWER", 0))
    sampling=expert_sampling_probabilities(np.asarray(actions),cfg['TRAIN'].get('BC_CLASS_POWER',0))
    model.policy.set_training_mode(True)
    for epoch in range(cfg["TRAIN"]["BC_EPOCHS"]):
        order = rng.choice(len(actions),len(actions),replace=True,p=sampling)
        epoch_loss = []
        for offset in range(0, len(order), cfg["TRAIN"]["BC_BATCH_SIZE"]):
            ids = order[offset:offset + cfg["TRAIN"]["BC_BATCH_SIZE"]]
            obs_tensor = torch.as_tensor(x[ids], device=device)
            labels = torch.as_tensor(np.asarray(actions)[ids], device=device, dtype=torch.long)
            kwargs = {"action_masks": np.asarray(masks)[ids]} if isinstance(model, MaskablePPO) else {}
            distribution = model.policy.get_distribution(obs_tensor, **kwargs)
            weights = torch.as_tensor(class_weights[np.asarray(actions)[ids]], device=device)
            loss = (-distribution.log_prob(labels) * weights).sum() / weights.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), cfg["TRAIN"].get("MAX_GRAD_NORM", 0.5))
            optimizer.step()
            epoch_loss.append(float(loss.detach().cpu()))
        losses.append({"epoch": epoch, "loss": float(np.mean(epoch_loss))})
    pd.DataFrame(losses).to_csv(directory / "bc_loss.csv", index=False)
    if isinstance(model, GuidedMaskablePPO):
        model.set_expert(raw_observations, np.asarray(actions), np.asarray(masks), normalizer, class_weights)
    pd.DataFrame(records).to_csv(directory / "bc_oracle_train.csv", index=False)
    write_json(directory / "bc_provenance.json", {"split": "train_only", "start": start, "end": end,
                                                  "teacher": cfg["TRAIN"].get("BC_TEACHER", "oracle"),
                                                  "sampling": "natural/side-balanced mixture using BC_CLASS_POWER",
                                                  "rows": len(actions), "seed": seed, "trajectories": ranges,
                                                  "class_counts": np.bincount(actions, minlength=model.action_space.n).tolist(),
                                                  "class_weights": class_weights.tolist()})


def fit_model(prices: pd.DataFrame, cfg: dict[str, Any], start: str, end: str,
              seed: int, budget: int, directory: Path,
              validation: tuple[str, str] | None = None,
              oracle_cache: Path | None = None) -> dict[str, Any]:
    """Fit trên đoạn train; lưu đầy đủ thống kê, cấu hình và ngân sách thực tế."""
    if pd.Timestamp(end) > pd.Timestamp(cfg["DATA"]["TRAIN_END"]):
        raise ValueError("Huấn luyện không được vượt TRAIN_END")
    if validation is not None:
        if not pd.Timestamp(end)<pd.Timestamp(validation[0])<=pd.Timestamp(validation[1])<pd.Timestamp(cfg['DATA']['TEST_START']):
            raise ValueError('Fit, validation and test must be strictly chronological and disjoint')
        if prices.index.max()>pd.Timestamp(validation[1]):
            raise ValueError('Training process may not receive any test row')
    input_end = end
    data_digest = hashlib.sha256(pd.util.hash_pandas_object(prices.loc[:input_end], index=True).values.tobytes()).hexdigest()
    run_file = directory / "run.json"
    if run_file.exists():
        raise FileExistsError("Use a fresh output for an actual new training run")
    seed_everything(seed)
    torch.set_num_threads(cfg["TRAIN"]["TORCH_THREADS"])
    directory.mkdir(parents=True, exist_ok=True)
    fit_prices = prices.loc[:end].copy()
    feature_frame = build_features(fit_prices, cfg)

    def factory(rank: int) -> Any:
        """Dựng một env có Monitor, dữ liệu riêng nhưng không rò rỉ holdout."""
        env = make_env(fit_prices, cfg, start, end, features=feature_frame)
        if cfg['TRAIN'].get('TRACK_INTRATRAIN_PEAKS',False):
            env=TrainingPeakLedger(env,directory/f'train_peak_events_rank_{rank}.csv',rank,cfg['TRAIN']['N_ENVS'],
                all_transitions=cfg['TRAIN'].get('TRACK_ALL_ASSETS',False),seed=seed,
                stage=cfg.get('RUN_STAGE','optimization'))
        env.reset(seed=seed + rank)
        return Monitor(env, filename=str(directory / f"env_{rank}"), info_keywords=("total_assets", "cum_reward"))

    vector = DummyVecEnv([lambda rank=rank: factory(rank) for rank in range(cfg["TRAIN"]["N_ENVS"])])
    norm_class = PortfolioVecNormalize if cfg["TRAIN"].get("PORTFOLIO_PRIOR", False) else VecNormalize
    normalizer = norm_class(vector, norm_obs=True, norm_reward=cfg["TRAIN"].get("NORM_REWARD", False),
                              gamma=cfg["TRAIN"]["GAMMA"], clip_obs=cfg["FEATURES"]["CLIP"])
    normalizer.seed(seed)
    train = cfg["TRAIN"]
    policy_kwargs: dict[str, Any] = {"net_arch": dict(pi=train["NET_ARCH"], vf=train["NET_ARCH"])}
    if train['ARCHITECTURE']!='state_mlp' or train['ALGORITHM']!='maskable':
        raise ValueError('Only the active Maskable PPO/state MLP is retained')
    policy_kwargs.update(features_extractor_class=MarketStateExtractor,
        features_extractor_kwargs={'window':cfg['WINDOW'],'hidden_size':train.get('STATE_HIDDEN',64)})
    use_guidance = train.get("GUIDANCE_ENABLED", False) and train["BC_ENABLED"] and train["ALGORITHM"] == "maskable"
    algorithm = GuidedMaskablePPO if use_guidance else MaskablePPO if train["ALGORITHM"] == "maskable" else PPO
    extra = {"guidance_start": train["GUIDANCE_START"], "guidance_end": train["GUIDANCE_END"],
             "guidance_batch_size": train["GUIDANCE_BATCH_SIZE"], "guidance_interval": train["GUIDANCE_INTERVAL"],
             "sampling_power":train.get('BC_CLASS_POWER',0)} if use_guidance else {}
    policy_class: Any = "MlpPolicy"
    if train.get("PORTFOLIO_PRIOR", False):
        if train["ALGORITHM"] != "maskable" or train["ARCHITECTURE"] != "state_mlp":
            raise ValueError("Portfolio prior cần MaskablePPO/state_mlp")
        policy_class = FlexiblePortfolioPolicy if cfg.get("DYNAMIC_SELL_PRICES", False) else PortfolioPriorPolicy
        policy_kwargs.update(cash_prior_scale=train["CASH_PRIOR_SCALE"], cash_prior_strength=train["CASH_PRIOR_STRENGTH"])
        policy_kwargs.update(cost_prior_strength=train.get("COST_PRIOR_STRENGTH", 0), transaction_fee=cfg["FEE"],
                             buy_fractions=tuple(cfg["BUY_FRACTIONS"]), sell_fractions=tuple(cfg["SELL_FRACTIONS"]))
        if cfg.get('FIXED_SELL_PRICE'):
            policy_kwargs['fixed_sell_price']=cfg['FIXED_SELL_PRICE']
            policy_kwargs.update(risk_hold_penalty=cfg.get('RISK',{}).get('hold_penalty',0.0),
                                 risk_sell_boost=cfg.get('RISK',{}).get('sell_boost',0.0))
    model = algorithm(policy_class, normalizer, learning_rate=lambda remaining: train["LEARNING_RATE"] * (train.get("MIN_LR_FRACTION", 0) + (1 - train.get("MIN_LR_FRACTION", 0)) * remaining),
                      n_steps=train["N_STEPS"], batch_size=train["BATCH_SIZE"], n_epochs=train["N_EPOCHS"],
                      gamma=train["GAMMA"], gae_lambda=train["GAE_LAMBDA"], clip_range=train["CLIP_RANGE"],
                      ent_coef=train["ENT_START"]*cfg.get('RISK',{}).get('entropy_multiplier',1.0), policy_kwargs=policy_kwargs, device=train["DEVICE"],
                      seed=seed, tensorboard_log=str(directory / "tensorboard"), verbose=0,
                      target_kl=train.get("TARGET_KL"), max_grad_norm=train.get("MAX_GRAD_NORM", 0.5),
                      vf_coef=train.get("VF_COEF", 0.5), **extra)
    with torch.no_grad():
        model.policy.action_net.bias[0] += train.get("HOLD_BIAS", 0.0)
        model.policy.action_net.bias[1:6] += train.get("BUY_BIAS", 0.0)
        model.policy.action_net.bias[6:11] += train.get("SELL_BIAS", 0.0)
    if train["BC_ENABLED"]:
        behavior_clone(model, normalizer, fit_prices, cfg, start, end, seed, directory, oracle_cache)
    callback = TrainingCallback(cfg, budget, directory, validation,
        prices.loc[:validation[1]].copy() if validation is not None else None)
    model.learn(total_timesteps=budget, callback=callback)
    model.save(directory / "last.zip")
    normalizer.save(directory / "last_vecnormalize.pkl")
    result = {"status": "complete", "seed": seed, "start": start, "end": end,
              "validation": validation, "requested_timesteps": budget,
              "actual_timesteps": model.num_timesteps, "best_checkpoint_steps": callback.best_steps,
              "validation_metrics": callback.best_result, "validation_checkpoints":callback.validation_rows,
              "config_hash": config_hash(cfg),
              "algorithm": train["ALGORITHM"], "data_hash": data_digest, "source_hash": training_source_hash()}
    artifact_names = ["last.zip", "last_vecnormalize.pkl", "learning_curve.csv"]
    result["artifacts"] = {name: file_hash(directory / name) for name in artifact_names}
    write_json(directory / "run.json", result)
    (directory / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    normalizer.close()
    return result
