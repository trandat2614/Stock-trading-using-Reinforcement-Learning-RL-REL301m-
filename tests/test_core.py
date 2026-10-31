"""One consolidated test file: rules, causal features, soft risk prior and real train/load."""
from pathlib import Path
from unittest.mock import patch
import shutil
import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch
from sb3_contrib import MaskablePPO
from gymnasium.utils.env_checker import check_env
from data import load_config,build_features
from environment import TradingEnv,PenaltyGuard,TrainingPeakLedger,cheapest_action,sell_price,audit_flexible
from baseline import rollout
from agent import make_env as lowhigh_env,fit_model,FlexiblePortfolioPolicy,MarketStateExtractor,expert_sampling_probabilities
from main import active_config as profile_config,frozen_history,compact_history,export_columns,campaign_config,select_validation_run


@pytest.fixture
def cfg() -> dict:
    """Cung cấp cấu hình độc lập cho từng test."""
    return load_config()


@pytest.fixture
def prices() -> pd.DataFrame:
    """Tạo giá tổng hợp xác định, tách hoàn toàn khỏi data/raw."""
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2015-10-01", "2026-10-29")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, len(dates))))
    opening = close * np.exp(rng.normal(0, 0.002, len(dates)))
    return pd.DataFrame({"Open": opening, "High": np.maximum(opening, close) * 1.01,
                         "Low": np.minimum(opening, close) * 0.99, "Close": close,
                         "Volume": rng.integers(1000000, 10000000, len(dates))}, index=dates).rename_axis("Date")


def make_env(prices: pd.DataFrame, cfg: dict, eval_mode: bool = True) -> TradingEnv:
    """Tạo môi trường ở đoạn train có đủ warm-up."""
    return TradingEnv(prices, cfg, start_date="2020-01-02", end_date="2024-10-16", eval_mode=eval_mode)


def test_fees_buy_sell(prices: pd.DataFrame, cfg: dict) -> None:
    """Cả hai phía thu đúng phí bốn phần trăm giá trị thực hiện."""
    env = make_env(prices, cfg)
    env.reset()
    opening = prices.iloc[env.idx]["Open"]
    _, _, _, _, buy = env.step(1)
    assert buy["fee"] == pytest.approx(100000 * 0.49 * 0.04)
    assert env.cash == pytest.approx(100000 - 49000 * 1.04)
    assert env.shares == pytest.approx(49000 / opening)
    old_shares, old_cash = env.shares, env.cash
    price = sell_price(prices.iloc[env.idx], cfg)
    _, _, _, _, sale = env.step(6)
    assert sale["fee"] == pytest.approx(old_shares * 0.21 * price * 0.04)
    assert env.cash == pytest.approx(old_cash + old_shares * 0.21 * price * 0.96)


def test_penalty_exact_19(prices: pd.DataFrame, cfg: dict) -> None:
    """Không phạt sớm; phiên idle 19 trừ đúng số tiền và reset."""
    env = make_env(prices, cfg)
    env.reset()
    for idle in range(1, 19):
        *_, info = env.step(0)
        assert env.idle == idle and info["penalty"] == 0
    *_, info = env.step(0)
    assert env.cash == 100000 - 3751 and env.idle == 0
    assert info["penalty"] == 3751


@pytest.mark.parametrize("cash,shares,outcome", [(-5000, 1000, "loss"), (999, 0, "loss"), (1000000, 0, "win")])
def test_termination(prices: pd.DataFrame, cfg: dict, cash: float, shares: float, outcome: str) -> None:
    """Thua/thắng ưu tiên trước truncated, kể cả ngay phiên cuối."""
    env = make_env(prices, cfg)
    env.reset(options={"length": 1})
    env.cash, env.shares = cash, shares
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and info["outcome"] == outcome


def test_mask_and_max_buy(prices: pd.DataFrame, cfg: dict) -> None:
    """Cấm bán khi không có cổ phiếu; mua tối đa vẫn giữ cash dương."""
    env = make_env(prices, cfg)
    env.reset()
    np.testing.assert_array_equal(env.action_masks(), [True] * 6 + [False] * 5)
    env.step(5)
    assert env.cash == pytest.approx(100000 * (1 - 0.83 * 1.04))
    assert env.cash > 0
    env.cash = 0
    np.testing.assert_array_equal(env.action_masks(), [True] + [False] * 5 + [True] * 5)


def test_guard_cheapest(prices: pd.DataFrame, cfg: dict) -> None:
    """Guard chỉ chọn phí nhỏ nhất theo giá đã biết, không nhìn High(t)."""
    base = make_env(prices, cfg)
    env = PenaltyGuard(base)
    env.reset()
    base.cash, base.shares, base.idle = 10000, 1000, 18
    expected = cheapest_action(base.cash, base.shares, prices.iloc[base.idx - 1]["Close"], cfg)
    assert expected == 1
    *_, info = env.step(0)
    assert info["action"] == expected and info["guard_applied"]
    assert info["penalty"] == 0 and base.idle == 0


def test_observation_no_lookahead(prices: pd.DataFrame, cfg: dict) -> None:
    """Quan sát và mask ở ngày t bất biến khi thay giá từ t trở đi."""
    first = make_env(prices, cfg)
    obs, _ = first.reset()
    changed = prices.copy()
    changed.iloc[first.idx:, :4] *= 10
    changed.iloc[first.idx:, 4] *= 20
    second = make_env(changed, cfg)
    obs2, _ = second.reset()
    np.testing.assert_array_equal(obs, obs2)
    np.testing.assert_array_equal(first.action_masks(), second.action_masks())


def test_same_seed(prices: pd.DataFrame, cfg: dict) -> None:
    """Cùng seed cho cùng điểm bắt đầu, hành động và lịch sử."""
    first, second = make_env(prices, cfg, False), make_env(prices, cfg, False)
    obs, _ = first.reset(seed=3)
    obs2, _ = second.reset(seed=3)
    np.testing.assert_array_equal(obs, obs2)
    for _ in range(20):
        a, b = first.action_space.sample(), second.action_space.sample()
        assert a == b
        left, right = first.step(a), second.step(b)
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]


def test_integer_and_min_trade(prices: pd.DataFrame, cfg: dict) -> None:
    """Làm tròn cổ phiếu và bỏ qua giao dịch dưới ngưỡng."""
    cfg["FRACTIONAL_SHARES"] = False
    cfg["MIN_TRADE_VALUE"] = 100
    env = make_env(prices, cfg)
    env.reset()
    env.step(5)
    assert env.shares == int(env.shares)
    env.cash = 50
    *_, info = env.step(1)
    assert not info["executed"] and info["fee"] == 0


def test_prev_high(prices: pd.DataFrame, cfg: dict) -> None:
    """High định giá là của phiên ngay trước phiên cuối."""
    env = make_env(prices, cfg)
    env.reset()
    traded_idx = env.idx
    env.step(5)
    assert env.total_assets_prev_high() == pytest.approx(env.cash + env.shares * prices.iloc[traded_idx - 1]["High"])


def test_gym_api(prices: pd.DataFrame, cfg: dict) -> None:
    """Môi trường đáp ứng hợp đồng reset/step của Gymnasium."""
    check_env(make_env(prices, cfg, False), skip_render_check=True)


@pytest.mark.parametrize("mode", ["open", "close", "avg", "high"])
def test_sell_modes(prices: pd.DataFrame, cfg: dict, mode: str) -> None:
    """Các chế độ chỉ thay giá khớp; không thay thời điểm quan sát."""
    cfg["SELL_PRICE_MODE"] = mode
    env = make_env(prices, cfg)
    env.reset()
    env.step(1)
    row = prices.iloc[env.idx]
    *_, info = env.step(6)
    expected = row[mode.title()] if mode != "avg" else row[["Open", "High", "Low", "Close"]].mean()
    assert info["price"] == pytest.approx(expected)


def test_invalid_and_skipped_trade(prices: pd.DataFrame, cfg: dict) -> None:
    """Bán khi không có cổ phiếu bị phạt reward và tăng idle."""
    cfg["REWARD"]["EXCESS"] = 0
    env = make_env(prices, cfg)
    env.reset()
    _, reward, _, _, info = env.step(6)
    assert info["invalid_action"] and not info["executed"]
    assert reward == -cfg["REWARD"]["INVALID"] and env.idle == 1


def test_guard_does_not_see_current_prices(prices: pd.DataFrame, cfg: dict) -> None:
    """Thay giá ngày t không thay hành động thay thế của guard."""
    env = PenaltyGuard(make_env(prices, cfg))
    env.reset()
    base = env.unwrapped
    changed = prices.copy()
    changed.iloc[base.idx:, :4] *= 50
    other = PenaltyGuard(make_env(changed, cfg))
    other.reset()
    for wrapped in (env, other):
        wrapped.unwrapped.cash, wrapped.unwrapped.shares, wrapped.unwrapped.idle = 10000, 500, 18
    *_, left = env.step(0)
    *_, right = other.step(0)
    assert left["action"] == right["action"]


def test_initial_and_episode_length(prices: pd.DataFrame, cfg: dict) -> None:
    """Snapshot bắt đầu đúng vốn; đoạn train không vượt giới hạn episode."""
    env = make_env(prices, cfg, False)
    _, info = env.reset(seed=4, options={"length": 2})
    assert info["event"] == "initial" and info["total_assets"] == 100000
    assert info["cash"] == 100000 and info["shares"] == 0
    assert not env.step(1)[3]
    assert env.step(0)[3]
    with pytest.raises(RuntimeError):
        env.step(0)


def test_rule_parameters_are_fixed():
    c=load_config()
    expected={'INIT_CASH':100000,'WIN':1000000,'TOTAL_ASSETS_FLOOR':1000,'CASH_FLOOR':-5000,
        'BUY_MIN':.49,'BUY_MAX':.83,'SELL_MIN':.21,'SELL_MAX':.69,'FEE':.04,
        'TRANSACTION_SESSION':19,'TRANSACTION_PENALTY':3751,'BUY_PRICE_MODE':'low','FIXED_SELL_PRICE':'high'}
    for k,v in expected.items():assert c[k]==v
    for seed in c['SEEDS']:
        current=profile_config(c,seed)
        assert all(current[k]==v for k,v in expected.items())
        assert current['REWARD']==c['REWARD'] and current['TRAIN']['GAMMA']==c['TRAIN']['GAMMA']


@pytest.mark.parametrize('buy,sell',[(1,16),(5,20)])
def test_low_high_fees_and_independent_accounting(prices,buy,sell):
    c=load_config();c['DATA'].update(TEST_START='2020-01-02',TEST_END='2020-02-14')
    env=lowhigh_env(prices,c,'2020-01-02','2020-02-14',evaluation=True)
    h=rollout(env,lambda o,s:buy if s.steps==0 else sell if s.steps==1 else 0)
    env.close();a,b=h.iloc[1],h.iloc[2]
    assert a.price==prices.loc[a.date,'Low'] and b.price==prices.loc[b.date,'High']
    assert a.shares_delta==pytest.approx(c['BUY_FRACTIONS'][buy-1]*100000/a.price)
    assert a.cash==pytest.approx(100000*(1-c['BUY_FRACTIONS'][buy-1]*1.04))
    assert b.shares_delta==pytest.approx(-c['SELL_FRACTIONS'][sell-16]*a.shares)
    assert audit_flexible(h,prices,c)['status']=='passed'


def test_future_extremes_do_not_enter_observation_or_risk_decision(prices):
    c=profile_config(load_config(),4);changed=prices.copy()
    changed.loc['2020-01-03':,'Low']*=.5;changed.loc['2020-01-03':,'High']*=2
    a=lowhigh_env(prices,c,'2020-01-03','2020-01-10',evaluation=True)
    b=lowhigh_env(changed,c,'2020-01-03','2020-01-10',evaluation=True)
    x,_=a.reset(seed=0);y,_=b.reset(seed=0)
    np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(a.action_masks(),b.action_masks())
    assert a.action_masks()[0] and not a.action_masks()[6:16].any()
    aa=a.step(1)[-1];bb=b.step(1)[-1]
    assert bb['cash']==aa['cash'] and bb['shares']==pytest.approx(2*aa['shares'])
    a.close();b.close()


def test_risk_prior_is_soft_and_survives_serialization(prices,tmp_path):
    c=load_config();env=lowhigh_env(prices,c,'2020-01-03','2020-01-10')
    model=MaskablePPO(FlexiblePortfolioPolicy,env,n_steps=8,batch_size=8,device='cpu',
        policy_kwargs={'fixed_sell_price':'high','risk_hold_penalty':1.8,'risk_sell_boost':.7},seed=0)
    p=model.policy
    latent=torch.zeros((1,p.mlp_extractor.latent_dim_pi),device=model.device)
    latent[:,-6]=.2;latent[:,-5]=.8
    aggressive=p._get_action_dist_from_latent(latent).distribution.probs.detach().clone()
    p.risk_hold_penalty=0;p.risk_sell_boost=0
    plain=p._get_action_dist_from_latent(latent).distribution.probs.detach().clone()
    assert 0<aggressive[0,0]<plain[0,0]
    assert aggressive[0,16:].sum()>plain[0,16:].sum()
    p.risk_hold_penalty=1.8;p.risk_sell_boost=.7
    model.save(tmp_path/'model.zip');loaded=MaskablePPO.load(tmp_path/'model.zip',device='cpu')
    assert loaded.policy.risk_hold_penalty==1.8 and loaded.policy.risk_sell_boost==.7
    torch.testing.assert_close(loaded.policy._get_action_dist_from_latent(latent).distribution.probs,aggressive)
    env.close()


class Toy(gym.Env):
    def __init__(self):
        self.observation_space=gym.spaces.Box(0,1,(1,));self.action_space=gym.spaces.Discrete(1);self.steps=0
    def reset(self,**kwargs):
        self.steps=0
        return np.zeros(1),{'event':'initial','date':'2020-01-01','total_assets':100000}
    def step(self,action):
        v=[120000,110000,130000,105000][self.steps];self.steps+=1
        return np.zeros(1),0,False,self.steps==4,{'event':'step','date':f'2020-01-0{self.steps+1}','total_assets':v}


def test_full_train_logger_retains_falls_and_unfinished_episode(tmp_path):
    p=tmp_path/'train_peak_events_rank_2.csv'
    e=TrainingPeakLedger(Toy(),p,2,8,all_transitions=True,seed=4,stage='final_train');e.reset()
    for i in range(3):e.step(0)
    e.close();h=pd.read_csv(p)
    assert h.total_assets.tolist()==[100000,120000,110000,130000]
    assert h.asset_change.tolist()==[0,20000,-10000,20000]
    assert h.new_episode_peak.tolist()==[True,True,False,True]
    assert h.total_assets.max()==130000
    assert h.global_transition.tolist()==[0,3,11,19]


def test_real_training_and_frozen_prediction(prices,tmp_path):
    c=profile_config(load_config(),1);c['WINDOW']=5;c['EPISODE_LEN']=12
    c['TRAIN'].update(N_ENVS=1,N_STEPS=16,BATCH_SIZE=8,N_EPOCHS=1,NET_ARCH=[16],STATE_HIDDEN=8,
        DEVICE='cpu',BC_EPOCHS=1,BC_TRAJECTORIES=1,BC_BATCH_SIZE=8,BC_TEACHER='buy_hold',GUIDANCE_BATCH_SIZE=8)
    c['DATA'].update(TRAIN_START='2020-01-02',TRAIN_END='2020-01-31',TEST_START='2020-02-03',TEST_END='2020-02-14')
    r=fit_model(prices,c,'2020-01-02','2020-01-31',1,32,tmp_path/'fit')
    assert r['actual_timesteps']==32
    (tmp_path/'models').mkdir()
    for old,new in [('last.zip','agent.zip'),('last_vecnormalize.pkl','normalizer.pkl')]:
        shutil.copy2(tmp_path/'fit'/old,tmp_path/'models'/new)
    with patch.object(MaskablePPO,'learn',side_effect=AssertionError('Frozen test must not learn')):
        a,proof=frozen_history(tmp_path,c,prices,1,'2020-02-03','2020-02-14')
        b,_=frozen_history(tmp_path,c,prices,1,'2020-02-03','2020-02-14')
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    assert proof['normalizer_unchanged'] and proof['audit']['status']=='passed'
    log=compact_history(a,prices,'test')
    assert {'assets','close','price_change_pct','is_peak','cumulative_reward'}.issubset(log.columns)
    assert len(log.columns)==20
    p=next((tmp_path/'fit').glob('train_peak_events_rank_*.csv'))
    raw=pd.read_csv(p)
    assert (raw.event=='step').sum()==32
    np.testing.assert_allclose(raw.total_assets,raw.cash+raw.shares*raw.close,atol=1e-6)


def test_temporal_encoder_distinguishes_equal_mean_variance_windows():
    torch.manual_seed(9)
    space=gym.spaces.Box(-np.inf,np.inf,(10*2+6,),dtype=np.float32)
    encoder=MarketStateExtractor(space,10,8)
    a=torch.zeros(1,10,2);a[0,2,0]=2;a[0,6,0]=-2
    b=a.clone();b[0,2,0]=-2;b[0,6,0]=2
    torch.testing.assert_close(a.mean(1),b.mean(1))
    torch.testing.assert_close(a.std(1),b.std(1));torch.testing.assert_close(a[:,-1],b[:,-1])
    portfolio=torch.tensor([[.3,.7,0,0,0,0]])
    x=encoder(torch.cat([a.flatten(1),portfolio],1))
    y=encoder(torch.cat([b.flatten(1),portfolio],1))
    assert not torch.allclose(x[:,:-6],y[:,:-6])
    torch.testing.assert_close(x[:,-6:],portfolio)


@pytest.mark.parametrize('power',[0,.5,1])
def test_expert_sampling_balances_sides_without_removing_hold(power):
    actions=np.array([0]*90+[1]*8+[20]*2)
    p=expert_sampling_probabilities(actions,power)
    assert np.isfinite(p).all() and (p>0).all() and p.sum()==pytest.approx(1)
    expected=(1-power)*.9+power/3
    assert p[actions==0].sum()==pytest.approx(expected)
    assert p[actions==20].sum()==pytest.approx((1-power)*.02+power/3)


def test_ten_seed_config_keeps_identical_rules_and_risk(prices):
    base=load_config();current,protocol=campaign_config(base,prices)
    assert protocol['fit_start']=='2014-10-29' and protocol['fit_end']=='2022-10-28'
    assert protocol['validation_start']=='2022-10-31' and protocol['validation_end']=='2024-10-29'
    assert protocol['test_start']==base['DATA']['TEST_START'] and protocol['test_end']==base['DATA']['TEST_END']
    assert current['TRAIN']==base['TRAIN']
    assert base['DATA']['TRAIN_START']=='2016-10-31'
    for seed in range(10):assert profile_config(base,seed)['RISK']==base['RISK_PROFILES'][1]


def test_run_selection_ignores_test_outcome():
    rows=[{'run':1,'seed':0,'validation':{'final_assets_close':110000,'max_drawdown':.2},'test':{'final_assets_close':1}},
        {'run':2,'seed':1,'validation':{'final_assets_close':105000,'max_drawdown':.1},'test':{'final_assets_close':1000000}}]
    assert select_validation_run(rows)['run']==1
    rows[1]['test']['final_assets_close']=1000000000
    assert select_validation_run(rows)['run']==1


def test_validation_checkpoints_are_real_and_fit_is_disjoint(prices,tmp_path):
    c=profile_config(load_config());c['WINDOW']=5;c['EPISODE_LEN']=12
    c['TRAIN'].update(N_ENVS=1,N_STEPS=16,BATCH_SIZE=8,N_EPOCHS=1,NET_ARCH=[16],STATE_HIDDEN=8,
        DEVICE='cpu',BC_EPOCHS=1,BC_TRAJECTORIES=1,BC_BATCH_SIZE=8,BC_TEACHER='buy_hold',GUIDANCE_BATCH_SIZE=8,EVAL_FREQ=16)
    c['DATA'].update(TRAIN_START='2020-01-02',TRAIN_END='2020-01-15',TEST_START='2020-02-03',TEST_END='2020-02-14')
    pretest=prices.loc[:'2020-01-31'];validation=('2020-01-16','2020-01-31')
    fitted=fit_model(pretest,c,'2020-01-02','2020-01-15',1,32,tmp_path/'fit',validation=validation)
    assert [v['timesteps'] for v in fitted['validation_checkpoints']]==[16,32]
    assert fitted['best_checkpoint_steps'] in [16,32]
    model_dir=tmp_path/'selected';model_dir.mkdir()
    for old,new in [('best.zip','agent.zip'),('best_vecnormalize.pkl','normalizer.pkl')]:
        shutil.copy2(tmp_path/'fit'/old,model_dir/new)
    with patch.object(MaskablePPO,'learn',side_effect=AssertionError('Validation must not learn')):
        history,proof=frozen_history(tmp_path,c,pretest,1,*validation,model_dir)
    assert history.iloc[-1].total_assets==pytest.approx(fitted['validation_metrics']['final_assets_close'])
    assert proof['normalizer_unchanged']
    with pytest.raises(ValueError,match='test row'):
        fit_model(prices,c,'2020-01-02','2020-01-15',1,32,tmp_path/'bad',validation=validation)
    with pytest.raises(ValueError,match='disjoint'):
        fit_model(pretest,c,'2020-01-02','2020-01-15',1,32,tmp_path/'bad2',validation=('2020-01-15','2020-01-31'))

