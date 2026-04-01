"""
models/rl_agents.py
====================
PPO and Deep Q-Learning agents with:
  - 3-action space: 0=Buy, 1=Hold, 2=Sell
  - Decomposable reward (P&L - drawdown - costs - overtrading)
  - Combined pyramiding + martingale scaling strategy
  - Dynamic stop-loss integration
"""

import numpy as np
import collections
import random
from typing import Optional, Tuple, List, Dict

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
# ACTIONS (3-action space as specified)
# ─────────────────────────────────────────────────────────────────────────────

BUY  = 0
HOLD = 1
SELL = 2
ACTION_NAMES = {BUY: "BUY", HOLD: "HOLD", SELL: "SELL"}


# ─────────────────────────────────────────────────────────────────────────────
# TRADING ENVIRONMENT  (3-action, combined scaling, dynamic SL)
# ─────────────────────────────────────────────────────────────────────────────

class ForexTradingEnv:
    """
    Forex environment for 3-action RL agents.

    Scaling strategy: BOTH combined (spec requirement)
      - Pyramiding:  add to winning positions (momentum)
      - Martingale:  add to losing positions (mean-reversion, guarded)

    Reward is DECOMPOSABLE into 4 components that can be monitored
    independently in production to diagnose performance decay.
    """

    def __init__(
        self,
        features:       np.ndarray,
        prices:         np.ndarray,
        atr:            np.ndarray,
        spreads:        np.ndarray,
        initial_equity: float = 10_000.0,
        lot_size:       float = 10_000.0,
        max_lots:       float = 3.0,
        commission_per_lot: float = 3.5,
        slippage_pips:  float = 0.5,
        pip_size:       float = 0.0001,
        reward_weights: Optional[Dict] = None,
        # Dynamic SL params
        atr_sl_mult:    float = 1.5,
        trail_activation_r: float = 1.0,
        breakeven_at_r: float = 0.5,
        # Scaling params (both)
        pyramid_pct:    float = 0.25,
        martingale_pct: float = 0.25,
    ):
        self.features       = features.astype(np.float32)
        self.prices         = prices.astype(np.float32)
        self.atr            = atr.astype(np.float32)
        self.spreads        = spreads.astype(np.float32)
        self.initial_equity = initial_equity
        self.lot_size       = lot_size
        self.max_lots       = max_lots
        self.commission     = commission_per_lot
        self.slippage_pips  = slippage_pips
        self.pip_size       = pip_size
        self.rw             = reward_weights or {
            "pnl": 1.0, "drawdown": 0.5, "tx_cost": 0.3, "overtrade": 0.2
        }
        self.atr_sl_mult    = atr_sl_mult
        self.trail_act_r    = trail_activation_r
        self.be_r           = breakeven_at_r
        self.pyramid_pct    = pyramid_pct
        self.martingale_pct = martingale_pct

        self.obs_size = features.shape[1] + 5  # + agent state
        self.n_actions = 3
        self.reset()

    def reset(self) -> np.ndarray:
        self.idx        = 0
        self.equity     = self.initial_equity
        self.peak       = self.initial_equity
        self.position   = 0.0      # lots (+long, -short)
        self.entry_price = 0.0
        self.stop_loss  = 0.0
        self.take_profit = 0.0
        self.holding    = 0
        self.n_trades   = 0
        self.total_costs = 0.0
        self.episode_pnl = []
        self.done       = False
        return self._obs()

    def _obs(self) -> np.ndarray:
        mkt = self.features[self.idx]
        p   = self.prices[self.idx]
        upnl = (p - self.entry_price) * self.position * self.lot_size if self.position != 0 else 0.0
        agent = np.array([
            np.clip(self.position / self.max_lots, -1, 1),
            np.clip(upnl / self.initial_equity, -0.5, 0.5),
            min(self.holding / 100, 1.0),
            np.clip((self.equity - self.initial_equity) / self.initial_equity, -0.5, 0.5),
            int(self.position != 0),
        ], dtype=np.float32)
        return np.concatenate([mkt, agent])

    def _exec_cost(self, lots: float) -> float:
        cost = abs(lots) * self.commission + abs(lots) * self.slippage_pips * self.pip_size * self.lot_size
        self.equity -= cost; self.total_costs += cost
        return cost

    def _set_dynamic_sl(self, direction: int, entry: float, current_atr: float):
        """Dynamic stop-loss: ATR-based, trails after profit, moves to breakeven."""
        self.stop_loss   = entry - direction * self.atr_sl_mult * current_atr
        self.take_profit = entry + direction * self.atr_sl_mult * 2 * current_atr

    def _update_trailing_sl(self, p: float, direction: int, entry: float, current_atr: float):
        """Trail SL after profit exceeds trail_activation_r × ATR."""
        if self.position == 0: return
        profit_r = direction * (p - entry) / (current_atr + 1e-9)
        if profit_r >= self.trail_act_r:
            new_sl = p - direction * self.atr_sl_mult * current_atr
            if direction > 0: self.stop_loss = max(self.stop_loss, new_sl)
            else:             self.stop_loss = min(self.stop_loss, new_sl)
        elif profit_r >= self.be_r:
            # Move to breakeven
            if direction > 0: self.stop_loss = max(self.stop_loss, entry)
            else:             self.stop_loss = min(self.stop_loss, entry)

    def _check_sl_tp(self, p: float, direction: int) -> Tuple[bool, float]:
        """Returns (hit, pnl_pips)."""
        if direction > 0:
            if p <= self.stop_loss:   return True, (self.stop_loss - self.entry_price) / self.pip_size
            if p >= self.take_profit: return True, (self.take_profit - self.entry_price) / self.pip_size
        else:
            if p >= self.stop_loss:   return True, (self.entry_price - self.stop_loss) / self.pip_size
            if p <= self.take_profit: return True, (self.entry_price - self.take_profit) / self.pip_size
        return False, 0.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert not self.done
        p   = self.prices[self.idx]
        atr = self.atr[self.idx]
        direction = int(np.sign(self.position)) if self.position != 0 else 0

        realised_pnl = 0.0; cost = 0.0

        # ── Check dynamic stop/TP before executing new action ───────────────
        if self.position != 0:
            self._update_trailing_sl(p, direction, self.entry_price, atr)
            hit, pips = self._check_sl_tp(p, direction)
            if hit:
                pnl_usd = pips * self.pip_size * abs(self.position) * self.lot_size
                realised_pnl += pnl_usd; self.equity += pnl_usd
                self.position = 0.0; self.holding = 0
                self.n_trades += 1

        # ── Execute action ──────────────────────────────────────────────────
        if action == BUY:
            if self.position == 0:                         # Open long
                lots = 1.0
                cost = self._exec_cost(lots)
                self.position = lots; self.entry_price = p
                self._set_dynamic_sl(+1, p, atr)
            elif self.position > 0 and abs(self.position) < self.max_lots:
                # Pyramid (add to winning long)
                upnl = (p - self.entry_price) * self.position * self.lot_size
                if upnl > 0:
                    add = min(self.pyramid_pct, self.max_lots - self.position)
                    self._update_avg_entry(p, add); cost = self._exec_cost(add)
                else:
                    # Martingale (add to losing long — guarded)
                    add = min(self.martingale_pct, self.max_lots - self.position)
                    self._update_avg_entry(p, add); cost = self._exec_cost(add)
            elif self.position < 0:                        # Close short
                pnl = (self.entry_price - p) * abs(self.position) * self.lot_size
                realised_pnl += pnl; self.equity += pnl
                cost = self._exec_cost(abs(self.position))
                self.position = 0.0; self.holding = 0; self.n_trades += 1

        elif action == SELL:
            if self.position == 0:                         # Open short
                lots = 1.0
                cost = self._exec_cost(lots)
                self.position = -lots; self.entry_price = p
                self._set_dynamic_sl(-1, p, atr)
            elif self.position < 0 and abs(self.position) < self.max_lots:
                # Pyramid or martingale on short
                upnl = (self.entry_price - p) * abs(self.position) * self.lot_size
                add_pct = self.pyramid_pct if upnl > 0 else self.martingale_pct
                add = min(add_pct, self.max_lots - abs(self.position))
                self._update_avg_entry(p, -add); cost = self._exec_cost(add)
            elif self.position > 0:                        # Close long
                pnl = (p - self.entry_price) * self.position * self.lot_size
                realised_pnl += pnl; self.equity += pnl
                cost = self._exec_cost(self.position)
                self.position = 0.0; self.holding = 0; self.n_trades += 1

        # HOLD: do nothing

        if self.position != 0: self.holding += 1
        self.peak = max(self.peak, self.equity)
        self.episode_pnl.append(realised_pnl)

        # ── Decomposable reward ──────────────────────────────────────────────
        w = self.rw
        dd = max(0, (self.peak - self.equity) / self.peak)
        reward = (
            w["pnl"]       * realised_pnl / self.initial_equity
          - w["drawdown"]  * dd
          - w["tx_cost"]   * cost / self.initial_equity
          - w["overtrade"] * (1.0 / max(self.n_trades, 1) if self.n_trades > 50 else 0)
        )

        self.idx += 1
        self.done = self.idx >= len(self.prices) - 1

        # Force-close at episode end
        if self.done and self.position != 0:
            last_p = self.prices[-1]
            d = int(np.sign(self.position))
            final_pnl = d * (last_p - self.entry_price) * abs(self.position) * self.lot_size
            self.equity += final_pnl

        obs = self._obs() if not self.done else np.zeros(self.obs_size, dtype=np.float32)
        return obs, float(reward), self.done, {"equity": self.equity, "pnl": realised_pnl}

    def _update_avg_entry(self, p: float, delta_lots: float):
        total = abs(self.position) + abs(delta_lots)
        self.entry_price = (abs(self.position) * self.entry_price + abs(delta_lots) * p) / total
        self.position += delta_lots

    def summary(self) -> dict:
        rets = np.array(self.episode_pnl)
        sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252) if len(rets) > 1 else 0.0
        return {
            "total_return_pct": (self.equity / self.initial_equity - 1) * 100,
            "sharpe": sharpe,
            "n_trades": self.n_trades,
            "total_costs": self.total_costs,
            "max_dd_pct": max(0, (self.peak - self.equity) / self.peak) * 100,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PPO AGENT
# ─────────────────────────────────────────────────────────────────────────────

if TORCH:

    class ActorCritic(nn.Module):
        """Shared backbone → actor (policy) + critic (value) heads."""
        def __init__(self, obs_size, n_actions=3, hidden=256):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(obs_size, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden),   nn.Tanh(),
            )
            self.actor  = nn.Linear(hidden, n_actions)
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.backbone(x)
            return self.actor(h), self.critic(h)

        def act(self, obs):
            logits, value = self(obs)
            dist   = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action, dist.log_prob(action), value.squeeze(-1)

        def evaluate(self, obs, actions):
            logits, value = self(obs)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(actions), dist.entropy(), value.squeeze(-1)


    class PPOAgent:
        """
        Proximal Policy Optimisation agent for 3-action forex trading.
        Uses GAE advantage estimation and clip-objective for stable training.
        """
        def __init__(self, obs_size, n_actions=3, hidden=256,
                     lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                     entropy_coef=0.01, value_coef=0.5, n_epochs=10, device="cpu"):
            self.gamma = gamma; self.lam = lam; self.clip = clip
            self.ent_c = entropy_coef; self.val_c = value_coef
            self.n_epochs = n_epochs
            self.device = torch.device(device)
            self.net = ActorCritic(obs_size, n_actions, hidden).to(self.device)
            self.opt = optim.Adam(self.net.parameters(), lr=lr)
            self.buffer = []

        def select_action(self, obs: np.ndarray):
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = self.net.act(x)
            return action.item(), log_prob.item(), value.item()

        def store(self, obs, action, reward, done, log_prob, value):
            self.buffer.append((obs, action, reward, done, log_prob, value))

        def update(self):
            if len(self.buffer) < 64: return {}
            obs_b, act_b, rew_b, done_b, lp_b, val_b = zip(*self.buffer)
            self.buffer.clear()

            obs_t  = torch.tensor(np.array(obs_b),  dtype=torch.float32, device=self.device)
            act_t  = torch.tensor(act_b,            dtype=torch.long,    device=self.device)
            rew_t  = torch.tensor(rew_b,            dtype=torch.float32, device=self.device)
            done_t = torch.tensor(done_b,           dtype=torch.float32, device=self.device)
            old_lp = torch.tensor(lp_b,             dtype=torch.float32, device=self.device)
            val_t  = torch.tensor(val_b,            dtype=torch.float32, device=self.device)

            # GAE advantage
            adv = torch.zeros_like(rew_t); gae = 0.0
            for t in reversed(range(len(rew_t))):
                nv  = val_t[t+1] if t < len(rew_t)-1 else 0.0
                delta = rew_t[t] + self.gamma * nv * (1 - done_t[t]) - val_t[t]
                gae   = delta + self.gamma * self.lam * (1 - done_t[t]) * gae
                adv[t] = gae
            returns = adv + val_t
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            total_loss = 0.0
            for _ in range(self.n_epochs):
                lp_new, entropy, val_new = self.net.evaluate(obs_t, act_t)
                ratio  = (lp_new - old_lp).exp()
                s1 = ratio * adv
                s2 = ratio.clamp(1 - self.clip, 1 + self.clip) * adv
                pol_loss = -torch.min(s1, s2).mean()
                val_loss = F.mse_loss(val_new, returns)
                loss = pol_loss + self.val_c * val_loss - self.ent_c * entropy.mean()
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                total_loss += loss.item()

            return {"loss": total_loss / self.n_epochs}


    # ── Deep Q-Network ────────────────────────────────────────────────────

    class DQNetwork(nn.Module):
        """Double DQN network."""
        def __init__(self, obs_size, n_actions=3, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_size, hidden), nn.ReLU(),
                nn.Linear(hidden,   hidden), nn.ReLU(),
                nn.Linear(hidden,   n_actions),
            )
        def forward(self, x): return self.net(x)


    class ReplayBuffer:
        def __init__(self, capacity=1_000_000):
            self.buf = collections.deque(maxlen=capacity)
        def push(self, *args): self.buf.append(args)
        def sample(self, n):   return random.sample(self.buf, n)
        def __len__(self):     return len(self.buf)


    class DQNAgent:
        """
        Double DQN agent — faster inference (~2ms) than PPO, ideal for TIP-Search
        fast model. Epsilon-greedy exploration decays over training.
        """
        def __init__(self, obs_size, n_actions=3, hidden=256,
                     lr=1e-4, gamma=0.99, eps_start=1.0, eps_end=0.01,
                     eps_decay=0.995, buf_size=1_000_000, batch=64,
                     target_update=100, double_dqn=True, device="cpu"):
            self.n_actions = n_actions; self.gamma = gamma; self.batch = batch
            self.double = double_dqn; self.target_update = target_update
            self.eps = eps_start; self.eps_end = eps_end; self.eps_decay = eps_decay
            self.device = torch.device(device)
            self.policy_net = DQNetwork(obs_size, n_actions, hidden).to(self.device)
            self.target_net = DQNetwork(obs_size, n_actions, hidden).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.opt    = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.buf    = ReplayBuffer(buf_size)
            self.steps  = 0

        def select_action(self, obs: np.ndarray) -> int:
            if random.random() < self.eps:
                return random.randrange(self.n_actions)
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                return self.policy_net(x).argmax(dim=1).item()

        def store(self, obs, action, reward, next_obs, done):
            self.buf.push(obs, action, reward, next_obs, done)
            self.eps = max(self.eps_end, self.eps * self.eps_decay)

        def update(self) -> dict:
            if len(self.buf) < self.batch: return {}
            batch = self.buf.sample(self.batch)
            obs_b, act_b, rew_b, nobs_b, done_b = zip(*batch)

            obs  = torch.tensor(np.array(obs_b),  dtype=torch.float32, device=self.device)
            acts = torch.tensor(act_b,             dtype=torch.long,    device=self.device)
            rews = torch.tensor(rew_b,             dtype=torch.float32, device=self.device)
            nobs = torch.tensor(np.array(nobs_b),  dtype=torch.float32, device=self.device)
            done = torch.tensor(done_b,             dtype=torch.float32, device=self.device)

            q_vals = self.policy_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                if self.double:
                    best_acts = self.policy_net(nobs).argmax(1)
                    next_q = self.target_net(nobs).gather(1, best_acts.unsqueeze(1)).squeeze(1)
                else:
                    next_q = self.target_net(nobs).max(1)[0]
                target = rews + self.gamma * next_q * (1 - done)

            loss = F.smooth_l1_loss(q_vals, target)
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.opt.step()

            self.steps += 1
            if self.steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            return {"loss": loss.item(), "epsilon": self.eps}

else:
    class PPOAgent:
        def __init__(self, **kw): pass
        def select_action(self, obs): return HOLD
        def store(self, *a): pass
        def update(self): return {}

    class DQNAgent:
        def __init__(self, **kw): pass
        def select_action(self, obs): return HOLD
        def store(self, *a): pass
        def update(self): return {}


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_agent(
    agent,
    env: ForexTradingEnv,
    n_episodes: int = 100,
    n_steps_ppo: int = 2048,
    agent_type: str = "ppo",
) -> list:
    """Generic training loop for PPO or DQN."""
    returns = []
    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        while not env.done:
            if agent_type == "ppo":
                action, log_prob, value = agent.select_action(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.store(obs, action, reward, done, log_prob, value)
                ep_reward += reward; obs = next_obs
                if len(agent.buffer) >= n_steps_ppo:
                    agent.update()
            else:  # dqn
                action = agent.select_action(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.store(obs, action, reward, next_obs, done)
                agent.update()
                ep_reward += reward; obs = next_obs

        summary = env.summary()
        returns.append(summary["total_return_pct"])
        if (ep + 1) % 10 == 0:
            avg = np.mean(returns[-10:])
            print(f"  Ep {ep+1:3d}/{n_episodes} | "
                  f"Return: {summary['total_return_pct']:+.2f}% | "
                  f"Avg10: {avg:+.2f}% | Trades: {summary['n_trades']}")

    return returns


if __name__ == "__main__":
    print("RL Agents smoke test (3-action space)")
    import sys; sys.path.insert(0, "..")
    from data.data_ingestion import generate_synthetic_tick_data, ForexDataPipeline
    from features.feature_engineering import FeatureEngineer

    ticks = generate_synthetic_tick_data(n_rows=200_000)
    bars  = ForexDataPipeline(bar_freq="5min", session_filter=False, apply_frac_diff=False).run(ticks)
    fe    = FeatureEngineer()
    feats = fe.build(bars)
    bars_a = bars.reindex(feats.index).dropna()

    prices  = bars_a["close"].values.astype(np.float32)
    atr     = feats["atr_6"].values.astype(np.float32)
    spreads = np.full(len(prices), 0.00005, dtype=np.float32)
    feat_arr = feats.values.astype(np.float32)

    env = ForexTradingEnv(feat_arr, prices, atr, spreads)
    print(f"Env obs_size={env.obs_size}, n_actions={env.n_actions}")

    if TORCH:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        dqn = DQNAgent(obs_size=env.obs_size, device=device)
        returns = train_agent(dqn, env, n_episodes=5, agent_type="dqn")
        print(f"\nDQN 5-episode returns: {[f'{r:+.2f}%' for r in returns]}")
    else:
        print("Install torch for full RL training.")
