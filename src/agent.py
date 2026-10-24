"""PPO for the batched charging simulator, with three interchangeable policy backbones.

  mlp       : flat concatenation of all M slots + global features (slot order matters)
  deepsets  : shared per-EV encoder, masked mean/max pooling (permutation-equivariant)
  attention : shared per-EV encoder + masked multi-head self-attention (equivariant)

The actor outputs, for every slot, a Beta distribution over the fraction of the
slot's feasible rate in [0, 1]; empty slots are masked out of the log-probability.
The critic sees the same backbone (pooled for the equivariant ones).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta

from env import HORIZON, BatchSim


@dataclass
class PPOConfig:
    arch: str = "deepsets"       # mlp | deepsets | attention
    hidden: int = 128
    layers: int = 2
    heads: int = 4               # attention only
    lr: float = 3e-4
    lr_anneal: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatches: int = 8
    rollout_len: int = 96        # decisions per env per update
    n_envs: int = 16
    total_steps: int = 1_000_000
    reward_scale: float = 1.0    # divides reward by (p_cap * dt) * reward_scale
    activation: str = "tanh"     # tanh | relu
    action: str = "aggregate"    # aggregate (hub set-point, 1-D) | per_ev (one rate per slot)
    safety: str = "guard+llf"   # none | llf | guard+llf  (see BatchSim.apply), per_ev only
    shaping: bool = True         # potential-based shaping on undeliverable energy
    center_price: bool = True    # train on (p - 1) and lambda - 1: same optimum, lower variance
    action_repeat: int = 4       # simulator steps per decision (4 = hourly set-points)
    norm_obs: bool = True        # running mean/std normalisation of the global features
    norm_reward: bool = True     # divide rewards by the running std of the discounted return
    price_aug: bool = False      # pair each sampled training week with a random 2018-19 price week
    dep_noise_train: float = 0.0 # max std (h) of departure-announcement error, drawn per training batch
    init_mean: float = 0.6       # initial mean of the action (set-point or per-slot fraction)
    seed: int = 0
    threads: int = 2


def _act(name):
    return {"tanh": nn.Tanh, "relu": nn.ReLU}[name]


def mlp(i, h, o, layers, act):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), act()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class Policy(nn.Module):
    def __init__(self, M, F, G, cfg: PPOConfig):
        super().__init__()
        self.arch, self.M = cfg.arch, M
        self.agg = cfg.action == "aggregate"
        act, h, L = _act(cfg.activation), cfg.hidden, cfg.layers
        if cfg.arch == "mlp":
            self.body = mlp(M * F + G, h, h, L, act)
            self.actor = nn.Linear(h, 2 if self.agg else 2 * M)
            self.critic = nn.Sequential(act(), nn.Linear(h, 1))
        elif self.agg:
            self.enc = mlp(F + G, h, h, L, act)
            if cfg.arch == "attention":
                self.attn = nn.MultiheadAttention(h, cfg.heads, batch_first=True)
                self.norm = nn.LayerNorm(h)
            self.actor = mlp(2 * h + G, h, 2, 1, act)    # pooled set + global -> alpha, beta
            self.critic = mlp(2 * h + G, h, 1, 1, act)
        else:
            self.enc = mlp(F + G, h, h, L, act)
            if cfg.arch == "attention":
                self.attn = nn.MultiheadAttention(h, cfg.heads, batch_first=True)
                self.norm = nn.LayerNorm(h)
            self.actor = mlp(3 * h, h, 2, 1, act)       # [own, mean, max] -> alpha, beta
            self.critic = mlp(2 * h + G, h, 1, 1, act)
        # start close to "charge at init_mean of the feasible rate" (Beta mean a/(a+b))
        last = self.actor if isinstance(self.actor, nn.Linear) else self.actor[-1]
        a0, b0 = 4.0 * cfg.init_mean, 4.0 * (1 - cfg.init_mean)
        inv = lambda y: float(np.log(np.expm1(max(y - 1.0, 1e-3))))  # softplus^-1(y - 1)  # noqa: E731
        with torch.no_grad():
            last.weight.mul_(0.01)
            last.bias.view(-1, 2)[:, 0] = inv(1 + a0)
            last.bias.view(-1, 2)[:, 1] = inv(1 + b0)

    def forward(self, slots, mask, glob):
        B, M, F = slots.shape
        if self.arch == "mlp":
            z = self.body(torch.cat([slots.reshape(B, -1), glob], 1))
            ab = self.actor(z).view(B, -1, 2)
            v = self.critic(z).squeeze(-1)
        else:
            # only the occupied prefix of the slot array is processed (EVs take the
            # lowest free slot), outputs for the unused tail are padded
            occ = mask.any(0).nonzero()
            Me = int(occ.max()) + 1 if len(occ) else 1
            full_M = M
            slots, mask, M = slots[:, :Me], mask[:, :Me], Me
            x = torch.cat([slots, glob[:, None, :].expand(B, M, glob.shape[1])], -1)
            e = self.enc(x)
            if self.arch == "attention":
                kpm = ~mask
                kpm = kpm & ~(kpm.all(1, keepdim=True))       # avoid all-masked rows (NaN)
                a, _ = self.attn(e, e, e, key_padding_mask=kpm, need_weights=False)
                e = self.norm(e + a)
            m = mask[..., None].float()
            n = m.sum(1).clamp(min=1)
            mean = (e * m).sum(1) / n
            mx = torch.where(mask[..., None], e, torch.full_like(e, -1e4)).max(1).values
            mx = torch.where(mask.any(1, keepdim=True), mx, torch.zeros_like(mx))
            pooled = torch.cat([mean, mx], -1)
            v = self.critic(torch.cat([pooled, glob], -1)).squeeze(-1)
            if self.agg:
                ab = self.actor(torch.cat([pooled, glob], -1)).view(B, 1, 2)
                return (nn.functional.softplus(ab) + 1.0)[..., 0], (nn.functional.softplus(ab) + 1.0)[..., 1], v
            ab = self.actor(torch.cat([e, pooled[:, None, :].expand(B, M, -1)], -1))
            if M < full_M:
                ab = torch.cat([ab, torch.ones(B, full_M - M, 2)], 1)
        ab = nn.functional.softplus(ab) + 1.0
        return ab[..., 0], ab[..., 1], v

    def dist(self, slots, mask, glob):
        a, b, v = self(slots, mask, glob)
        return Beta(a, b), v


def _t(x):
    return torch.as_tensor(x)


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean, self.var, self.count = np.zeros(shape), np.ones(shape), 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot


class PPO:
    def __init__(self, cfg: PPOConfig, scenarios, M=48):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.scenarios = scenarios if isinstance(scenarios, list) else [scenarios]
        probe = self.scenarios[0].sim("train", episodes=self.scenarios[0].train[:1])
        F, G = probe.SLOT_FEATS, probe.GLOB_FEATS
        self.M = M
        self.net = Policy(M, F, G, cfg)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.obs_rms = RunningMeanStd((G,))
        self.ret_rms = RunningMeanStd()
        torch.set_num_threads(cfg.threads)

    # one simulator per site so every env in the batch shares p_cap
    def _sample_train(self, sc, k):
        idx = self.rng.choice(len(sc.train), size=k, replace=len(sc.train) < k)
        eps = [sc.train[i] for i in idx]
        if self.cfg.price_aug:
            from scenario import price_bank
            bank = price_bank("train")
            eps = [replace(e, price=bank[self.rng.integers(len(bank))]) for e in eps]
        return eps

    def _train_noise(self):
        return float(self.rng.uniform(0, self.cfg.dep_noise_train)) if self.cfg.dep_noise_train > 0 else 0.0

    def _new_sims(self):
        n = self.cfg.n_envs
        per = [n // len(self.scenarios) + (i < n % len(self.scenarios)) for i in range(len(self.scenarios))]
        sims = []
        for sc, k in zip(self.scenarios, per):
            if k == 0:
                continue
            s = self.make_sim(sc, "train", episodes=self._sample_train(sc, k),
                              seed=int(self.rng.integers(1 << 30)), dep_noise_h=self._train_noise())
            s.rscale = 1.0 / (s.cap[:, 0] * 0.25 * self.cfg.reward_scale)
            s.sc = sc
            sims.append(s)
        return sims

    def make_sim(self, sc, split, episodes=None, seed=0, dep_noise_h=0.0):
        sf = "guard+llf" if self.cfg.action == "aggregate" else self.cfg.safety
        return sc.sim(split, seed=seed, episodes=episodes, dep_noise_h=dep_noise_h,
                      projection="proportional" if sf == "none" else "llf", guard=sf == "guard+llf")

    def _amask(self, mask):
        """Which action dimensions count in the log-probability."""
        if self.cfg.action == "aggregate":
            return torch.ones(mask.shape[0], 1, dtype=torch.bool)
        return mask

    def _step(self, s, a):
        if self.cfg.action == "aggregate":
            return s.step_aggregate(a[:, 0])
        return s.step(a)

    def _norm(self, glob, update=False):
        if not self.cfg.norm_obs:
            return glob
        if update:
            self.obs_rms.update(glob)
        return np.clip((glob - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10, 10).astype(np.float32)

    @torch.no_grad()
    def act(self, slots, mask, glob, deterministic=True):
        glob = self._norm(glob)
        d, _ = self.net.dist(_t(slots), _t(mask), _t(glob))
        a = d.mean if deterministic else d.sample()
        return a.numpy()

    def learn(self, log_every=10, callback=None):
        cfg = self.cfg
        T = cfg.rollout_len
        sims = self._new_sims()
        obs = [s.reset() for s in sims]
        ret_acc = np.zeros(cfg.n_envs)
        n_updates = max(1, cfg.total_steps // (T * cfg.n_envs * cfg.action_repeat))  # simulator steps
        hist = []
        t0 = time.time()
        for u in range(n_updates):
            if cfg.lr_anneal:
                for g in self.opt.param_groups:
                    g["lr"] = cfg.lr * (1 - u / n_updates)
            buf = {k: [] for k in ["s", "m", "g", "a", "lp", "r", "v", "d"]}
            for _ in range(T):
                S = np.concatenate([o[0] for o in obs]); Mk = np.concatenate([o[1] for o in obs])
                Gl = self._norm(np.concatenate([o[2] for o in obs]), update=True)
                with torch.no_grad():
                    dist, v = self.net.dist(_t(S), _t(Mk), _t(Gl))
                    a = dist.sample().clamp(1e-4, 1 - 1e-4)
                    lp = (dist.log_prob(a) * self._amask(_t(Mk))).sum(1)
                an = a.numpy()
                rs, ds, k = [], [], 0
                new_obs = []
                for j, s in enumerate(sims):
                    B = s.B
                    lam = s.cfg.lam
                    d0, u0 = s.delivered.copy(), s.unmet.copy()
                    phi0 = -lam * s.infeasible_energy()
                    r = 0.0
                    for _rep in range(cfg.action_repeat):
                        ri, done = self._step(s, an[k:k + B])
                        r = r + ri
                    if cfg.center_price:
                        # J = sum (p-1) e + (lam-1) U + D, and D is exogenous
                        r = r + (s.delivered - d0) + (s.unmet - u0)
                        lam = lam - 1.0
                        phi0 = phi0 * lam / s.cfg.lam
                    if cfg.shaping:
                        phi1 = 0.0 if done else -lam * s.infeasible_energy()
                        r = r + cfg.gamma * phi1 - phi0
                    k += B
                    rs.append(r * s.rscale)
                    ds.append(np.full(B, done))
                    if done:   # replace finished episodes with fresh weeks
                        sims[j] = self._new_sims_one(j, sims[j])
                        new_obs.append(sims[j].reset())
                    else:
                        new_obs.append(s.obs())
                obs = new_obs
                R, D = np.concatenate(rs), np.concatenate(ds)
                if cfg.norm_reward:
                    ret_acc = ret_acc * cfg.gamma + R
                    self.ret_rms.update(ret_acc)
                    ret_acc[D] = 0.0
                    R = R / np.sqrt(self.ret_rms.var + 1e-8)
                for key, val in zip(buf, [S, Mk, Gl, an, lp.numpy(), R, v.numpy(), D]):
                    buf[key].append(val)
            S = np.concatenate([o[0] for o in obs]); Mk = np.concatenate([o[1] for o in obs])
            Gl = self._norm(np.concatenate([o[2] for o in obs]))
            with torch.no_grad():
                _, v_last = self.net.dist(_t(S), _t(Mk), _t(Gl))
            stats = self._update(buf, v_last.numpy())
            hist.append(dict(update=u, steps=(u + 1) * T * cfg.n_envs * cfg.action_repeat,
                             reward=float(np.mean(buf["r"])), **stats, time=time.time() - t0))
            if log_every and u % log_every == 0:
                h = hist[-1]
                print(f"[{cfg.arch}] u={u}/{n_updates} steps={h['steps']} r={h['reward']:.4f} "
                      f"pl={h['pi_loss']:.3f} vl={h['v_loss']:.3f} ent={h['entropy']:.3f} {h['time']:.0f}s", flush=True)
            if callback is not None and callback(self, u, n_updates) is False:
                break
        return hist

    def _new_sims_one(self, j, old):
        sc = old.sc
        s = self.make_sim(sc, "train", episodes=self._sample_train(sc, old.B), seed=int(self.rng.integers(1 << 30)),
                          dep_noise_h=self._train_noise())
        s.rscale, s.sc = 1.0 / (s.cap[:, 0] * 0.25 * self.cfg.reward_scale), sc
        return s

    def _update(self, buf, v_last):
        cfg = self.cfg
        r = np.stack(buf["r"]); v = np.stack(buf["v"]); d = np.stack(buf["d"]).astype(float)
        T, N = r.shape
        adv = np.zeros_like(r)
        last = 0
        for t in reversed(range(T)):
            nv = v_last if t == T - 1 else v[t + 1]
            nonterm = 1.0 - d[t]
            delta = r[t] + cfg.gamma * nv * nonterm - v[t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + v
        flat = lambda k: np.concatenate(buf[k])  # noqa: E731
        S, Mk, G, A, LP = (_t(flat(k)) for k in ["s", "m", "g", "a", "lp"])
        ADV, RET = _t(adv.reshape(-1)).float(), _t(ret.reshape(-1)).float()
        n = len(ADV)
        mb = n // cfg.minibatches
        stats = dict(pi_loss=0.0, v_loss=0.0, entropy=0.0, kl=0.0)
        cnt = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for i in range(cfg.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                dist, val = self.net.dist(S[idx], Mk[idx], G[idx])
                m = self._amask(Mk[idx]).float()
                lp = (dist.log_prob(A[idx]) * m).sum(1)
                ent = ((dist.entropy() * m).sum(1) / m.sum(1).clamp(min=1)).mean()
                ratio = torch.exp(lp - LP[idx])
                a = ADV[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pl = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vl = 0.5 * ((val - RET[idx]) ** 2).mean()
                loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["kl"] += float(((ratio - 1) - torch.log(ratio)).mean())
                stats["pi_loss"] += pl.item(); stats["v_loss"] += vl.item(); stats["entropy"] += ent.item()
                cnt += 1
        return {k: v / cnt for k, v in stats.items()}

    # ---- evaluation -------------------------------------------------------------
    def evaluate(self, sc, split="val", episodes=None, seed=0, dep_noise_h=0.0):
        sim = self.make_sim(sc, split, episodes=episodes, seed=seed, dep_noise_h=dep_noise_h)
        o = sim.reset()
        done = False
        while not done:
            a = self.act(*o, deterministic=True)
            for _rep in range(self.cfg.action_repeat):
                _, done = self._step(sim, a)
            o = sim.obs()
        return sim.metrics()

    def save(self, path):
        torch.save({"cfg": asdict(self.cfg), "state": self.net.state_dict(),
                    "obs_rms": (self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count)}, path)

    @staticmethod
    def load(path, scenarios):
        ck = torch.load(path, weights_only=False)
        agent = PPO(PPOConfig(**ck["cfg"]), scenarios)
        agent.net.load_state_dict(ck["state"])
        agent.obs_rms.mean, agent.obs_rms.var, agent.obs_rms.count = ck["obs_rms"]
        return agent
