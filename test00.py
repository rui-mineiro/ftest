# reinforce_backtest_sma.py
import math, numpy as np, pandas as pd, torch, torch.nn as nn, yfinance as yf

torch.manual_seed(0)
device = "cpu"

# ------------------ data ------------------
ticker = "AAPL"
px = yf.download(ticker, start="2024-01-01", end="2025-01-01", progress=False, auto_adjust=False)["Adj Close"].dropna()
ret = px.pct_change().fillna(0.0)
split = int(len(px) * 0.7)
idx_tr, idx_te = px.index[:split], px.index[split:]

# ------------------ backtest ------------------
def sma_metrics(px, n_fast, n_slow, thr, idx, tcost=1e-3):
    # 1) Price as Series
    if isinstance(px, pd.DataFrame):
        px = px.squeeze("columns")  # single column -> Series

    # 2) Signals as Series
    sf = px.rolling(n_fast).mean()
    ss = px.rolling(n_slow).mean()
    spread = (sf - ss) / ss
    sig = spread.reindex(idx)
    if isinstance(sig, pd.DataFrame):
        sig = sig.squeeze("columns")

    # 3) Position as Series
    pos = (sig.gt(thr).astype(float) - sig.lt(-thr).astype(float))
    pos = pos.shift(1).fillna(0.0)

    # 4) Returns as Series
    r = px.pct_change().reindex(idx).fillna(0.0)

    # 5) PnL as Series (drop NaNs/Infs)
    trades = pos.diff().abs().fillna(pos.abs())
    pnl = pos.mul(r, fill_value=0) - tcost * trades
    pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()

    sharpe_val = pnl.mean() / (pnl.std(ddof=0) + 1e-3)
    cumret_val = (1.0 + pnl).prod() - 1.0
    turn_val   = trades.mean()

    return float(sharpe_val), float(cumret_val), float(turn_val)


# map raw [-3,3] -> parameter ranges
def to_unit(z):  # [-3,3] -> [0,1]
    return float(np.clip((z + 3.0) / 6.0, 0.0, 1.0))

def map_param(z, lo, hi, as_int=False):
    u = to_unit(z)
    x = lo + u * (hi - lo)
    return int(round(x)) if as_int else float(x)

# reward on TRAIN
def reward_from_x(x_vec: np.ndarray) -> float:
    n_fast = map_param(x_vec[0], 5, 60, as_int=True)
    n_slow = map_param(x_vec[1], 20, 250, as_int=True)
    thr    = map_param(x_vec[2], 0.0, 0.02, as_int=False)
    sh, _, _ = sma_metrics(px, n_fast, n_slow, thr, idx_tr)
    return sh

# ------------------ policy ------------------
class Policy(nn.Module):
    def __init__(self, d=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, d)
        self.log_std = nn.Linear(hidden, d)
    def forward(self, batch: int):
        z = torch.randn(batch, 4, device=device)
        h = self.net(z)
        mu = self.mu(h)
        logstd = torch.clamp(self.log_std(h), -4, 1)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()                   # raw in R^d (later mapped to params)
        logp = dist.log_prob(x).sum(-1)
        ent = dist.entropy().sum(-1)
        return x, logp, ent

# vectorized reward wrapper
def f_batch(X: torch.Tensor) -> torch.Tensor:
    out = []
    for row in X.detach().cpu().numpy():
        out.append(reward_from_x(row))
    return torch.tensor(out, dtype=torch.float32, device=device)

# ------------------ train (REINFORCE) ------------------
def train_reinforce(steps=50, batch=51, lr=3e-3, ent_coef=1e-3, clip_grad=1.0):
    pi = Policy(d=3).to(device)
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    baseline = None
    alpha = 0.05
    best_r, best_x = -math.inf, None

    for t in range(1, steps + 1):
        X, logp, ent = pi(batch)
        r = f_batch(X)                           # Sharpes on train
        with torch.no_grad():
            baseline = r.mean() if baseline is None else (1 - alpha) * baseline + alpha * r.mean()
        adv = (r - baseline).detach()
        loss = -(adv * logp).mean() - ent_coef * ent.mean()

        opt.zero_grad(); loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(pi.parameters(), clip_grad)
        opt.step()

        with torch.no_grad():
            i = torch.argmax(r)
            if r[i] > best_r:
                best_r, best_x = float(r[i]), X[i].clone()

        if t % 250 == 0:
            print(f"iter {t:4d}  best train Sharpe ~ {best_r:.4f}")

    return best_x.detach().cpu().numpy(), best_r

best_x, best_train_sh = train_reinforce()

# decode best params
n_fast = map_param(best_x[0], 5, 60, as_int=True)
n_slow = map_param(best_x[1], 20, 250, as_int=True)
thr    = map_param(best_x[2], 0.0, 0.02, as_int=False)

tr_sh, tr_cr, tr_to = sma_metrics(px, n_fast, n_slow, thr, idx_tr)
te_sh, te_cr, te_to = sma_metrics(px, n_fast, n_slow, thr, idx_te)

print("\nBest parameters (decoded):")
print({"n_fast": n_fast, "n_slow": n_slow, "thr": round(thr, 5)})
print(f"Train: Sharpe {tr_sh:.3f}  CumRet {tr_cr:.3f}  Turnover {tr_to:.4f}")
print(f"Test : Sharpe {te_sh:.3f}  CumRet {te_cr:.3f}  Turnover {te_to:.4f}")
