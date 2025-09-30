# pip install pandas numpy yfinance torch scikit-learn
import math
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# --------------------
# 1) Data
# --------------------
ticker = "AAPL"
start  = "2015-01-01"
end    = None  # today

df = yf.download(ticker, start=start, end=end, auto_adjust=True)  # OHLCV
df = df.rename(columns=str.title).dropna()

# Features: returns, rolling stats, RSI
def rsi(series: pd.Series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    rs = up.rolling(window).mean() / dn.rolling(window).mean()
    return 100 - 100 / (1 + rs)

df["ret1"]   = df["Close"].pct_change()
df["logret"] = np.log1p(df["ret1"])
df["vol10"]  = df["ret1"].rolling(10).std()
df["vol20"]  = df["ret1"].rolling(20).std()
df["mom5"]   = df["Close"].pct_change(5)
df["mom10"]  = df["Close"].pct_change(10)
df["rsi14"]  = rsi(df["Close"], 14) / 100.0
df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]
df["oc_gap"]    = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

# Label: next-day return sign (binary). Also store numeric next-day return for PnL.
df["y_ret_next"] = df["ret1"].shift(-1)
df["y_class"] = (df["y_ret_next"] > 0).astype(int)

feat_cols = ["logret","vol10","vol20","mom5","mom10","rsi14","hl_spread","oc_gap"]
df = df.dropna(subset=feat_cols + ["y_class","y_ret_next"]).copy()

X = df[feat_cols].values.astype(np.float32)
y = df["y_class"].values.astype(np.int64)
y_next_ret = df["y_ret_next"].values.astype(np.float32)
dates = df.index

# Train/val/test split by time
n = len(df)
i_train = int(n*0.7)
i_val   = int(n*0.85)

X_train, y_train = X[:i_train], y[:i_train]
X_val,   y_val   = X[i_train:i_val], y[i_train:i_val]
X_test,  y_test  = X[i_val:], y[i_val:]
ret_test        = y_next_ret[i_val:]
dates_test      = dates[i_val:]

# Scale features using only train fit
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# --------------------
# 2) PyTorch model
# --------------------
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # logits for 2 classes
        )
    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=X_train.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()

def to_tensor(a, dtype=torch.float32):
    return torch.from_numpy(a).to(device=device, dtype=dtype)

Xt = to_tensor(X_train)
yt = torch.from_numpy(y_train).to(device)
Xv = to_tensor(X_val)
yv = torch.from_numpy(y_val).to(device)

# Train
epochs = 30
batch = 128
for ep in range(epochs):
    model.train()
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), batch):
        idx = perm[i:i+batch]
        xb = Xt[idx]
        yb = yt[idx]
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

# --------------------
# 3) Validation accuracy
# --------------------
model.eval()
with torch.no_grad():
    val_logits = model(Xv)
    val_pred = val_logits.argmax(1).cpu().numpy()
print("Validation report:")
print(classification_report(y_val, val_pred, digits=3))

# --------------------
# 4) Backtest on test set
# --------------------
# Strategy: go long if P(long)>0.55, short if P(long)<0.45, else flat.
# Transaction cost: 10 bps per trade on notional of position change.
with torch.no_grad():
    probs = torch.softmax(model(to_tensor(X_test)), dim=1)[:,1].cpu().numpy()

long_thr  = 0.55
short_thr = 0.45
pos = np.zeros_like(probs, dtype=np.float32)
pos[probs >= long_thr]  = 1.0
pos[probs <= short_thr] = -1.0
# pos in {-1,0,1}; daily returns are next-day returns relative to signal day.

# Shift position to avoid look-ahead: use today's signal for tomorrow's return
pos_shifted = np.roll(pos, 1)
pos_shifted[0] = 0.0

# Costs on position changes
turnover = np.abs(np.diff(np.r_[0.0, pos_shifted]))  # position change magnitude
tcost_bps = 10  # 10 basis points
costs = turnover * (tcost_bps / 1e4)

# Strategy gross return
strat_gross = pos_shifted * ret_test
# Net return after costs (cost applied on the day of the change)
strat_net = strat_gross - costs

# Equity curve
equity = (1.0 + strat_net).cumprod()

# Metrics
def sharpe(returns, periods_per_year=252):
    mu = np.mean(returns)
    sd = np.std(returns, ddof=1)
    if sd == 0:
        return 0.0
    return (mu * periods_per_year) / (sd * math.sqrt(periods_per_year))

def max_drawdown(curve):
    peak = np.maximum.accumulate(curve)
    dd = (curve / peak) - 1.0
    return dd.min()

def cagr(curve, periods_per_year=252):
    n = len(curve)
    if n == 0:
        return 0.0
    years = n / periods_per_year
    return curve[-1]**(1/years) - 1

sr   = sharpe(strat_net)
mdd  = max_drawdown(equity)
cg   = cagr(equity)

print("\nBacktest period:", dates_test[0].date(), "to", dates_test[-1].date())
print(f"Test samples: {len(strat_net)} | Long %: {np.mean(pos==1):.2%} | Short %: {np.mean(pos==-1):.2%}")
print(f"Sharpe: {sr:.2f} | MaxDD: {mdd:.2%} | CAGR: {cg:.2%} | Final equity: {equity[-1]:.3f}")

# Optional: save CSV with results
out = pd.DataFrame({
    "Date": dates_test,
    "prob_long": probs,
    "position": pos_shifted,
    "ret_next": ret_test,
    "strat_net": strat_net,
    "equity": equity
}).set_index("Date")
out.to_csv("pytorch_backtest_results.csv")
print("Saved: pytorch_backtest_results.csv")
