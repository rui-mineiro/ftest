import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

# -------------------------------
# PARAMETERS
# -------------------------------
TICKER = "AAPL"
SEQ_LEN = 60
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -------------------------------
# DATA PREP
# -------------------------------

# Fetch historical data
def fetch_data(ticker, start="2015-01-01"):
    df = yf.download(ticker, start=start,auto_adjust=False)['Adj Close']
#    df = df[['Adj Close']]
    df['log_return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df.dropna(inplace=True)
    return df

df = fetch_data(TICKER)
returns = df['log_return'].values.reshape(-1, 1)

# Normalize returns
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

# Create sequences
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(seq_len, len(data) - 1):
        x = data[i-seq_len:i]
        y = data[i+1]  # predict next day's return
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(returns_scaled, SEQ_LEN)

# Split train/test
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# -------------------------------
# PyTorch Dataset
# -------------------------------
class ReturnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ReturnDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ReturnDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# LSTM Model
# -------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take the last output
        return self.fc(out)

model = LSTMModel().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# TRAINING
# -------------------------------
def train(model, loader, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(loader):.6f}")

train(model, train_loader, EPOCHS)

# -------------------------------
# EVALUATION
# -------------------------------
model.eval()
predictions, actuals = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        output = model(X_batch)
        predictions.append(output.cpu().numpy())
        actuals.append(y_batch.numpy())

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

# Inverse scale predictions
predictions = scaler.inverse_transform(predictions)
actuals = scaler.inverse_transform(actuals)

rmse = np.sqrt(mean_squared_error(actuals, predictions))
print(f"\nTest RMSE: {rmse:.6f}")

# -------------------------------
# PLOT
# -------------------------------
plt.figure(figsize=(12, 5))
plt.plot(actuals, label="Actual Returns")
plt.plot(predictions, label="Predicted Returns")
plt.title(f"{TICKER} - Actual vs Predicted Returns")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
