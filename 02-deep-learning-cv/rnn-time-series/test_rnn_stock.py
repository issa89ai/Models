import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

COMPANY = 'AMZN'
START = '2015-01-01'
END = '2024-03-01'
LOOKBACK = 5
BATCH_SIZE = 8
NUM_EPOCHS = 50
HIDDEN_SIZE = 4
LR = 0.001


def get_company_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Close']]
    return data


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


print(f"Downloading {COMPANY} stock data ({START} to {END})...")
data = get_company_data(COMPANY, START, END)
print(f"Downloaded {data.shape[0]} days of data")

shifted_df = prepare_dataframe_for_lstm(data, LOOKBACK)
shifted_df_as_np = shifted_df.to_numpy()

scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]
X = dc(np.flip(X, axis=1))

split_index = int(len(X) * 0.95)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

X_train = X_train.reshape((-1, LOOKBACK, 1))
X_test = X_test.reshape((-1, LOOKBACK, 1))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = LSTM(1, HIDDEN_SIZE, 1).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def train_one_epoch():
    model.train(True)
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss / len(train_loader)


def validate_one_epoch():
    model.train(False)
    running_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
    return running_loss / len(test_loader)


for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch()
    val_loss = validate_one_epoch()
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

print(f"\nFinal: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

# Naive baseline: "tomorrow's price = today's price" (no learning at all)
# X_test[:, -1, 0] is the most recent day in each input window (yesterday, relative to the target)
naive_pred = X_test[:, -1, 0]
naive_mse = torch.mean((naive_pred - y_test.squeeze()) ** 2).item()
print(f"Naive baseline ('no change') MSE: {naive_mse:.6f}")
print(f"LSTM val_loss:                    {val_loss:.6f}")
print(f"LSTM beats naive baseline by: {(1 - val_loss / naive_mse) * 100:.1f}%")
