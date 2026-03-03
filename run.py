import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)


def collate_fn(batch):
    x_enc, x_mark_enc, x_dec, x_mark_dec, y_true, time_pred = zip(*batch)
    return (torch.stack(x_enc), torch.stack(x_mark_enc), torch.stack(x_dec),
            torch.stack(x_mark_dec), torch.stack(y_true), list(time_pred))


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        return self.dropout(self.tokenConv(x))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.05):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_output, _ = self.self_attention(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.05):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        attn_output, _ = self.self_attention(x, x, x, attn_mask=self_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        attn_output, _ = self.cross_attention(x, enc_out, enc_out, attn_mask=cross_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, e_layers, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, d_layers, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(d_layers)
        ])

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, cross_mask)
        return x


class AdaptiveGateFusion(nn.Module):
    def __init__(self, d_model, gate_hidden_dim=256):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, local_feat, global_feat):
        concat = torch.cat([local_feat, global_feat], dim=-1)
        gate_logits = self.gate_net(concat)          # [B, L, 1]
        lambda_ = self.sigmoid(gate_logits)          # [B, L, 1] ∈ (0,1)
        return lambda_


class Transformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 d_model=512, n_heads=8, e_layers=2, d_layers=2, d_ff=2048,
                 dropout=0.05, lstm_hidden_dim=512, lstm_layers=1):
        super().__init__()
        self.enc_embedding = TokenEmbedding(enc_in, d_model)
        self.dec_embedding = TokenEmbedding(dec_in, d_model)
        self.pos_enc = PositionalEmbedding(d_model)
        self.pos_dec = PositionalEmbedding(d_model)


        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)


        self.transformer_encoder = TransformerEncoder(d_model, n_heads, e_layers, d_ff, dropout)


        self.fusion_norm = nn.LayerNorm(d_model)


        self.fusion_proj = nn.Linear(lstm_hidden_dim, d_model)


        self.adaptive_gate = AdaptiveGateFusion(d_model)


        self.decoder = TransformerDecoder(d_model, n_heads, d_layers, d_ff, dropout)


        self.projection = nn.Linear(d_model, c_out)
        self.dropout = nn.Dropout(dropout)

        self.label_len = label_len
        self.out_len = out_len

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc) + self.pos_enc(x_enc)  # [B, seq_len, d_model]


        lstm_out, _ = self.lstm(enc_out)                           # [B, seq_len, lstm_hidden_dim]
        lstm_out = self.fusion_proj(lstm_out)                      # [B, seq_len, d_model]


        trans_out = self.transformer_encoder(enc_out, src_mask=enc_self_mask)


        lambda_ = self.adaptive_gate(lstm_out, trans_out)          # [B, seq_len, 1]


        fused = lambda_ * lstm_out + (1 - lambda_) * trans_out
        enc_out_fused = self.fusion_norm(fused)                    # [B, seq_len, d_model]


        dec_out = self.dec_embedding(x_dec) + self.pos_dec(x_dec)
        dec_out = self.decoder(dec_out, enc_out_fused, self_mask=dec_self_mask, cross_mask=dec_enc_mask)


        dec_out = self.projection(dec_out[:, -self.out_len:, :])
        return dec_out

def calculate_metrics(pred, true):
    pred = pred.flatten()
    true = true.flatten()
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / (true + 1e-8))) * 100
    smape = np.mean(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8)) * 100
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    ss_res = np.sum((true - pred) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "SMAPE": smape, "R2": r2}

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file '{file_path}' not found")
    print(f"Actual data rows: {df.shape[0]}")
    if df.shape[1] < 3:
        raise ValueError(f"Excel file must have at least 3 columns (time, feature(s), target), got {df.shape[1]}")
    time_col = df.iloc[:, 0].values
    try:
        time_col = pd.to_datetime(time_col).values
        print("Timestamp converted to datetime format")
    except:
        print("Warning: Could not convert time_col to datetime, using as-is")
        time_col = df.iloc[:, 0].values.astype(str)
    features = df.iloc[:, 1:-1].values
    target = df.iloc[:, -1].values

    if features.shape[1] == 0:
        raise ValueError("No feature columns found. Please check the Excel file format")

    try:
        features = features.astype(np.float32)
        target = target.astype(np.float32)
    except ValueError:
        raise ValueError("Feature and target columns must be numeric")

    data = np.hstack([features, target.reshape(-1, 1)])
    print(f"Data shape: {data.shape}, Feature columns: {data.shape[1]}")
    return data, time_col

class TimeSeriesDataset(Dataset):
    def __init__(self, data, time_col, seq_len, label_len, pred_len):
        self.data = data
        self.time_col = time_col
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        if len(data) < seq_len + pred_len:
            raise ValueError(f"Data length {len(data)} is insufficient for seq_len={seq_len} + pred_len={pred_len}")

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        x = self.data[s_begin:s_end, :]
        x_dec = self.data[r_begin:r_end, :]
        y_true = self.data[s_end:s_end + self.pred_len, -1:]
        x_mark = torch.zeros(self.seq_len, 1, dtype=torch.float32)
        x_dec_mark = torch.zeros(self.label_len + self.pred_len, 1, dtype=torch.float32)
        time_pred = self.time_col[s_end:s_end + self.pred_len]

        return (torch.tensor(x, dtype=torch.float32).clone().detach(),
                x_mark.clone().detach(),
                torch.tensor(x_dec, dtype=torch.float32).clone().detach(),
                x_dec_mark.clone().detach(),
                torch.tensor(y_true, dtype=torch.float32).clone().detach(),
                time_pred)

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, patience=3):
    model.train()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = "best_model.pth"

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_trues = []
        for batch in train_loader:
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_true, _ = batch
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_true = [b.to(device) for b in
                                                            [x_enc, x_mark_enc, x_dec, x_mark_dec, y_true]]
            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(output, y_true)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_predictions.append(output.detach().cpu().numpy())
            train_trues.append(y_true.detach().cpu().numpy())

        train_predictions = np.concatenate(train_predictions, axis=0)
        train_trues = np.concatenate(train_trues, axis=0)
        train_predictions = train_predictions[:, :, 0].flatten()
        train_trues = train_trues[:, :, 0].flatten()
        train_metrics = calculate_metrics(train_predictions, train_trues)

        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_trues = []
        with torch.no_grad():
            for batch in test_loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y_true, _ = batch
                x_enc, x_mark_enc, x_dec, x_mark_dec, y_true = [b.to(device) for b in
                                                                [x_enc, x_mark_enc, x_dec, x_mark_dec, y_true]]
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                loss = criterion(output, y_true)
                total_val_loss += loss.item()
                val_predictions.append(output.cpu().numpy())
                val_trues.append(y_true.cpu().numpy())

        val_predictions = np.concatenate(val_predictions, axis=0)
        val_trues = np.concatenate(val_trues, axis=0)
        val_predictions = val_predictions[:, :, 0].flatten()
        val_trues = val_trues[:, :, 0].flatten()
        val_metrics = calculate_metrics(val_predictions, val_trues)
        avg_val_loss = total_val_loss / len(test_loader)

        train_metrics_str = ", ".join([f"Train {metric}: {value:.4f}" for metric, value in train_metrics.items()])
        val_metrics_str = ", ".join([f"Val {metric}: {value:.4f}" for metric, value in val_metrics.items()])
        print(f"Epoch {epoch + 1}, Train Loss: {total_train_loss / len(train_loader):.4f}, {train_metrics_str}")
        print(f"Validation Loss: {avg_val_loss:.4f}, {val_metrics_str}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model

def main(file_path):
    seq_len = 96
    label_len = 48
    pred_len = 24
    batch_size = 32
    epochs = 100
    patience = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, time_col = load_data(file_path)
    df = pd.read_csv(file_path)
    if len(data) < 2000:
        print(f"Warning: Data length is {len(data)} rows, but you indicated over 2000 rows. Please verify.")

    min_data_len = seq_len + pred_len

    if len(data) < min_data_len:
        suggested_seq_len = max(12, len(data) // 2)
        suggested_pred_len = max(6, len(data) // 4)
        raise ValueError(
            f"Data length {len(data)} is insufficient for seq_len={seq_len} + pred_len={pred_len}. "
            f"Suggest setting seq_len={suggested_seq_len}, pred_len={suggested_pred_len}"
        )

    dataset = TimeSeriesDataset(data, time_col, seq_len, label_len, pred_len)
    total_samples = len(dataset)
    print(f"Total dataset samples:  {total_samples}")
    if total_samples < 128:
        raise ValueError(
            f"Dataset size {total_samples} is too small for training and testing with batch_size={batch_size}. "
            "Suggest reducing batch_size or increasing data length.")

    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    print(f"Train samples: {train_size}, Test samples: {test_size}")
    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, total_samples))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = Transformer(
        enc_in=data.shape[1],
        dec_in=data.shape[1],
        c_out=1,
        seq_len=seq_len,
        label_len=label_len,
        out_len=pred_len,
        d_model=512,
        n_heads=8,
        e_layers=4,
        d_layers=2,
        d_ff=2048,
        dropout=0.05,
        lstm_hidden_dim=512,
        lstm_layers=1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Starting training...")
    model = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, patience)


    model.eval()
    predictions = []
    trues = []
    times = []
    full_indices = list(range(total_samples))
    full_dataset = torch.utils.data.Subset(dataset, full_indices)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in full_loader:
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_true, time_pred = batch
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_true = [b.to(device) for b in
                                                            [x_enc, x_mark_enc, x_dec, x_mark_dec, y_true]]
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            predictions.append(output.cpu().numpy())
            trues.append(y_true.cpu().numpy())
            times.append(np.concatenate(time_pred, axis=0))

    predictions = np.concatenate(predictions, axis=0)
    trues = np.concatenate(trues, axis=0)
    times = np.concatenate(times, axis=0)
    full_samples = predictions.shape[0]
    print(f"Full dataset samples: {full_samples}, Prediction rows: {full_samples * pred_len}")
    predictions = predictions[:, :, 0].reshape(-1)
    trues = trues[:, :, 0].reshape(-1)
    times = times.reshape(-1)

    results = pd.DataFrame({
        "Time": times,
        "True_Value": trues,
        "Prediction": predictions,
        "Residual": predictions - trues
    })
    results["Time"] = pd.to_datetime(results["Time"])
    results = results.groupby("Time").last().reset_index()
    results = results.sort_values("Time")


    model.eval()
    test_predictions = []
    test_trues = []
    with torch.no_grad():
        for batch in test_loader:
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_true, _ = batch
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_true = [b.to(device) for b in
                                                            [x_enc, x_mark_enc, x_dec, x_mark_dec, y_true]]
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            test_predictions.append(output.cpu().numpy())
            test_trues.append(y_true.cpu().numpy())

    test_predictions = np.concatenate(test_predictions, axis=0)
    test_trues = np.concatenate(test_trues, axis=0)
    test_predictions = test_predictions[:, :, 0].flatten()
    test_trues = test_trues[:, :, 0].flatten()
    metrics = calculate_metrics(test_predictions, test_trues)
    print("Test set evaluation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


    model.eval()
    with torch.no_grad():
        last_enc = data[-seq_len:, :]
        x_enc = torch.tensor(last_enc, dtype=torch.float32).unsqueeze(0).to(device)
        x_mark_enc = torch.zeros((1, seq_len, 1), dtype=torch.float32).to(device)

        if len(data) >= label_len:
            last_dec_known = data[-label_len:, :]
        else:
            last_dec_known = np.zeros((label_len, data.shape[1]), dtype=np.float32)
            last_dec_known[-len(data):, :] = data
        future_placeholder = np.zeros((pred_len, data.shape[1]), dtype=np.float32)
        future_placeholder[:, :] = last_dec_known[-1, :]
        last_dec = np.vstack([last_dec_known, future_placeholder])
        x_dec = torch.tensor(last_dec, dtype=torch.float32).unsqueeze(0).to(device)
        x_mark_dec = torch.zeros((1, label_len + pred_len, 1), dtype=torch.float32).to(device)

        next_pred = np.zeros(pred_len, dtype=np.float32)
        for i in range(pred_len):
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            next_val = output[0, i, 0]
            next_pred[i] = next_val
            x_dec[0, label_len + i, -1] = next_val

    if len(time_col) >= 2 and isinstance(time_col[-1], pd.Timestamp) and isinstance(time_col[-2], pd.Timestamp):
        delta = time_col[-1] - time_col[-2]
    else:
        delta = pd.Timedelta(days=1)
    last_time = pd.to_datetime(time_col[-1]) if not isinstance(time_col[-1], pd.Timestamp) else time_col[-1]
    future_times = [last_time + (i + 1) * delta for i in range(pred_len)]

    next_window_df = pd.DataFrame({
        "Time": future_times,
        "Next_Window_Prediction": next_pred
    })
    next_window_df = next_window_df.groupby("Time").last().reset_index()
    next_window_df = next_window_df.sort_values("Time")



    results.to_csv("results_full_dataset.csv", index=False)
    next_window_df.to_csv("results_next_window.csv", index=False)
    print(f"Full dataset result rows: {results.shape[0]}")
    print(f"Next window rows: {next_window_df.shape[0]}")

if __name__ == "__main__":
    file_path = "Data_BEA.csv"
    main(file_path)