#!/usr/bin/env python
import os
import pickle
import time
import cProfile, pstats
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.nn import GATv2Conv, GraphNorm
from torch_geometric.data import Data

# ——— DEVICE SETUP —————————————————————————————————————————————
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(" GPU:", torch.cuda.get_device_name(0))

# ——— 1. LOAD & FEATURE ENGINEERING —————————————————————————————
def load_stock_data(symbols, in_dir="data"):
    with open(os.path.join(in_dir, "all_data.pkl"), "rb") as f:
        return pickle.load(f)

def feature_engineer(all_data, symbols):
    data = {}
    for s in symbols:
        df = all_data.get(s)
        if df is None: continue
        d = df.copy()
        d["ret"] = d["Close"].pct_change()
        for c in ["Open", "High", "Low"]:
            d[f"pct_{c}"] = d[c].pct_change()
        for w in [5, 10, 15, 20]:
            d[f"ma{w}"] = d["Close"].rolling(w).mean()
        # additional features
        d["vol10"] = d["ret"].rolling(10).std()
        d["mom10"] = d["Close"].pct_change(10)
        d.dropna(inplace=True)
        data[s] = d
    return data

def normalize_close_price(data):
    scaler = StandardScaler()
    for df in data.values():
        df["norm_close"] = scaler.fit_transform(df[["Close"]])
    return data

# ——— 2. DAILY RETURNS —————————————————————————————————————————
def compute_daily_returns(data):
    return {s: (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]
            for s, df in data.items()}

# ——— 3. STATIC SECTOR GRAPH —————————————————————————————————————
def build_static_graph(nifty, symbols):
    G = nx.Graph()
    for _, grp in nifty.groupby("Industry")["Symbol"]:
        members = [s for s in grp if s in symbols]
        for u, v in combinations(members, 2):
            G.add_edge(u, v, weight=1.0, type=0)
    return G

# ——— 4. TIME-DECAYED CORRELATION EDGES —————————————————————————
def add_corr_edges(G_static, data, window=30, corr_th=0.5):
    windowed = {s: df.tail(window) for s, df in data.items() if len(df) >= window}
    syms = list(windowed.keys())
    if not syms:
        return G_static.copy()
    common = set(windowed[syms[0]].index)
    for s in syms[1:]: common &= set(windowed[s].index)
    common = sorted(common)
    rets = pd.DataFrame({s: windowed[s].loc[common, "ret"] for s in syms}, index=common)
    corr = rets.corr()
    G = G_static.copy()
    for u, v in combinations(syms, 2):
        w = corr.at[u, v]
        if w > corr_th:
            G.add_edge(u, v, weight=float(w), type=1)
    return G

# ——— 5. SEQUENCE ENCODER & PyG CONVERSION —————————————————————————
class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, batch_first=True)
    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]

cached_scaler = StandardScaler()

def convert_to_pyg(G, data, seq_enc, seq_len, feat_cols):
    valid_nodes = [s for s in G.nodes() if s in data and len(data[s]) >= seq_len]
    G_sub = G.subgraph(valid_nodes).copy()
    nodes = list(G_sub.nodes())
    if not nodes:
        empty = Data(x=torch.empty((0, seq_enc.gru.hidden_size), device=device))
        empty.train_mask = empty.val_mask = empty.test_mask = torch.zeros((0,), dtype=torch.bool, device=device)
        return empty, []
    idx_map = {s: i for i, s in enumerate(nodes)}
    embs = []
    for s in nodes:
        arr = data[s].loc[:, feat_cols].iloc[-seq_len:].values
        seq = torch.tensor(arr, dtype=torch.float, device=device).unsqueeze(0)
        with torch.no_grad(): embs.append(seq_enc(seq).squeeze(0))
    X = torch.tensor(cached_scaler.fit_transform(torch.stack(embs).cpu()), dtype=torch.float, device=device)
    rows, cols, wgt, typ = [], [], [], []
    for u, v, attr in G_sub.edges(data=True):
        ui, vi = idx_map[u], idx_map[v]
        for a, b in ((ui, vi), (vi, ui)):
            rows.append(a); cols.append(b)
            wgt.append(attr['weight']); typ.append(attr['type'])
    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
    edge_attr  = torch.tensor([wgt, typ], dtype=torch.float, device=device).t()
    d = Data(x=X, edge_index=edge_index, edge_attr=edge_attr)
    N = X.size(0)
    idxr = torch.arange(N, device=device)
    d.train_mask = idxr < int(0.6 * N)
    d.val_mask   = (idxr >= int(0.6 * N)) & (idxr < int(0.8 * N))
    d.test_mask  = idxr >= int(0.8 * N)
    return d, nodes

# ——— 6. ENHANCED FinGAT MODEL —————————————————————————————————————
class EnhancedFinGAT(nn.Module):
    def __init__(self, hid, layers=4, heads=4, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            GATv2Conv(hid, hid // heads, heads=heads, dropout=dropout, concat=True)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([GraphNorm(hid) for _ in range(layers)])
        self.skip  = nn.Linear(hid, hid)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(hid, 1)
    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            h = F.elu(norm(conv(x, ei)))
            x = x + self.drop(self.skip(h))
        return self.fc(x).squeeze(-1)

# ——— 7. ONLINE TRAINING LOOP —————————————————————————————————————
def online_train(model, pyg, epochs=5, lr=5e-4):
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
    loss_fn = nn.MSELoss()
    train_idx = pyg.train_mask.nonzero(as_tuple=False).squeeze()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(pyg)[train_idx]
        true = pyg.y.squeeze()[train_idx]
        loss_fn(pred, true).backward()
        opt.step()
        scheduler.step()
    model.eval()
    return model

# ——— 8. METRICS —————————————————————————————————————————————
import numpy as _np  # alias for clarity

def precision_at_k(t, p, k):
    # fraction of overlap between top-k predictions and top-k true
    top_p = _np.argsort(p)[::-1][:k]
    top_t = _np.argsort(t)[::-1][:k]
    return _np.intersect1d(top_p, top_t).size / k


def mrr_at_k(t, p, k):
    sorted_p = _np.argsort(p)[::-1]
    top_t = _np.argsort(t)[::-1][:k]
    ranks = []
    for idx in top_t:
        pos = _np.where(sorted_p == idx)[0]
        if pos.size > 0:
            ranks.append(1.0 / (pos[0] + 1))
    return _np.mean(ranks) if ranks else 0.0

def irr_at_k(t, p, k):
    top = _np.argsort(p)[::-1][:k]
    return _np.sum(t[top]) - _np.sum(p[top])

# ——— 9. DAILY PREDICTIONS & SAVE ———————————————————————————————————
def predict_and_save_daily(data, nifty, seq_enc, feat_cols,
                             start='2025-01-11', end='2025-03-22', window=30, corr_th=0.5, seq_len=30):
    os.makedirs('daily_preds', exist_ok=True)
    symbols = list(data.keys())
    static_G = build_static_graph(nifty, symbols)
    model    = EnhancedFinGAT(hid=64).to(device)
    for current_date in pd.bdate_range(start, end):
        subset = {s: df[df.index <= current_date] for s, df in data.items()}
        G_day = add_corr_edges(static_G, subset, window, corr_th)
        pyg_day, nodes_day = convert_to_pyg(G_day, subset, seq_enc, seq_len, feat_cols)
        if not nodes_day: continue
        pyg_day.y = torch.tensor([compute_daily_returns(subset)[s] for s in nodes_day], dtype=torch.float, device=device).unsqueeze(1)
        model = online_train(model, pyg_day)
        with torch.no_grad(): preds = model(pyg_day).cpu().detach().numpy()
        df_out = pd.DataFrame({'Symbol': nodes_day, 'PredictedReturn': preds})
        df_out['Rank'] = df_out['PredictedReturn'].rank(ascending=False, method='first').astype(int)
        df_out.sort_values('Rank', inplace=True)
        date_str = current_date.strftime('%Y %m %d')
        df_out.to_csv(f'daily_preds/Predicted {date_str}.csv', index=False)

# ——— 10. MAIN TRAIN/VAL/TEST SPLIT & EXECUTION —————————————————————————

def main():  # Split/Train/Val/Test and daily predictions
    nifty = pd.read_csv("ind_nifty500list_filtered_final.csv").dropna(subset=["Industry"])
    symbols = nifty['Symbol'].tolist()
    raw = load_stock_data(symbols)
    data = normalize_close_price(feature_engineer(raw, symbols))
    feat_cols = ['ret','pct_Open','pct_High','pct_Low'] + [f'ma{w}' for w in [5,10,15,20]] + ['norm_close']

    # Split dates
    all_dates = sorted({d for df in data.values() for d in df.index})
    bdates = pd.bdate_range(all_dates[0], all_dates[-1])
    dates = [d for d in bdates if d in all_dates]
    n = len(dates)
    n60 = int(0.6 * n)
    n20 = int(0.2 * n)
    train_dates = dates[:n60]
    val_dates   = dates[n60:n60+n20]
    test_dates  = dates[n60+n20:]
    print(f"Dates split → Train: {len(train_dates)}, Val: {len(val_dates)}, Test: {len(test_dates)}")

    # Initialize model & encoder
    seq_enc = TemporalEncoder(len(feat_cols), hid=64).to(device)
    model = EnhancedFinGAT(hid=64).to(device)
    
    # Online train
    for d in tqdm(train_dates, desc="Training"):
        subset = {s: df[df.index <= d] for s, df in data.items()}
        G = add_corr_edges(build_static_graph(nifty, symbols), subset)
        pyg, nodes = convert_to_pyg(G, subset, seq_enc, 30, feat_cols)
        train_idx_tensor = pyg.train_mask.nonzero(as_tuple=False).squeeze()
        if train_idx_tensor.numel() == 0:
            continue
        train_idx = train_idx_tensor.tolist() if train_idx_tensor.ndim > 0 else [int(train_idx_tensor)]
        returns = compute_daily_returns(subset)
        y_train = [returns[nodes[i]] for i in train_idx]
        pyg.y = torch.tensor(y_train, dtype=torch.float, device=device).unsqueeze(1)
        model = online_train(model, pyg)

    def evaluate_dates(dates_list, name):
        results = []
        for d in tqdm(dates_list, desc=name):
            subset = {s: df[df.index <= d] for s, df in data.items()}
            G = add_corr_edges(build_static_graph(nifty, symbols), subset)
            pyg, nodes = convert_to_pyg(G, subset, seq_enc, 30, feat_cols)
            if not nodes: continue
            returns = compute_daily_returns(subset)
            pyg.y = torch.tensor([returns[s] for s in nodes], dtype=torch.float, device=device)
            with torch.no_grad(): preds = model(pyg).cpu().numpy()
            truths = pyg.y.cpu().numpy().squeeze()
            results.append({k: (precision_at_k(truths, preds, k), mrr_at_k(truths, preds, k)) for k in [5,10,15,20]})
        avg = {k: (np.mean([r[k][0] for r in results]), np.mean([r[k][1] for r in results])) for k in [5,10,15,20]}
        print(f"\n=== {name} Metrics ===")
        for k, (p, m) in avg.items(): print(f"K={k}: Prec={p:.4f}, MRR={m:.4f}")

    evaluate_dates(val_dates, "Validation")
    evaluate_dates(test_dates, "Test")

    seq_enc = TemporalEncoder(len(feat_cols), hid=64).to(device)
    predict_and_save_daily(data, nifty, seq_enc, feat_cols)

if __name__ == "__main__":
    main()
