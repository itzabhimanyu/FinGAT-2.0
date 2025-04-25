import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.nn import GATv2Conv, GraphNorm
from torch_geometric.utils import from_networkx

# ——— DEVICE ——————————————————————————————————————————————————
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——— 1. DATA LOADING & FEATURE ENGINEERING —————————————————————————
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
        for c in ["Open","High","Low"]:
            d[f"pct_{c}"] = d[c].pct_change()
        for w in [5,10,15,20]:
            d[f"ma{w}"] = d["Close"].rolling(w).mean()
        d.dropna(inplace=True)
        data[s] = d
    return data

def normalize_close_price(data):
    for df in data.values():
        df["norm_close"] = StandardScaler().fit_transform(df[["Close"]])
    return data

# ——— 2. DAILY RETURNS —————————————————————————————————————————
def compute_daily_returns(data):
    dr = {}
    for s, df in data.items():
        prev, curr = df["Close"].iloc[-2].item(), df["Close"].iloc[-1].item()
        dr[s] = (curr - prev)/prev if prev != 0 else 0.0
    return dr

# ——— 3. GRAPH ———————————————————————————————————————————————————
def build_windowed_graph(nifty, data, window=30, corr_th=0.5):
    windowed = {s: df.tail(window) for s, df in data.items() if len(df)>=window}
    syms     = [s for s in nifty["Symbol"] if s in windowed]
    idxs     = sorted(set.intersection(*(set(windowed[s].index) for s in syms)))
    rets     = pd.DataFrame(
        {s: windowed[s].loc[idxs, "ret"].values for s in syms},
        index=idxs
    )
    corr     = rets.corr()

    G = nx.Graph()
    # industry clique edges
    for _, grp in nifty.groupby("Industry")["Symbol"]:
        members = [s for s in grp if s in windowed]
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                G.add_edge(members[i], members[j], weight=1.0)
    # correlation edges
    for i in corr.index:
        for j in corr.columns:
            if i!=j and corr.at[i,j]>corr_th:
                G.add_edge(i, j, weight=float(corr.at[i,j]))
    return G

# ——— 4. SEQUENCE ENCODER ———————————————————————————————————————
class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hid=64, layers=1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, layers, batch_first=True)
    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]

# ——— 5. PYG CONVERSION ————————————————————————————————————————
def convert_to_pyg(G, data, seq_enc, seq_len, feat_cols):
    nodes, embs = list(G.nodes()), []
    for s in nodes:
        df = data.get(s)
        if df is None or len(df)<seq_len:
            embs.append(torch.zeros(seq_enc.gru.hidden_size, device=device))
        else:
            seq = torch.tensor(
                df[feat_cols].values[-seq_len:], device=device, dtype=torch.float
            ).unsqueeze(0)
            with torch.no_grad():
                embs.append(seq_enc(seq).squeeze(0))
    X = torch.stack(embs)
    X = StandardScaler().fit_transform(X.cpu().numpy())
    X = torch.tensor(X, dtype=torch.float, device=device)

    pyg = from_networkx(G)
    pyg.edge_index = pyg.edge_index.to(device)
    pyg.x = X

    N = X.size(0)
    idx = torch.arange(N, device=device)
    pyg.train_mask = idx < int(0.6*N)
    pyg.val_mask   = (idx>=int(0.6*N)) & (idx<int(0.8*N))
    pyg.test_mask  = idx >= int(0.8*N)
    return pyg, nodes

# ——— 6. ENHANCED FinGAT MODEL ————————————————————————————————————
class EnhancedFinGAT(nn.Module):
    def __init__(self, hid, layers=4, heads=4, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            conv = GATv2Conv(hid, hid//heads, heads=heads, dropout=dropout, concat=True)
            self.convs.append(conv)
            self.norms.append(GraphNorm(hid))
        self.skip_proj = nn.Linear(hid, hid)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hid,1)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, ei)
            h = F.elu(norm(h))
            x = x + self.dropout(self.skip_proj(h))
        return self.fc(x).squeeze(-1)

# ——— 7. TRAINING (MSE + hinge‐rank) —————————————————————————————————
def train_model(model, pyg, epochs=100, lr=5e-4, α=0.7, β=0.3):
    opt    = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse_fn = nn.MSELoss()
    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        pred = model(pyg)
        true = pyg.y
        mse  = mse_fn(pred[pyg.train_mask], true[pyg.train_mask])
        dp   = pred.unsqueeze(1) - pred.unsqueeze(0)
        dt   = true.unsqueeze(1) - true.unsqueeze(0)
        rankl = F.relu(1.0 - (dt>0).float()*dp)[pyg.train_mask].mean()
        loss  = α*mse + β*rankl
        loss.backward(); opt.step()
        if ep%20==0:
            print(f"Epoch {ep}/{epochs} Loss={loss:.4f} (MSE={mse:.4f}, Rank={rankl:.4f})")
    return model

# ——— 8. METRICS & TOP‐K PRINT —————————————————————————————————————
def precision_at_k(t,p,k):
    return len(set(np.argsort(p)[::-1][:k]) & set(np.argsort(t)[::-1][:k]))/k
def mrr_at_k(t,p,k):
    op = np.argsort(p)[::-1]; rm={i:idx+1 for idx,i in enumerate(op)}
    return sum(1.0/rm[i] for i in np.argsort(t)[::-1][:k])/k
def irr_at_k(t,p,k):
    ot, op = np.argsort(t)[::-1][:k], np.argsort(p)[::-1][:k]
    return t[ot].sum() - p[op].sum()

def print_topk(true, pred, nodes, k):
    prec = precision_at_k(true, pred, k)
    mrr  = mrr_at_k(true, pred, k)
    irr  = irr_at_k(true, pred, k)
    print(f"\n--- K={k} --- Prec@{k}={prec:.4f}, MRR@{k}={mrr:.4f}, IRR@{k}={irr:.4f}")
    idxs = np.argsort(pred)[::-1][:k]
    print("| Rank | Ticker | PredRet | TrueRet |")
    print("|-|-|-|-|")
    for i,j in enumerate(idxs,1):
        print(f"| {i} | {nodes[j]} | {pred[j]:.4f} | {true[j]:.4f} |")

# ——— 9. MAIN —————————————————————————————————————————————————————
def main():
    nifty   = pd.read_csv("ind_nifty500list_filtered_final.csv").dropna(subset=["Industry"])
    syms    = nifty["Symbol"].tolist()
    raw     = load_stock_data(syms)
    data    = normalize_close_price(feature_engineer(raw, syms))
    dr      = compute_daily_returns(data)

    feat_cols = ["ret","pct_Open","pct_High","pct_Low"] + [f"ma{w}" for w in [5,10,15,20]] + ["norm_close"]
    seq_enc   = TemporalEncoder(len(feat_cols), hid=64).to(device)

    G, nodes = build_windowed_graph(nifty, data, window=30, corr_th=0.5), None
    pyg,nodes= convert_to_pyg(G, data, seq_enc, seq_len=30, feat_cols=feat_cols)
    pyg.y     = torch.tensor([dr[s] for s in nodes], dtype=torch.float, device=device)

    model = EnhancedFinGAT(hid=64, layers=4, heads=4, dropout=0.2).to(device)
    model = train_model(model, pyg, epochs=100, lr=5e-4, α=0.7, β=0.3)

    model.eval()
    with torch.no_grad():
        pred = model(pyg).cpu().numpy()
    true = pyg.y.cpu().numpy()

    for k in [5,10,15,20]:
        print_topk(true, pred, nodes, k)

if __name__=="__main__":
    main()
