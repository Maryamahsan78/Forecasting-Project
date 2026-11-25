"""Core logic of the forecasting project: data helpers, the GRU model, baselines and multi-step forecasting.

Idea: instead of forecasting the raw values (which grow over time), we forecast the
seasonal log-change  d_t = log(y_t) - log(y_(t-12))  and convert back with
y_t = y_(t-12) * exp(d_t).  See the Day 6 notebook for the experiments behind this.
"""
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

K = 12                       # window length: the last 12 months
SEASON = 12                  # monthly data with a yearly pattern
MODEL_NAMES = ['GRU (on d)', 'Linear regression (on d)', 'Seasonal naive x avg growth', 'Seasonal naive', 'Naive (last month)']


class Net(nn.Module):
    """Many-to-one recurrent forecaster: 12 past (scaled) values -> the next (scaled) value."""
    def __init__(self, kind='gru', hidden=16, layers=2, dropout=0.1):
        super().__init__()
        cell = {'lstm': nn.LSTM, 'gru': nn.GRU}[kind]
        self.rnn = cell(1, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (batch, 12)
        out, _ = self.rnn(x[..., None])
        return self.fc(out[:, -1]).squeeze(1)


# ----------------------------------------------------------------------------- data
def load_airline(path='data/airline_passengers.csv'):
    df = pd.read_csv(path, parse_dates=['date'])
    return pd.Series(df['passengers'].astype(float).values, index=pd.DatetimeIndex(df['date']).to_period('M').to_timestamp(), name='passengers')


def parse_series(df, date_col, value_col):
    """Turn an uploaded table into a clean monthly series. Raises ValueError with a readable message."""
    s = df[[date_col, value_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors='coerce'); s[value_col] = pd.to_numeric(s[value_col], errors='coerce')
    if s[date_col].isna().any(): raise ValueError(f'Some values in "{date_col}" are not valid dates.')
    if s[value_col].isna().any(): raise ValueError(f'Some values in "{value_col}" are not numbers.')
    s = s.sort_values(date_col).drop_duplicates(date_col)
    idx = s[date_col].dt.to_period('M')
    if (idx.astype('int64').diff().dropna() != 1).any(): raise ValueError('The dates must be monthly with no gaps (one row per month).')
    if (s[value_col] <= 0).any(): raise ValueError('All values must be positive (the model works with logarithms).')
    return pd.Series(s[value_col].values.astype(float), index=idx.dt.to_timestamp(), name=value_col)


def seasonal_log_change(values):
    lg = np.log(np.asarray(values, dtype=float)); return lg[SEASON:] - lg[:-SEASON]        # length n - 12


def make_windows(arr, lo, hi):
    """Targets arr[lo:hi]; inputs = the K values before each target."""
    return np.array([arr[i - K:i] for i in range(lo, hi)]), arr[lo:hi]


# ----------------------------------------------------------------------------- model bundle
class Bundle:
    """Everything needed to forecast: scaling numbers of d, a linear model and (optionally) a trained GRU."""
    def __init__(self, dm, ds, lin, gru=None, info=''):
        self.dm, self.ds, self.lin, self.gru, self.info = dm, ds, lin, gru, info

    def _next_fns(self):
        fns = {'Linear regression (on d)': lambda w: float(self.lin.predict(np.asarray(w)[None, :])[0])}
        if self.gru is not None:
            def gru_next(w):
                with torch.no_grad(): return float(self.gru(torch.tensor(np.asarray(w), dtype=torch.float32)[None, :]).item())
            fns = {'GRU (on d)': gru_next, **fns}
        return fns

    def available(self): return list(self._next_fns()) + MODEL_NAMES[2:]

    def forecast(self, values, H):
        """Forecast the H months after `values` (H <= 12). Returns {model name: array of H forecasts}."""
        v = np.asarray(values, dtype=float); n = len(v)
        if not 1 <= H <= SEASON: raise ValueError('H must be between 1 and 12')
        if n < SEASON + K: raise ValueError(f'Need at least {SEASON + K} months of history.')
        dz = (seasonal_log_change(v) - self.dm) / self.ds; out = {}
        for name, nxt in self._next_fns().items():
            w, preds = list(dz[-K:]), []
            for h in range(H):
                z = nxt(w[-K:]); w.append(z); preds.append(v[n + h - SEASON] * np.exp(z * self.ds + self.dm))     # last year's value is always known for h < 12
            out[name] = np.array(preds)
        last_year = np.array([v[n + h - SEASON] for h in range(H)])
        out['Seasonal naive x avg growth'] = last_year * np.exp(self.dm); out['Seasonal naive'] = last_year; out['Naive (last month)'] = np.repeat(v[-1], H)
        return out

    def backtest_table(self, values, first_origin, H_max=6):
        """Average absolute error per horizon over all forecast origins from `first_origin` on (months before it are never evaluated)."""
        v = np.asarray(values, dtype=float); errs, counts = None, np.zeros(H_max, int)
        for o in range(first_origin, len(v)):
            H = min(H_max, len(v) - o); f = self.forecast(v[:o], H)
            if errs is None: errs = {k: [[] for _ in range(H_max)] for k in f}
            for k, p in f.items():
                for h in range(H): errs[k][h].append(abs(v[o + h] - p[h]))
            counts[:H] += 1
        tab = pd.DataFrame({k: [np.mean(e) if e else np.nan for e in lst] for k, lst in errs.items()}, index=[f'{h + 1}' for h in range(H_max)])
        tab.index.name = 'months ahead'; tab['forecasts averaged'] = counts; return tab[[c for c in MODEL_NAMES if c in tab.columns] + ['forecasts averaged']]


# ----------------------------------------------------------------------------- loading the saved airline model
def load_airline_bundle(values, model_path='forecaster.pt', meta_path='forecaster_meta.json', n_train=96):
    """The model trained in the Day 6 notebook (trained on 1949-1956 only). The linear model is refitted on the same training part."""
    meta = json.load(open(meta_path)); dm, ds = meta['dm'], meta['ds']
    gru = Net(meta['kind'], meta['hidden'], meta['layers'], meta['dropout']); gru.load_state_dict(torch.load(model_path, map_location='cpu')); gru.eval()
    d = np.full(len(values), np.nan); d[SEASON:] = seasonal_log_change(values); dz = (d - dm) / ds
    X, y = make_windows(dz, SEASON + K, n_train); lin = LinearRegression().fit(X, y)
    return Bundle(dm, ds, lin, gru, info=f"GRU ({meta['hidden']} units, {meta['layers']} layers) trained on the first {n_train} months (1949-1956); early stopping on 1957-1958.")


# ----------------------------------------------------------------------------- training on uploaded data
def fit_bundle_from_series(values, train_gru=True, seed=0, epochs=400, patience=40, holdout=12):
    """Fit the linear model (and a small GRU) on `values` WITHOUT the last `holdout` months, which stay untouched for a backtest."""
    v = np.asarray(values, dtype=float)
    if len(v) < 60: raise ValueError('Need at least 60 months of data to train a model.')
    tr = v[:-holdout]; d = seasonal_log_change(tr); dm, ds = d.mean(), d.std(); dz = (d - dm) / ds
    X, y = make_windows(dz, K, len(dz)); n_val = max(8, int(0.2 * len(X))); Xa, ya, Xb, yb = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]      # chronological split
    lin = LinearRegression().fit(Xa, ya); gru = None
    if train_gru:
        torch.manual_seed(seed); gru = Net(); opt = torch.optim.Adam(gru.parameters(), lr=0.01); lf = nn.MSELoss(); T = lambda a: torch.tensor(a, dtype=torch.float32)
        xa, ya_t, xb, yb_t = T(Xa), T(ya), T(Xb), T(yb); best, best_state, bad = float('inf'), None, 0
        for ep in range(epochs):
            gru.train(); perm = torch.randperm(len(xa))
            for i in range(0, len(perm), 32):
                idx = perm[i:i + 32]; opt.zero_grad(); lf(gru(xa[idx]), ya_t[idx]).backward(); nn.utils.clip_grad_norm_(gru.parameters(), 1.0); opt.step()
            gru.eval()
            with torch.no_grad(): vl = lf(gru(xb), yb_t).item()
            if vl < best - 1e-6: best, best_state, bad = vl, copy.deepcopy(gru.state_dict()), 0
            else:
                bad += 1
                if bad >= patience: break
        gru.load_state_dict(best_state); gru.eval()
    return Bundle(dm, ds, lin, gru, info=f'Trained on the first {len(tr)} months of your data (last {holdout} months held out). One seed only, so treat the GRU result as one sample.')
