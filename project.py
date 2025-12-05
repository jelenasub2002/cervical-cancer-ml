# CervicalCancer_sign_transfer_experiments.ipynb
# Potrebne biblioteke: pandas, numpy, scikit-learn, matplotlib, torch
# Instaliraj ako treba: pip install pandas numpy scikit-learn matplotlib torch

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ---------- CONFIG ----------
DATA_PATH = r"risk_factors_cervical_cancer.csv"
TARGET_COLS = {
    'H': 'Hinselmann',   
    'S': 'Schiller',
    'C': 'Citology',
    'B': 'Biopsy'
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------

# ---------- UTIL: load & preprocess ----------
def load_and_preprocess(path):
    df = pd.read_csv(path)
    # drop obviously irrelevant columns if exists (ID itd.)
    # identify feature columns (exclude our target label columns)
    target_names = list(TARGET_COLS.values())

    present_targets = [c for c in target_names if c in df.columns]
    if len(present_targets) != len(target_names):
        print(f"Upozorenje: Neke ciljne kolone su izostavljene iz CSV-a: {list(set(target_names) - set(present_targets))}")
        target_names = present_targets
    # Some datasets encode bool×value as two columns; here we assume CSV already has numeric columns.
    feature_cols = [c for c in df.columns if c not in target_names]
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    y = df[target_names].copy()
    # Impute missing with mean (kao u radu)
    imputer = SimpleImputer(strategy='mean')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    # One-hot for categorical (if any object columns)
    obj_cols = X_imp.select_dtypes(include=['object', 'category']).columns.tolist()
    if obj_cols:
        enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        oh = enc.fit_transform(X_imp[obj_cols])
        oh_df = pd.DataFrame(oh, columns=enc.get_feature_names_out(obj_cols))
        X_imp = X_imp.drop(columns=obj_cols).reset_index(drop=True)
        X_imp = pd.concat([X_imp, oh_df], axis=1)
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)
    # Ensure binary target columns are numeric 0/1 or continuous risk values depending dataset
    for col in target_names:
        if col in y.columns:
            y[col] = pd.to_numeric(y[col], errors='coerce').fillna(0)
    return X_scaled, y, feature_cols

# ---------- UTIL: train source linear model (Ridge) ----------
def train_source_ridge(X_train, y_train, alpha=1.0):
    # fits linear ridge and returns coefficients (including intercept)
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    coef = np.concatenate(([model.intercept_], model.coef_))
    return model, coef

# ---------- IMPLEMENT: sign-transfer regularizer and target linear model using PyTorch ----------
class LinearTorchModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        # weights vector (no bias separate) - we'll include bias as first param
        self.bias = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(n_features))
    def forward(self, x):
        return x.matmul(self.w) + self.bias

def sign_transfer_penalty(weights, src_coef, alpha=1.0):
    # weights: torch tensor shape (n_features,)
    # src_coef: numpy array shape (n_features,) (excluding bias) -> we use sign(src_coef)
    # Implements Δ2,α penalty per Eq (3) and (4) smooth p=2 version:
    # Δ2,α = α * sum(max(0, -w_i * sign(src_i))^2) + (1-α) * sum(w_i^2)
    src_sign = np.sign(src_coef)
    src_sign_t = torch.tensor(src_sign, dtype=weights.dtype, device=weights.device)
    # hinge-like term: max(0, -w_i * sign(src_i))
    hinge = torch.clamp(-weights * src_sign_t, min=0.0)
    delta = alpha * torch.sum(hinge ** 2) + (1.0 - alpha) * torch.sum(weights ** 2)
    return delta

def train_target_with_sign_transfer(X_train, y_train, src_coef, lambda_reg=1.0, alpha=1.0,
                                   n_epochs=200, lr=1e-2, batch_size=64, verbose=False):
    # X_train: np.array (n, d), y_train: np.array (n,)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LinearTorchModel(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            pred = model(xb).squeeze()
            mse = nn.functional.mse_loss(pred, yb)
            # sign-transfer penalty applies only on weights (exclude bias)
            penalty = sign_transfer_penalty(model.w, src_coef, alpha=alpha)
            loss = mse + lambda_reg * penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch}/{n_epochs-1}, loss={epoch_loss/len(dataset):.6f}")
    # extract coefficients
    w_np = model.w.detach().cpu().numpy()
    b_np = model.bias.detach().cpu().numpy()[0]
    coef = np.concatenate(([b_np], w_np))
    return model, coef

# ---------- SIMPLE MLP (PyTorch) for comparison ----------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64,32]):
        super().__init__()
        layers = []
        d_in = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d_in, h))
            layers.append(nn.ReLU())
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze()

def train_mlp(X_train, y_train, X_val=None, y_val=None, epochs=200, lr=1e-3, batch_size=64, verbose=False):
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleMLP(X_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (ep % 50 == 0):
            print(f"MLP epoch {ep}, loss {loss.item():.6f}")
    return model

# ---------- helper: evaluate linear model given coef (intercept + weights) ----------
def predict_from_coef(coef, X):
    # coef: [intercept, w1, w2, ...]
    X_np = np.asarray(X)
    intercept = coef[0]
    w = coef[1:]
    # Osiguravamo da su dimenzije usklađene, posebno ako X ima samo jedan uzorak
    if X_np.ndim == 1:
        X_np = X_np.reshape(1, -1)
    return X_np.dot(w) + intercept

# ---------- MAIN EXPERIMENT LOOP ----------
def run_all_pairs(X, y_df, pairs=None, test_size=0.2, seed=42):
    # pairs: list of tuples [('H','S'), ...] where keys correspond to TARGET_COLS keys (H,S,C,B)
    if pairs is None:
        keys = list(TARGET_COLS.keys())
        pairs = [(s,t) for s in keys for t in keys if s != t]
    results = []

    X_df = X.copy()
    
    for src_key, tgt_key in pairs:
        src_col = TARGET_COLS[src_key]
        tgt_col = TARGET_COLS[tgt_key]
        print(f"\n--- Experiment: Source={src_col}  ->  Target={tgt_col} ---")
        # Prepare (drop rows where target is missing? here we use all rows and treat missing as 0)
        y_src = y_df[src_col].values
        y_tgt = y_df[tgt_col].values
        # Stratified split based on target (if classification-like) — here we just use continuous -> stratify by bin
        # Create a simple binning for stratify
        bins = np.digitize(y_tgt, np.quantile(y_tgt, [0.25,0.5,0.75]))
        X_train, X_test, y_src_train, y_src_test, y_tgt_train, y_tgt_test = train_test_split(
            X, y_src, y_tgt, test_size=test_size, random_state=seed, stratify=bins)
        # --- train source model on source labels (Ridge) ---
        ridge_model, src_coef = train_source_ridge(X_train, y_src_train, alpha=1.0)
        # src_coef includes intercept + coef vector
        # For sign-transfer penalty we use coefficients (excluding intercept)
        src_weights = src_coef[1:]
        # --- Baseline: target trained without transfer (Ridge on target) ---
        baseline_model, baseline_coef = train_source_ridge(X_train, y_tgt_train, alpha=1.0)
        y_pred_base = predict_from_coef(baseline_coef, X_test)
        rmse_base = np.sqrt(mean_squared_error(y_tgt_test, y_pred_base))
        # --- Target with sign-transfer regularizer ---
        # grid-search for lambda and alpha (small grid to keep compute feasible)
        best = {'rmse': np.inf}
        lambdas = [0.001, 0.01, 0.1, 1.0]
        alphas = [1.0, 0.9, 0.5]
        for lam, alpha in itertools.product(lambdas, alphas):
            model_tgt, tgt_coef = train_target_with_sign_transfer(X_train.values, y_tgt_train,
                                                                 src_weights, lambda_reg=lam, alpha=alpha,
                                                                 n_epochs=300, lr=1e-2, batch_size=64, verbose=False)
            y_pred = predict_from_coef(tgt_coef, X_test)
            rmse = np.sqrt(mean_squared_error(y_tgt_test, y_pred))
            if rmse < best['rmse']:
                best = {'rmse': rmse, 'lambda': lam, 'alpha': alpha, 'coef': tgt_coef}
        # --- Train MLP on target (as additional algorithm) ---
        mlp = train_mlp(X_train.values, y_tgt_train, epochs=400, lr=1e-3, batch_size=64, verbose=False)
        # evaluate mlp
        mlp.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test.values, dtype=torch.float32, device=DEVICE)
            mlp_preds = mlp(X_test_t).cpu().numpy()
        rmse_mlp = np.sqrt(mean_squared_error(y_tgt_test, mlp_preds))
        # Collect results
        rel_gain = (baseline_rmse_to_positive_gain(rmse_base, best['rmse']))
        print(f"Baseline RMSE: {rmse_base:.4f} | Sign-Transfer best RMSE: {best['rmse']:.4f} (lambda={best['lambda']}, alpha={best['alpha']}) | MLP RMSE: {rmse_mlp:.4f}")
        results.append({
            'source': src_col,
            'target': tgt_col,
            'rmse_baseline': rmse_base,
            'rmse_sign_transfer': best['rmse'],
            'sign_lambda': best['lambda'],
            'sign_alpha': best['alpha'],
            'rmse_mlp': rmse_mlp,
            'relative_gain_percent': rel_gain
        })
    return pd.DataFrame(results)

def baseline_rmse_to_positive_gain(baseline_rmse, transfer_rmse):
    # Paper measures normalized signed AUC / percentage relative gain; here we compute simple % improvement:
    # positive gain if transfer_rmse < baseline_rmse
    if baseline_rmse == 0:
        return 0.0
    gain = (baseline_rmse - transfer_rmse) / baseline_rmse * 100.0
    return gain

# ---------- RUN ----------
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please provide CSV file.")
    X, y_df, feat_cols = load_and_preprocess(DATA_PATH)
    results_df = run_all_pairs(X, y_df)
    print("\nALL RESULTS:")
    print(results_df.head(50))
    results_df.to_csv("sign_transfer_experiment_results.csv", index=False)
    # quick plot of baseline vs sign-transfer RMSE
    ind = np.arange(len(results_df))
    width = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(ind - width/2, results_df['rmse_baseline'], width, label='Baseline (Ridge)')
    plt.bar(ind + width/2, results_df['rmse_sign_transfer'], width, label='Sign-Transfer')
    plt.xticks(ind, results_df.apply(lambda r: f"{r.source}->{r.target}", axis=1), rotation=45, ha='right')
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rmse_comparison.png")
    plt.show()
