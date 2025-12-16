import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from plot_flux_diagnostics import plot_results


#CONFIG


PKL_PATH = "merged_optical_tabular_0p1.pkl"

TAB_AE_FULL = PKL_PATH.replace(".pkl", "_tabAE_full.npy")
TAB_AE_ENR  = PKL_PATH.replace(".pkl", "_tabAE_enriched.npy")
IMG_EMB     = PKL_PATH.replace(".pkl", "_imgResNet.npy")

TARGET_COL = "Total_flux"
TEST_SPLIT = 0.20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Features for RAW and ENRICHED
KEY_FEATURES_ENRICHED = [
    "z_best","zphot","zphot_err","flag_qual",
    "mag_g","mag_r","mag_z","mag_w1","mag_w2",
    "magerr_g","magerr_r","magerr_z","magerr_w1","magerr_w2",
    "g_rest","r_rest","z_rest","U_rest","V_rest","J_rest","K_rest","w1_rest","w2_rest",
    "Mass_median","Mass_l68","Mass_u68",
    "r_50","r_50_err",
    "pstar",
    "optRA","optDec"
]

# Features for FULL dataset
KEY_FEATURES_FULL = [
    "mag_g","mag_r","mag_z","mag_w1","mag_w2",
    "magerr_g","magerr_r","magerr_z","magerr_w1","magerr_w2",
    "optRA","optDec"
]



# MLP MODEL

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

# TRAIN NN

def train_nn(X_train, y_train, X_test, y_test, epochs=25):
    model = MLP(X_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE).view(-1, 1)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE).view(-1, 1)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for ep in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  Epoch {ep+1}/{epochs}  Loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred_train = model(X_train_t).cpu().numpy().flatten()
        pred_test  = model(X_test_t).cpu().numpy().flatten()

    return pred_train, pred_test

# EVAL FUNCTION (correlation removed)

def evaluate_model(name, y_train, pred_train, y_test, pred_test):
    print(f"\n===========================")
    print(f"   RESULTS: {name}")
    print("===========================")

    # Train
    train_err = pred_train - y_train
    print("Train MAE:", mean_absolute_error(y_train, pred_train))
    print("Train Median AE:", np.median(np.abs(train_err)))
    print("Train Std:", np.std(train_err))

    # Test
    test_err = pred_test - y_test
    print("\nTest MAE:", mean_absolute_error(y_test, pred_test))
    print("Test Median AE:", np.median(np.abs(test_err)))
    print("Test Std:", np.std(test_err))


#LOAD DATA

df = pd.read_pickle(PKL_PATH)
y_raw = df[TARGET_COL].values.astype(np.float32)
y_log = np.log1p(y_raw).astype(np.float32)
img_emb = np.load(IMG_EMB)

#FULL DATASET

df_full = df[KEY_FEATURES_FULL].dropna()
tab_full = np.load(TAB_AE_FULL)
img_full = img_emb[df_full.index]

X_full = np.concatenate([img_full, tab_full], axis=1)
y_full_raw = y_raw[df_full.index]
y_full_log = y_log[df_full.index]



#ENRICHED DATASET

df_enr = df[KEY_FEATURES_ENRICHED].dropna()
tab_enr = np.load(TAB_AE_ENR)
img_enr = img_emb[df_enr.index]

X_enr = np.concatenate([img_enr, tab_enr], axis=1)
y_enr_raw = y_raw[df_enr.index]
y_enr_log = y_log[df_enr.index]

#RAW DATASET

df_raw = df[KEY_FEATURES_ENRICHED].dropna()

tab_raw = df_raw.values.astype(np.float32)       
img_raw = img_emb[df_raw.index].astype(np.float32)

X_raw = np.concatenate([img_raw, tab_raw], axis=1)
y_raw2 = y_raw[df_raw.index]
y_log2 = y_log[df_raw.index]

IMG_DIM_RAW = img_raw.shape[1]  

#PIPELINE

def pipeline(name, X, y, make_plots=False, scale_tab=False, img_dim=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42
    )

    if scale_tab:
        assert img_dim is not None, "Pass img_dim so we can split image vs tabular columns."

        scaler = StandardScaler()

        Xtr_img, Xtr_tab = X_train[:, :img_dim], X_train[:, img_dim:]
        Xte_img, Xte_tab = X_test[:, :img_dim],  X_test[:, img_dim:]

        Xtr_tab = scaler.fit_transform(Xtr_tab)   # <-- fit ONLY on train
        Xte_tab = scaler.transform(Xte_tab)

        X_train = np.concatenate([Xtr_img, Xtr_tab], axis=1)
        X_test  = np.concatenate([Xte_img, Xte_tab], axis=1)

    #Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    pred_train = ridge.predict(X_train)
    pred_test  = ridge.predict(X_test)

    evaluate_model(f"Ridge ({name})", y_train, pred_train, y_test, pred_test)

    if make_plots:
        plot_results(y_test, pred_test, title=f"{name} – Ridge")

    #Neural Net
    pred_train_nn, pred_test_nn = train_nn(X_train, y_train, X_test, y_test)
    evaluate_model(f"NN ({name})", y_train, pred_train_nn, y_test, pred_test_nn)

    if make_plots:
        plot_results(y_test, pred_test_nn, title=f"{name} – NN")

#RUN ALL EXPERIMENTS WITH PLOTS

pipeline("FULL (flux)",     X_full, y_full_raw, make_plots=True)
pipeline("FULL (log flux)", X_full, y_full_log, make_plots=True)

pipeline("ENRICHED (flux)",     X_enr, y_enr_raw, make_plots=True)
pipeline("ENRICHED (log flux)", X_enr, y_enr_log, make_plots=True)

pipeline("RAW (flux)",     X_raw, y_raw2, make_plots=True, scale_tab=True, img_dim=IMG_DIM_RAW)
pipeline("RAW (log flux)", X_raw, y_log2, make_plots=True, scale_tab=True, img_dim=IMG_DIM_RAW)

