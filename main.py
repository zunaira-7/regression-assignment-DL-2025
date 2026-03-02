import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from linear_regression import LinearRegressionNetwork
from training_utils import feed_forward, l2_loss, compute_gradient, update_parameters

df_train = pd.read_csv("Task_1/california_housing_train.csv")
df_test  = pd.read_csv("Task_1/california_housing_test.csv")

X_all = df_train.drop("target", axis=1).values
Y_all = df_train["target"].values.reshape(-1, 1)

if "target" in df_test.columns:
    df_test["target"] = pd.to_numeric(df_test["target"], errors="coerce")
    if df_test["target"].isna().any():
        print("Test target has NaNs → Test Loss/R² cannot be computed.")
        X_test_raw = df_test.drop("target", axis=1).values
        Y_test = None
    else:
        X_test_raw = df_test.drop("target", axis=1).values
        Y_test = df_test["target"].values.reshape(-1, 1)
else:
    X_test_raw = df_test.values
    Y_test = None

print("X_all shape:", X_all.shape)
print("Y_all shape:", Y_all.shape)
print("X_test shape:", X_test_raw.shape)
print("Y_test available:", Y_test is not None)

N = X_all.shape[0]
split_index = int(0.85 * N)

X_train_raw = X_all[:split_index]
Y_train = Y_all[:split_index]

X_val_raw = X_all[split_index:]
Y_val = Y_all[split_index:]

print("Train X shape:", X_train_raw.shape, "Train Y shape:", Y_train.shape)
print("Val X shape:", X_val_raw.shape, "Val Y shape:", Y_val.shape)

mean = np.mean(X_train_raw, axis=0)
std = np.std(X_train_raw, axis=0)
std[std == 0] = 1e-8

X_train_norm = (X_train_raw - mean) / std
X_val_norm   = (X_val_raw - mean) / std
X_test_norm  = (X_test_raw - mean) / std

print("Normalization completed.")

class HousingDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

tmp_loader = DataLoader(HousingDataset(X_train_norm, Y_train), batch_size=64, shuffle=True)
for xb, yb in tmp_loader:
    print("Batch X shape:", xb.shape)
    print("Batch Y shape:", yb.shape)
    break

def r2_score(y_true, y_pred):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def run_experiment(Xtr, Ytr, Xva, Yva, hidden_dim, lr, epochs, batch_size):
    train_ds = HousingDataset(Xtr, Ytr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = LinearRegressionNetwork(input_dim=8, hidden_dim=hidden_dim)
    model.mean, model.std = mean, std

    train_losses, val_losses, val_r2s = [], [], []

    for ep in range(epochs):
        for X_b, Y_b in train_loader:
            X_b = X_b.numpy()
            Y_b = Y_b.numpy()

            y_hat = feed_forward(model, X_b)
            grads = compute_gradient(model, X_b, Y_b, y_hat)
            update_parameters(model, grads, lr)

        y_hat_tr = feed_forward(model, Xtr)
        y_hat_va = feed_forward(model, Xva)

        tr_loss = l2_loss(Ytr, y_hat_tr)
        va_loss = l2_loss(Yva, y_hat_va)
        va_r2   = r2_score(Yva, y_hat_va)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        val_r2s.append(va_r2)

    return model, train_losses, val_losses, val_r2s

os.makedirs("plots", exist_ok=True)
os.makedirs("report_plots", exist_ok=True)

hyperparam_list = [
    {"hidden": 8, "lr": 0.01,  "epochs": 100, "batch": 64},
    {"hidden": 2, "lr": 0.01, "epochs": 100, "batch": 64},
    {"hidden": 8, "lr": 0.01, "epochs": 150, "batch": 64},
    {"hidden": 2, "lr": 0.01, "epochs": 150, "batch": 64},
    {"hidden": 8, "lr": 0.00001, "epochs": 100, "batch": 64},
    {"hidden": 2, "lr": 0.00001, "epochs": 100, "batch": 64},
    {"hidden": 8, "lr": 0.00001, "epochs": 150, "batch": 64},
    {"hidden": 2, "lr": 0.00001, "epochs": 150, "batch": 64},
    {"hidden": 8, "lr": 0.9,  "epochs": 100, "batch": 64},
    {"hidden": 2, "lr": 0.9, "epochs": 100, "batch": 64},
    {"hidden": 8, "lr": 0.9, "epochs": 150, "batch": 64},
    {"hidden": 2, "lr": 0.9, "epochs": 150, "batch": 64},
]

best_model = None
best_name = ""
best_val_loss = float("inf")

best_train_losses = None
best_val_losses   = None
best_val_r2s      = None
best_used_norm    = True  

for hp in hyperparam_list:
    hidden_dim = hp["hidden"]
    lr = hp["lr"]
    epochs = hp["epochs"]
    batch_size = hp["batch"]

    for use_norm in [True, False]:

        if use_norm:
            Xtr, Xva = X_train_norm, X_val_norm
            tag = "norm"
        else:
            Xtr, Xva = X_train_raw, X_val_raw
            tag = "nonorm"

        model, tr_losses, va_losses, va_r2s = run_experiment(
            Xtr, Y_train, Xva, Y_val,
            hidden_dim=hidden_dim,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size
        )

        if not use_norm:
            continue

        run_name = f"{tag}_hidden{hidden_dim}_lr{lr}_bs{batch_size}"

        plt.figure()
        plt.plot(tr_losses, label="Train Loss")
        plt.plot(va_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(run_name)
        plt.legend()
        plt.grid(True)
        loss_path = f"plots/{run_name}_loss.png"
        plt.savefig(loss_path)
        plt.close()

        plt.figure()
        plt.plot(va_r2s)
        plt.xlabel("Epoch")
        plt.ylabel("R² (Validation)")
        plt.title(f"{run_name}_r2")
        plt.grid(True)
        r2_path = f"plots/{run_name}_r2.png"
        plt.savefig(r2_path)
        plt.close()

        if va_losses[-1] < best_val_loss:
            best_val_loss = va_losses[-1]
            best_model = model
            best_name = run_name
            best_train_losses = tr_losses.copy()
            best_val_losses = va_losses.copy()
            best_val_r2s = va_r2s.copy()
            best_used_norm = use_norm

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n Best model saved as model.pkl")
print("Best run:", best_name, "| best final val loss =", best_val_loss)

with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

print("Model loaded from model.pkl")

X_val_for_best = X_val_norm if best_used_norm else X_val_raw
X_test_for_best = X_test_norm if best_used_norm else X_test_raw

y_hat_val = feed_forward(loaded_model, X_val_for_best)
val_loss = l2_loss(Y_val, y_hat_val)
val_r2 = r2_score(Y_val, y_hat_val)

print("FINAL Validation Loss:", val_loss)
print("FINAL Validation R²:", val_r2)

if Y_test is not None and not np.isnan(Y_test).any():
    y_hat_test = feed_forward(loaded_model, X_test_for_best)
    test_loss = l2_loss(Y_test, y_hat_test)
    test_r2 = r2_score(Y_test, y_hat_test)
    print("FINAL Test Loss:", test_loss)
    print("FINAL Test R²:", test_r2)
else:
    print("⚠️ Test Y missing/NaN → Test Loss and Test R² cannot be computed (dataset issue).")

plt.figure()
plt.plot(best_train_losses, label="Train Loss")
plt.plot(best_val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"BEST Loss Curve: {best_name}")
plt.legend()
plt.grid(True)
plt.savefig("report_plots/best_loss_curve.png")
plt.show()

plt.figure()
plt.plot(best_val_r2s, label="Val R²")
plt.xlabel("Epoch")
plt.ylabel("R²")
plt.title(f"BEST R² Curve: {best_name}")
plt.legend()
plt.grid(True)
plt.savefig("report_plots/best_r2_curve.png")
plt.show()

k = min(200, len(Y_val))  
plt.figure()
plt.plot(Y_val[:k], label="True")
plt.plot(y_hat_val[:k], label="Predicted")
plt.xlabel("Samples (first 200)")
plt.ylabel("Target")
plt.title(f"BEST Prediction Curve: {best_name}")
plt.legend()
plt.grid(True)
plt.savefig("report_plots/best_prediction_curve.png")
plt.show()
