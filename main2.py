import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

def _resolve_sizes(n, train_size, val_size, test_size):
    """Allow float ratios (sum=1.0) OR integer counts."""
    sizes = [train_size, val_size, test_size]
    if all(isinstance(s, (float, np.floating)) for s in sizes):
        total = float(train_size + val_size + test_size)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("If using ratios, train_size+val_size+test_size must equal 1.0")
        n_train = int(train_size * n)
        n_val = int(val_size * n)
        n_test = n - n_train - n_val
        return n_train, n_val, n_test

    n_train, n_val, n_test = int(train_size), int(val_size), int(test_size)
    if n_train + n_val + n_test > n:
        raise ValueError("train_size+val_size+test_size cannot exceed dataset size")
    return n_train, n_val, n_test

def data_split(X, Y, train_size, val_size, test_size, seed=42, shuffle=True):
    """Return: train_X, train_Y, val_X, val_Y, test_X, test_Y"""
    n = len(X)
    n_train, n_val, n_test = _resolve_sizes(n, train_size, val_size, test_size)

    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    X = X[idx]
    Y = Y[idx]

    train_X = X[:n_train]
    train_Y = Y[:n_train]

    val_X = X[n_train:n_train + n_val]
    val_Y = Y[n_train:n_train + n_val]

    test_X = X[n_train + n_val:n_train + n_val + n_test]
    test_Y = Y[n_train + n_val:n_train + n_val + n_test]

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def normalize_batch(X_b, eps=1e-8):
    """
    Normalize batch:
      Xn = (X - mean_batch) / std_batch
      then scale so variance becomes 0.5: Xn *= sqrt(0.5)
    """
    mean_b = np.mean(X_b, axis=0, keepdims=True)
    std_b = np.std(X_b, axis=0, keepdims=True)
    std_b = np.where(std_b < eps, 1.0, std_b)
    Xn = (X_b - mean_b) / std_b
    Xn = Xn * np.sqrt(0.5)
    return Xn

def loadDataset(dataset_path, train_size, val_size, test_size, batch_size):
    """
    REQUIRED SIGNATURE:
      loadDataset(dataset_path, train_size, val_size, test_size, batch_size)
    RETURN:
      X_train, y_train, X_val, y_val, X_test, y_test
    """
    train_csv = os.path.join(dataset_path, "train.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv not found at: {train_csv}")

    df = pd.read_csv(train_csv)

    if "Survived" not in df.columns:
        raise ValueError("Survived column not found in Titanic train.csv")

    y = df["Survived"].values.reshape(-1, 1).astype(np.float32)

    use_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    for c in use_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    Xdf = df[use_cols].copy()

    Xdf["Sex"] = Xdf["Sex"].map({"male": 1, "female": 0}).astype(np.float32)

    Xdf["Age"] = Xdf["Age"].fillna(Xdf["Age"].median())
    Xdf["Fare"] = Xdf["Fare"].fillna(Xdf["Fare"].median())
    Xdf["Embarked"] = Xdf["Embarked"].fillna(Xdf["Embarked"].mode()[0])

    emb = pd.get_dummies(Xdf["Embarked"], prefix="Emb", dtype=np.float32)
    Xdf = Xdf.drop("Embarked", axis=1)
    Xdf = pd.concat([Xdf, emb], axis=1)

    for col in ["Emb_C", "Emb_Q", "Emb_S"]:
        if col not in Xdf.columns:
            Xdf[col] = 0.0

    X = Xdf.values.astype(np.float32)

    X_train, y_train, X_val, y_val, X_test, y_test = data_split(
        X, y,
        train_size=train_size, val_size=val_size, test_size=test_size,
        seed=42, shuffle=True
    )

    if len(X_train) >= batch_size:
        Xb = X_train[:batch_size]
        Xb_n = normalize_batch(Xb)
        print("Train mean (approx):", np.mean(Xb_n, axis=0)[:5])
        print("Train var  (approx):", np.var(Xb_n, axis=0)[:5])

    return X_train, y_train, X_val, y_val, X_test, y_test

class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LogisticRegressionNet:
    def __init__(self, input_size):
        self.w = np.zeros((input_size, 1), dtype=np.float32)
        self.b = np.zeros((1, 1), dtype=np.float32)

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def feed_forward(self, X_b):
        z = X_b @ self.w + self.b
        return self.sigmoid(z)

def logistic_loss(y_true, y_hat, eps=1e-8):
    y_hat = np.clip(y_hat, eps, 1 - eps)
    loss = -(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))
    return float(np.mean(loss))

def compute_gradient(X_b, y_b, y_hat):
    dz = (y_hat - y_b)   
    B = X_b.shape[0]
    dw = (X_b.T @ dz) / B
    db = np.sum(dz, axis=0, keepdims=True) / B
    return dw, db

def optimization(lr, grad, net):
    dw, db = grad
    net.w = net.w - lr * dw
    net.b = net.b - lr * db
    return net

def evaluate(net, X, y, batch_size=64, use_norm=True):
    losses = []
    preds = []
    n = len(X)

    for i in range(0, n, batch_size):
        X_b = X[i:i + batch_size]
        y_b = y[i:i + batch_size]

        if use_norm:
            X_b = normalize_batch(X_b)

        y_hat = net.feed_forward(X_b)

        losses.append(logistic_loss(y_b, y_hat))
        preds.append((y_hat >= 0.5).astype(np.int32))

    y_pred = np.vstack(preds)
    acc = float(np.mean(y_pred == y))
    return float(np.mean(losses)), acc

def predict_labels(net, X, batch_size=64, use_norm=True):
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        X_b = X[i:i + batch_size]
        if use_norm:
            X_b = normalize_batch(X_b)
        y_hat = net.feed_forward(X_b)
        preds.append((y_hat >= 0.5).astype(np.int32))
    return np.vstack(preds)

def train(net, train_X, train_Y, val_X, val_Y, test_X, test_Y,
          batch_size, n_epochs, lr, use_norm=True):

    train_loader = DataLoader(
        TitanicDataset(train_X, train_Y),
        batch_size=batch_size,
        shuffle=True
    )

    loss_epoch_tr, loss_epoch_val, loss_epoch_test = [], [], []
    acc_epoch_tr, acc_epoch_val, acc_epoch_test = [], [], []

    for ep in range(n_epochs):
        # SGD batches
        for X_b_t, y_b_t in train_loader:
            X_b = X_b_t.numpy()
            y_b = y_b_t.numpy()

            if use_norm:
                X_b = normalize_batch(X_b)

            y_hat = net.feed_forward(X_b)
            grad = compute_gradient(X_b, y_b, y_hat)
            net = optimization(lr, grad, net)

        tr_loss, tr_acc = evaluate(net, train_X, train_Y, batch_size, use_norm)
        va_loss, va_acc = evaluate(net, val_X, val_Y, batch_size, use_norm)
        te_loss, te_acc = evaluate(net, test_X, test_Y, batch_size, use_norm)

        loss_epoch_tr.append(tr_loss)
        loss_epoch_val.append(va_loss)
        loss_epoch_test.append(te_loss)

        acc_epoch_tr.append(tr_acc)
        acc_epoch_val.append(va_acc)
        acc_epoch_test.append(te_acc)

        print(f"[{'NORM' if use_norm else 'NO-NORM'}] "
              f"Epoch {ep+1:03d} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"Val loss {va_loss:.4f} acc {va_acc:.3f} | "
              f"Test loss {te_loss:.4f} acc {te_acc:.3f}")

    return net, loss_epoch_tr, loss_epoch_val, loss_epoch_test, acc_epoch_tr, acc_epoch_val, acc_epoch_test

def test_function(model, test_X, test_Y, batch_size=64, use_norm=True):
    test_loss, test_acc = evaluate(model, test_X, test_Y, batch_size, use_norm)
    y_pred = predict_labels(model, test_X, batch_size, use_norm)
    return test_loss, test_acc, y_pred

def visualize(loss_tr, loss_val, loss_test,
              acc_tr, acc_val, acc_test,
              y_true_test, y_pred_test,
              save_dir="task2_plots"):

    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(loss_tr, label="Train Loss")
    plt.plot(loss_val, label="Val Loss")
    plt.plot(loss_test, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves (Train/Val/Test)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curves.png"))
    plt.show()

    plt.figure()
    plt.plot(acc_tr, label="Train Acc")
    plt.plot(acc_val, label="Val Acc")
    plt.plot(acc_test, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves (Train/Val/Test)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curves.png"))
    plt.show()

    cm = confusion_matrix(y_true_test.reshape(-1), y_pred_test.reshape(-1))
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()

    f1 = f1_score(y_true_test.reshape(-1), y_pred_test.reshape(-1))
    acc = accuracy_score(y_true_test.reshape(-1), y_pred_test.reshape(-1))
    print("\nTEST Accuracy:", acc)
    print("TEST F1-score:", f1)
    print("\nClassification report:\n", classification_report(y_true_test.reshape(-1), y_pred_test.reshape(-1)))

def main():
    dataset_path = "Task_2"
    train_size = 0.70
    val_size = 0.15
    test_size = 0.15

    batch_size = 64
    n_epochs = 20
    lr = 0.1

    X_train, y_train, X_val, y_val, X_test, y_test = loadDataset(
        dataset_path, train_size, val_size, test_size, batch_size
    )

    print("\nShapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    input_size = X_train.shape[1]
    net_norm = LogisticRegressionNet(input_size)
    net_norm, loss_tr_n, loss_val_n, loss_test_n, acc_tr_n, acc_val_n, acc_test_n = train(
        net_norm,
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, n_epochs=n_epochs, lr=lr,
        use_norm=True
    )
    test_loss_n, test_acc_n, y_pred_test_n = test_function(net_norm, X_test, y_test, batch_size, use_norm=True)
    print(f"\n[WITH NORM] FINAL Test Acc: {test_acc_n:.3f}  (must be >= 0.60)")

    with open("model_task2.pkl", "wb") as f:
        pickle.dump(net_norm, f)
    print("\n Saved model as model_task2.pkl")

    with open("model_task2.pkl", "rb") as f:
        loaded = pickle.load(f)
    print("Model reloaded from model_task2.pkl")

    test_loss2, test_acc2, y_pred_test2 = test_function(loaded, X_test, y_test, batch_size, use_norm=True)
    print(f"RETEST Test Acc (with norm): {test_acc2:.3f}")

    visualize(
        loss_tr_n, loss_val_n, loss_test_n,
        acc_tr_n, acc_val_n, acc_test_n,
        y_test, y_pred_test2,
        save_dir=os.path.join("task2_plots", "with_norm")
    )

    net_non = LogisticRegressionNet(input_size)
    net_non, loss_tr_0, loss_val_0, loss_test_0, acc_tr_0, acc_val_0, acc_test_0 = train(
        net_non,
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, n_epochs=n_epochs, lr=lr,
        use_norm=False
    )
    test_loss_0, test_acc_0, y_pred_test_0 = test_function(net_non, X_test, y_test, batch_size, use_norm=False)
    print(f"\n[NO NORM] FINAL Test Acc: {test_acc_0:.3f}  (must be >= 0.60)")

    visualize(
        loss_tr_0, loss_val_0, loss_test_0,
        acc_tr_0, acc_val_0, acc_test_0,
        y_test, y_pred_test_0,
        save_dir=os.path.join("task2_plots", "without_norm")
    )

if __name__ == "__main__":
    main()
