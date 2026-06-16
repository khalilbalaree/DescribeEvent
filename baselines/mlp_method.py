# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""MLP sliding-window baseline for event prediction.

Pools sliding windows from all history sequences for training.
"""

import numpy as np
from collections import Counter
from baselines.statistical import BaselineMethod

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class MLPNet(nn.Module):
        """Simple MLP for event prediction from a sliding window."""

        def __init__(self, input_size, num_types, hidden_size=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.type_head = nn.Linear(hidden_size, num_types)
            self.time_head = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Softplus(),
            )

        def forward(self, x):
            h = self.net(x)
            type_logits = self.type_head(h)
            time_pred = self.time_head(h).squeeze(1)
            return type_logits, time_pred


class OnlineMLP(BaselineMethod):
    """MLP with sliding window, trained on pooled histories."""

    def __init__(self, window_size=3, hidden_size=64, epochs=50, lr=0.001, patience=10):
        self.W = window_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.type_list = []
        self.type_to_idx = {}
        self.patience = patience
        self.time_mean = 0.0
        self.time_std = 1.0
        self.device = torch.device("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")

    def fit(self, all_histories):
        if not HAS_TORCH:
            print("WARNING: torch not available, MLP will fallback to most_common")
            return

        # Collect types
        all_types = set()
        for hist in all_histories:
            for e in hist:
                all_types.add(e["type_event"])
        self.type_list = sorted(all_types)
        self.type_to_idx = {t: i for i, t in enumerate(self.type_list)}
        K = len(self.type_list)

        if K == 0:
            return

        # Build sliding window dataset from all histories
        X_list = []
        y_type_list = []
        y_time_list = []

        for hist in all_histories:
            if len(hist) < self.W + 1:
                continue
            for i in range(len(hist) - self.W):
                window = hist[i:i + self.W]
                target = hist[i + self.W]
                # Feature: one-hot types + time deltas for each event in window
                features = []
                for e in window:
                    one_hot = [0.0] * K
                    idx = self.type_to_idx.get(e["type_event"])
                    if idx is not None:
                        one_hot[idx] = 1.0
                    features.extend(one_hot)
                    features.append(e["time_since_last_event"])
                X_list.append(features)
                y_type_list.append(self.type_to_idx.get(target["type_event"], 0))
                y_time_list.append(target["time_since_last_event"])

        if not X_list:
            return

        X = np.array(X_list, dtype=np.float32)
        y_type = np.array(y_type_list, dtype=np.int64)
        y_time = np.array(y_time_list, dtype=np.float32)

        # Normalize time features
        time_cols = list(range(K, X.shape[1], K + 1))
        all_time_vals = X[:, time_cols].flatten()
        self.time_mean = float(np.mean(all_time_vals))
        self.time_std = float(np.std(all_time_vals))
        if self.time_std == 0:
            self.time_std = 1.0
        X[:, time_cols] = (X[:, time_cols] - self.time_mean) / self.time_std

        # Train/val split
        n = len(X)
        indices = np.random.RandomState(42).permutation(n)
        val_size = max(1, n // 10)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        device = self.device
        input_size = X.shape[1]
        self.model = MLPNet(input_size, K, self.hidden_size).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        ce_loss = nn.CrossEntropyLoss()

        X_train = torch.tensor(X[train_idx], device=device)
        y_type_train = torch.tensor(y_type[train_idx], device=device)
        y_time_train = torch.tensor(y_time[train_idx], device=device)
        X_val = torch.tensor(X[val_idx], device=device)
        y_type_val = torch.tensor(y_type[val_idx], device=device)
        y_time_val = torch.tensor(y_time[val_idx], device=device)

        batch_size = 256
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(X_train), device=device)
            for b in range(0, len(X_train), batch_size):
                idx = perm[b:b + batch_size]
                optimizer.zero_grad()
                type_logits, time_preds = self.model(X_train[idx])
                loss = ce_loss(type_logits, y_type_train[idx]) + \
                    torch.mean((time_preds - y_time_train[idx]) ** 2)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                type_logits, time_preds = self.model(X_val)
                val_loss = ce_loss(type_logits, y_type_val) + \
                    torch.mean((time_preds - y_time_val) ** 2)
                val_loss = val_loss.item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if 'best_state' in dir():
            self.model.load_state_dict(best_state)
        self.model.eval()
        print(f"MLP trained: {epoch + 1} epochs, {len(X_train)} samples, val_loss={best_val_loss:.4f}")

    def predict(self, history):
        if self.model is None or not HAS_TORCH:
            types = [e["type_event"] for e in history]
            pred_type = Counter(types).most_common(1)[0][0]
            pred_time = float(np.mean([e["time_since_last_event"] for e in history[-10:]]))
            return pred_type, pred_time, "online_mlp: fallback (no model)"

        K = len(self.type_list)
        W = self.W

        if len(history) < W:
            types = [e["type_event"] for e in history]
            pred_type = Counter(types).most_common(1)[0][0]
            pred_time = float(np.mean([e["time_since_last_event"] for e in history[-10:]]))
            return pred_type, pred_time, "online_mlp: fallback (insufficient history)"

        # Build feature from last W events
        window = history[-W:]
        features = []
        for e in window:
            one_hot = [0.0] * K
            idx = self.type_to_idx.get(e["type_event"])
            if idx is not None:
                one_hot[idx] = 1.0
            features.extend(one_hot)
            features.append((e["time_since_last_event"] - self.time_mean) / self.time_std)

        device = self.device
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32, device=device)
            type_logits, time_pred = self.model(x)
            pred_idx = int(torch.argmax(type_logits[0]))
            pred_type = self.type_list[pred_idx]
            pred_time = max(0.0, float(time_pred[0]))

        return pred_type, pred_time, f"online_mlp: W={W} prediction (type_idx={pred_idx})"
