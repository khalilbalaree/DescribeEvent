# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""RMTPP (Recurrent Marked Temporal Point Process) baseline.

GRU-based neural model trained on pooled history sequences.
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
    class RMTPPNet(nn.Module):
        """GRU-based RMTPP network."""

        def __init__(self, num_types, hidden_size=32):
            super().__init__()
            self.num_types = num_types
            self.hidden_size = hidden_size
            # Input: one-hot type + raw time_delta
            input_size = num_types + 1
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
            self.type_head = nn.Linear(hidden_size, num_types)
            self.time_head = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Softplus(),
            )

        def forward(self, type_indices, time_deltas):
            """
            type_indices: (batch, seq_len) LongTensor
            time_deltas: (batch, seq_len) FloatTensor
            Returns: type_logits (batch, seq_len, num_types), time_preds (batch, seq_len)
            """
            batch_size, seq_len = type_indices.shape
            # One-hot encode types
            one_hot = torch.zeros(batch_size, seq_len, self.num_types,
                                  device=type_indices.device)
            one_hot.scatter_(2, type_indices.unsqueeze(2), 1.0)
            # Raw time delta
            x = torch.cat([one_hot, time_deltas.unsqueeze(2)], dim=2)
            h, _ = self.gru(x)
            type_logits = self.type_head(h)
            time_preds = self.time_head(h).squeeze(2)
            return type_logits, time_preds


class RMTPP(BaselineMethod):
    """RMTPP: GRU-based recurrent marked temporal point process."""

    def __init__(self, hidden_size=32, epochs=100, lr=0.001, patience=10):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.model = None
        self.type_list = []
        self.type_to_idx = {}
        self.device = torch.device("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")

    def fit(self, all_histories):
        if not HAS_TORCH:
            print("WARNING: torch not available, RMTPP will fallback to most_common")
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

        # Prepare sequences: each history becomes a training sequence
        # Input: events[:-1], Target: events[1:]
        sequences = []
        for hist in all_histories:
            if len(hist) < 3:
                continue
            types = [self.type_to_idx[e["type_event"]] for e in hist]
            times = [e["time_since_last_event"] for e in hist]
            sequences.append((types, times))

        if not sequences:
            return

        # Split train/val (90/10)
        np.random.seed(42)
        indices = np.random.permutation(len(sequences))
        val_size = max(1, len(sequences) // 10)
        val_idx = set(indices[:val_size])
        train_seqs = [sequences[i] for i in range(len(sequences)) if i not in val_idx]
        val_seqs = [sequences[i] for i in range(len(sequences)) if i in val_idx]

        device = self.device
        self.model = RMTPPNet(K, self.hidden_size).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        ce_loss = nn.CrossEntropyLoss()

        def _batch_loss(seqs):
            """Compute loss on a list of sequences (padded batch)."""
            max_len = max(len(s[0]) for s in seqs) - 1
            batch_types = torch.zeros(len(seqs), max_len, dtype=torch.long, device=device)
            batch_times = torch.zeros(len(seqs), max_len, device=device)
            target_types = torch.zeros(len(seqs), max_len, dtype=torch.long, device=device)
            target_times = torch.zeros(len(seqs), max_len, device=device)
            mask = torch.zeros(len(seqs), max_len, device=device)

            for i, (types, times) in enumerate(seqs):
                L = len(types) - 1
                batch_types[i, :L] = torch.tensor(types[:-1])
                batch_times[i, :L] = torch.tensor(times[:-1], dtype=torch.float32)
                target_types[i, :L] = torch.tensor(types[1:])
                target_times[i, :L] = torch.tensor(times[1:], dtype=torch.float32)
                mask[i, :L] = 1.0

            type_logits, time_preds = self.model(batch_types, batch_times)
            # Type loss
            type_loss = ce_loss(type_logits[mask.bool()], target_types[mask.bool()])
            # Time loss (raw MSE)
            time_loss = torch.mean((time_preds[mask.bool()] - target_times[mask.bool()]) ** 2)
            return type_loss + time_loss

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        batch_size = 32

        for epoch in range(self.epochs):
            self.model.train()
            np.random.shuffle(train_seqs)
            train_loss = 0.0
            n_batches = 0
            for b in range(0, len(train_seqs), batch_size):
                batch = train_seqs[b:b + batch_size]
                optimizer.zero_grad()
                loss = _batch_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss = _batch_loss(val_seqs).item() if val_seqs else 0.0

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restore best model
        if 'best_state' in dir():
            self.model.load_state_dict(best_state)
        self.model.eval()
        print(f"RMTPP trained: {epoch + 1} epochs, val_loss={best_val_loss:.4f}")

    def predict(self, history):
        if self.model is None or not HAS_TORCH:
            # Fallback
            types = [e["type_event"] for e in history]
            pred_type = Counter(types).most_common(1)[0][0]
            pred_time = float(np.mean([e["time_since_last_event"] for e in history[-10:]]))
            return pred_type, pred_time, "rmtpp: fallback (no model)"

        device = self.device
        types = [self.type_to_idx.get(e["type_event"], 0) for e in history]
        times = [e["time_since_last_event"] for e in history]

        with torch.no_grad():
            type_tensor = torch.tensor([types], dtype=torch.long, device=device)
            time_tensor = torch.tensor([times], dtype=torch.float32, device=device)
            type_logits, time_preds = self.model(type_tensor, time_tensor)
            # Take prediction from last position
            last_logits = type_logits[0, -1]
            pred_idx = int(torch.argmax(last_logits))
            pred_type = self.type_list[pred_idx]
            pred_time = max(0.0, float(time_preds[0, -1]))

        return pred_type, pred_time, f"rmtpp: GRU prediction (type_idx={pred_idx})"
