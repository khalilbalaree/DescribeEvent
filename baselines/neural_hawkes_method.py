# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Neural Hawkes Process baseline (Mei & Eisner, NIPS 2017).

Uses a continuous-time LSTM where the hidden state decays between events,
and the intensity is computed as softplus(linear(h(t))).
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
    class CTLSTMCell(nn.Module):
        """Continuous-Time LSTM cell.

        At each event, updates like a standard LSTM.
        Between events, the cell state decays toward a steady state:
            c(t) = c_bar + (c_i - c_bar) * exp(-delta_i * (t - t_i))
        where c_bar is the steady-state cell, c_i is the cell right after event i,
        and delta_i controls the decay rate.
        """

        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            # Standard LSTM gates + decay gate
            self.linear = nn.Linear(input_size + hidden_size, hidden_size * 7)

        def forward(self, x, h_prev, c_prev, c_bar_prev):
            """
            x: (batch, input_size) - input at current event
            h_prev: (batch, hidden_size) - hidden state before this event
            c_prev: (batch, hidden_size) - cell state before this event (after decay)
            c_bar_prev: (batch, hidden_size) - previous steady-state cell
            Returns: h, c, c_bar, delta
            """
            combined = torch.cat([x, h_prev], dim=1)
            gates = self.linear(combined)
            HS = self.hidden_size

            i_gate = torch.sigmoid(gates[:, :HS])
            f_gate = torch.sigmoid(gates[:, HS:2*HS])
            z_gate = torch.tanh(gates[:, 2*HS:3*HS])
            o_gate = torch.sigmoid(gates[:, 3*HS:4*HS])
            i_bar = torch.sigmoid(gates[:, 4*HS:5*HS])  # input gate for steady state
            f_bar = torch.sigmoid(gates[:, 5*HS:6*HS])  # forget gate for steady state
            delta = torch.nn.functional.softplus(gates[:, 6*HS:7*HS])  # decay rate

            # Update cell state (at event time)
            c = f_gate * c_prev + i_gate * z_gate
            # Update steady-state cell
            c_bar = f_bar * c_bar_prev + i_bar * z_gate
            # Hidden state at event time
            h = o_gate * torch.tanh(c)

            return h, c, c_bar, delta

        def decay(self, c, c_bar, delta, dt):
            """Decay cell state over time interval dt.

            c(t) = c_bar + (c - c_bar) * exp(-delta * dt)
            """
            c_decay = c_bar + (c - c_bar) * torch.exp(-delta * dt.unsqueeze(1))
            return c_decay

    class NeuralHawkesNet(nn.Module):
        """Neural Hawkes Process network."""

        def __init__(self, num_types, hidden_size=32):
            super().__init__()
            self.num_types = num_types
            self.hidden_size = hidden_size
            # Input: one-hot type + time_delta
            input_size = num_types + 1
            self.ct_lstm = CTLSTMCell(input_size, hidden_size)
            # Intensity: softplus(linear(h))
            self.intensity_layer = nn.Linear(hidden_size, num_types)

        def forward(self, type_indices, time_deltas, mask):
            """
            type_indices: (batch, seq_len) LongTensor
            time_deltas: (batch, seq_len) FloatTensor
            mask: (batch, seq_len) FloatTensor
            Returns: all_intensities (batch, seq_len, num_types)
            """
            B, L = type_indices.shape
            device = type_indices.device
            HS = self.hidden_size

            h = torch.zeros(B, HS, device=device)
            c = torch.zeros(B, HS, device=device)
            c_bar = torch.zeros(B, HS, device=device)
            delta = torch.ones(B, HS, device=device) * 0.1

            all_intensities = []

            for t in range(L):
                # Decay hidden state by dt before this event
                dt = time_deltas[:, t]
                c_decayed = self.ct_lstm.decay(c, c_bar, delta, dt)
                h_decayed = torch.tanh(c_decayed)  # approximate h after decay

                # Compute intensity at this event time (before update)
                lam = torch.nn.functional.softplus(self.intensity_layer(h_decayed))
                all_intensities.append(lam)

                # Build input
                one_hot = torch.zeros(B, self.num_types, device=device)
                one_hot.scatter_(1, type_indices[:, t:t+1], 1.0)
                x = torch.cat([one_hot, dt.unsqueeze(1)], dim=1)

                # Update CT-LSTM
                h, c, c_bar, delta = self.ct_lstm(x, h_decayed, c_decayed, c_bar)

            return torch.stack(all_intensities, dim=1)  # (B, L, K)

        def get_state(self, type_indices, time_deltas):
            """Process a single sequence and return final state for prediction."""
            B, L = type_indices.shape
            device = type_indices.device
            HS = self.hidden_size

            h = torch.zeros(B, HS, device=device)
            c = torch.zeros(B, HS, device=device)
            c_bar = torch.zeros(B, HS, device=device)
            delta = torch.ones(B, HS, device=device) * 0.1

            for t in range(L):
                dt = time_deltas[:, t]
                c_decayed = self.ct_lstm.decay(c, c_bar, delta, dt)
                h_decayed = torch.tanh(c_decayed)

                one_hot = torch.zeros(B, self.num_types, device=device)
                one_hot.scatter_(1, type_indices[:, t:t+1], 1.0)
                x = torch.cat([one_hot, dt.unsqueeze(1)], dim=1)

                h, c, c_bar, delta = self.ct_lstm(x, h_decayed, c_decayed, c_bar)

            return h, c, c_bar, delta


class NeuralHawkes(BaselineMethod):
    """Neural Hawkes Process with continuous-time LSTM."""

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
            print("WARNING: torch not available, NeuralHawkes will fallback")
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

        # Prepare sequences
        sequences = []
        for hist in all_histories:
            if len(hist) < 3:
                continue
            types = [self.type_to_idx[e["type_event"]] for e in hist]
            times = [e["time_since_last_event"] for e in hist]
            sequences.append((types, times))

        if not sequences:
            return

        # Split train/val
        np.random.seed(42)
        indices = np.random.permutation(len(sequences))
        val_size = max(1, len(sequences) // 10)
        val_idx = set(indices[:val_size])
        train_seqs = [sequences[i] for i in range(len(sequences)) if i not in val_idx]
        val_seqs = [sequences[i] for i in range(len(sequences)) if i in val_idx]

        device = self.device
        self.model = NeuralHawkesNet(K, self.hidden_size).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        def _batch_nll(seqs):
            """Compute negative log-likelihood for a batch."""
            max_len = max(len(s[0]) for s in seqs) - 1
            B = len(seqs)
            types_in = torch.zeros(B, max_len, dtype=torch.long, device=device)
            times_in = torch.zeros(B, max_len, device=device)
            types_target = torch.zeros(B, max_len, dtype=torch.long, device=device)
            mask = torch.zeros(B, max_len, device=device)

            for i, (types, times) in enumerate(seqs):
                L = len(types) - 1
                types_in[i, :L] = torch.tensor(types[:-1])
                times_in[i, :L] = torch.tensor(times[:-1], dtype=torch.float32)
                types_target[i, :L] = torch.tensor(types[1:])
                mask[i, :L] = 1.0

            # Forward pass: get intensities at each position
            intensities = self.model(types_in, times_in, mask)  # (B, L, K)

            # Log-likelihood of observed types
            # Gather intensity of the true next type
            target_intensity = intensities.gather(2, types_target.unsqueeze(2)).squeeze(2)  # (B, L)
            log_intensity = torch.log(target_intensity.clamp(min=1e-8))

            # Total intensity (for normalization / integral approximation)
            total_intensity = intensities.sum(dim=2)  # (B, L)

            # NLL = -log λ_k(t) + Σ_k λ_k(t) * dt_next (integral approximation)
            # Use time to next event for integral
            times_next = torch.zeros(B, max_len, device=device)
            for i, (types, times) in enumerate(seqs):
                L = len(types) - 1
                times_next[i, :L] = torch.tensor(times[1:], dtype=torch.float32)

            nll = (-log_intensity + total_intensity * times_next) * mask
            return nll.sum() / mask.sum()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        batch_size = 32
        best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        for epoch in range(self.epochs):
            self.model.train()
            np.random.shuffle(train_seqs)
            for b in range(0, len(train_seqs), batch_size):
                batch = train_seqs[b:b + batch_size]
                optimizer.zero_grad()
                loss = _batch_nll(batch)
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss = _batch_nll(val_seqs).item() if val_seqs else 0.0

            if np.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()
        print(f"NeuralHawkes trained: {epoch + 1} epochs, val_loss={best_val_loss:.4f}")

    def predict(self, history):
        if self.model is None or not HAS_TORCH:
            types = [e["type_event"] for e in history]
            pred_type = Counter(types).most_common(1)[0][0]
            pred_time = float(np.mean([e["time_since_last_event"] for e in history[-10:]]))
            return pred_type, pred_time, "neural_hawkes: fallback (no model)"

        device = self.device
        types = [self.type_to_idx.get(e["type_event"], 0) for e in history]
        times = [e["time_since_last_event"] for e in history]

        with torch.no_grad():
            type_tensor = torch.tensor([types], dtype=torch.long, device=device)
            time_tensor = torch.tensor([times], dtype=torch.float32, device=device)

            # Get final state after processing history
            h, c, c_bar, delta = self.model.get_state(type_tensor, time_tensor)

            # Compute intensity at current time (dt=0, right after last event)
            # Use a small dt to get the intensity just after the last event
            intensities = torch.nn.functional.softplus(self.model.intensity_layer(h))  # (1, K)

            pred_idx = int(torch.argmax(intensities[0]))
            pred_type = self.type_list[pred_idx]

            # Time prediction: 1 / total intensity
            total_intensity = float(intensities.sum())
            pred_time = max(0.0, 1.0 / total_intensity) if total_intensity > 0 else 0.0

        return pred_type, pred_time, f"neural_hawkes: λ_max={float(intensities[0, pred_idx]):.3f}, Σλ={total_intensity:.3f}"
