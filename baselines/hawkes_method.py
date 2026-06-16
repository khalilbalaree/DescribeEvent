# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Multivariate Hawkes Process baseline.

Trains on pooled history from all test sequences. Uses conditional intensity
for type prediction and expected inter-arrival time.

MLE fitting uses PyTorch + CUDA for fast gradient-based optimization.
"""

import numpy as np
from baselines.statistical import BaselineMethod

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class Hawkes(BaselineMethod):
    """Multivariate Hawkes Process with exponential kernel.

    Intensity: λ_k(t) = μ_k + Σ_{t_j < t} α_{j_type, k} · exp(-β(t - t_j))
    Shared decay β across all pairs.
    """

    def __init__(self, max_events_per_seq=200, max_history=50, lr=0.05, max_iter=200):
        self.mu = None       # (K,) background rates
        self.alpha = None    # (K, K) excitation matrix
        self.beta = 1.0      # shared decay
        self.type_list = []  # ordered list of event types
        self.type_to_idx = {}
        self.max_events_per_seq = max_events_per_seq
        self.max_history = max_history
        self.lr = lr
        self.max_iter = max_iter
        self.device = torch.device("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")

    def fit(self, all_histories):
        """Train on pooled histories using MLE with PyTorch."""
        # Collect all types
        all_types = set()
        for hist in all_histories:
            for e in hist:
                all_types.add(e["type_event"])
        self.type_list = sorted(all_types)
        self.type_to_idx = {t: i for i, t in enumerate(self.type_list)}
        K = len(self.type_list)

        if K == 0:
            return

        # Convert histories to absolute timestamps
        sequences = []
        for hist in all_histories:
            if len(hist) < 2:
                continue
            abs_times = []
            types_idx = []
            t = 0.0
            for e in hist[:self.max_events_per_seq]:
                t += e["time_since_last_event"]
                abs_times.append(t)
                types_idx.append(self.type_to_idx.get(e["type_event"], 0))
            sequences.append((abs_times, types_idx))

        if not sequences:
            self.mu = np.ones(K) * 0.1
            self.alpha = np.ones((K, K)) * 0.01
            return

        if not HAS_TORCH:
            self._fit_empirical(sequences, K)
            return

        try:
            self._fit_torch(sequences, K)
        except Exception as e:
            print(f"Hawkes torch fitting failed ({e}), falling back to empirical")
            self._fit_empirical(sequences, K)

    def _fit_empirical(self, sequences, K):
        """Fallback: set parameters from empirical statistics."""
        type_counts = np.zeros(K)
        total_time = 0.0
        for times, types in sequences:
            T = times[-1] - times[0]
            if T > 0:
                total_time += T
                for k in range(K):
                    type_counts[k] += sum(1 for t in types if t == k)

        if total_time > 0:
            self.mu = type_counts / total_time
        else:
            self.mu = np.ones(K) * 0.1
        self.alpha = np.ones((K, K)) * 0.01
        self.beta = 1.0

    def _fit_torch(self, sequences, K):
        """Fit via MLE using PyTorch with CUDA acceleration."""
        device = self.device
        H = self.max_history
        print(f"Hawkes fitting on {device} with {len(sequences)} sequences, K={K}")

        # Pad sequences into tensors for batched computation
        # For each event i, we need: its type, its time, and the types/times of preceding events
        # Strategy: for each sequence, compute NLL contribution in a vectorized way

        # Precompute padded tensors: (num_seqs, max_len) for times and types, plus a mask
        max_len = max(len(t) for t, _ in sequences)
        max_len = min(max_len, self.max_events_per_seq)
        S = len(sequences)

        times_pad = torch.zeros(S, max_len, device=device)
        types_pad = torch.zeros(S, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(S, max_len, device=device)
        T_total = torch.zeros(S, device=device)

        for s, (times, types) in enumerate(sequences):
            n = min(len(times), max_len)
            times_pad[s, :n] = torch.tensor(times[:n], dtype=torch.float32)
            types_pad[s, :n] = torch.tensor(types[:n], dtype=torch.long)
            mask[s, :n] = 1.0
            T_total[s] = times[n - 1] - times[0] if n > 1 else 0.0

        # Normalize times to prevent large exponentials
        all_times_flat = times_pad[mask.bool()].detach()
        time_scale = all_times_flat.max().item() if all_times_flat.numel() > 0 else 1.0
        if time_scale <= 0:
            time_scale = 1.0
        times_pad = times_pad / time_scale
        T_total = T_total / time_scale

        # Learnable parameters (use softplus to ensure positivity)
        mu_raw = torch.zeros(K, device=device, requires_grad=True)
        alpha_raw = torch.full((K, K), -3.0, device=device, requires_grad=True)
        beta_raw = torch.tensor([1.0], device=device, requires_grad=True)

        optimizer = torch.optim.Adam([mu_raw, alpha_raw, beta_raw], lr=self.lr)

        best_nll = float('inf')
        patience = 0
        # Initialize best params to starting values
        best_mu = torch.nn.functional.softplus(mu_raw).detach().cpu().numpy()
        best_alpha = torch.nn.functional.softplus(alpha_raw).detach().cpu().numpy()
        best_beta = float((torch.nn.functional.softplus(beta_raw) + 0.01).detach().cpu())

        for it in range(self.max_iter):
            optimizer.zero_grad()

            mu = torch.nn.functional.softplus(mu_raw)          # (K,)
            alpha = torch.nn.functional.softplus(alpha_raw).clamp(max=10.0)  # (K, K)
            beta = torch.nn.functional.softplus(beta_raw) + 0.01  # scalar

            nll = torch.tensor(0.0, device=device)

            # Process in mini-batches of sequences to manage memory
            batch_size = 64
            for b in range(0, S, batch_size):
                b_end = min(b + batch_size, S)
                b_times = times_pad[b:b_end]       # (B, L)
                b_types = types_pad[b:b_end]       # (B, L)
                b_mask = mask[b:b_end]             # (B, L)
                b_T = T_total[b:b_end]             # (B,)
                B, L = b_times.shape

                # --- Log-likelihood of event times ---
                # For each event i (starting from 1), compute λ(t_i)
                # λ_k(t_i) = μ_k + Σ_{j<i} α_{type_j, k} * exp(-β * (t_i - t_j))
                # We only need λ_{type_i}(t_i)

                # Compute pairwise dt: (B, L, L) where dt[b,i,j] = t_i - t_j
                # Only use j in [max(0, i-H), i) for memory efficiency
                # Use a sliding window approach

                # For tractability, compute intensity at each event using last H events
                log_lam_sum = torch.tensor(0.0, device=device)
                for i in range(1, L):
                    event_mask = b_mask[:, i]  # (B,) which seqs have event at pos i
                    if event_mask.sum() == 0:
                        continue

                    j_start = max(0, i - H)
                    # dt: (B, window)
                    dt = b_times[:, i:i+1] - b_times[:, j_start:i]  # (B, window)
                    j_mask = b_mask[:, j_start:i]  # (B, window)

                    # Excitation from past events: α[type_j, type_i] * exp(-β * dt)
                    j_types = b_types[:, j_start:i]  # (B, window)
                    i_type = b_types[:, i]            # (B,)

                    # Gather α[j_type, i_type] for each (b, j)
                    alpha_vals = alpha[j_types, i_type.unsqueeze(1).expand_as(j_types)]  # (B, window)
                    excitation = alpha_vals * torch.exp((-beta * dt).clamp(min=-20, max=0)) * j_mask  # (B, window)

                    # Base rate for event type
                    mu_i = mu[i_type]  # (B,)

                    lam_i = mu_i + excitation.sum(dim=1)  # (B,)
                    log_lam_sum = log_lam_sum + (torch.log(lam_i.clamp(min=1e-8)) * event_mask).sum()

                # --- Integral term: ∫ Σ_k λ_k(t) dt ---
                # = Σ_k μ_k * T + Σ_{i} Σ_k α[type_i, k] / β * (1 - exp(-β * (T - t_i + t_0)))
                integral = (mu.sum() * b_T).sum()

                for i in range(L):
                    event_mask = b_mask[:, i]
                    if event_mask.sum() == 0:
                        continue
                    dt_end = b_T - (b_times[:, i] - b_times[:, 0])  # (B,)
                    dt_end = dt_end.clamp(min=0)
                    i_types = b_types[:, i]  # (B,)
                    # Σ_k α[type_i, k] / β * (1 - exp(-β * dt_end))
                    alpha_sum = alpha[i_types, :].sum(dim=1)  # (B,)
                    contrib = alpha_sum / beta * (1 - torch.exp(-beta * dt_end))
                    integral = integral + (contrib * event_mask).sum()

                nll = nll - log_lam_sum + integral

            nll_val = nll.item()
            if not np.isfinite(nll_val):
                # Reset to best params and reduce lr
                with torch.no_grad():
                    mu_raw.copy_(torch.tensor(np.log(np.exp(best_mu) - 1 + 1e-8), device=device))
                    alpha_raw.copy_(torch.tensor(np.log(np.exp(best_alpha) - 1 + 1e-8), device=device))
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.5
                patience += 5
                if patience >= 20:
                    print(f"  Early stopping at iter {it+1} (NaN)")
                    break
                continue

            nll.backward()
            torch.nn.utils.clip_grad_norm_([mu_raw, alpha_raw, beta_raw], 5.0)
            optimizer.step()

            if (it + 1) % 20 == 0:
                print(f"  Hawkes iter {it+1}/{self.max_iter}: NLL={nll_val:.2f}")

            if nll_val < best_nll - 1e-3:
                best_nll = nll_val
                patience = 0
                best_mu = torch.nn.functional.softplus(mu_raw).detach().cpu().numpy()
                best_alpha = torch.nn.functional.softplus(alpha_raw).detach().cpu().numpy()
                best_beta = float((torch.nn.functional.softplus(beta_raw) + 0.01).detach().cpu())
            else:
                patience += 1
                if patience >= 20:
                    print(f"  Early stopping at iter {it+1}")
                    break

        # Rescale parameters back to original time units
        self.mu = best_mu / time_scale
        self.alpha = best_alpha / time_scale
        self.beta = best_beta / time_scale
        print(f"Hawkes fitted: β={self.beta:.4f}, μ_range=[{self.mu.min():.4f}, {self.mu.max():.4f}]")

    def predict(self, history):
        K = len(self.type_list)
        if K == 0 or not history:
            return history[-1]["type_event"] if history else "unknown", 0.0, "hawkes: no types"

        # Convert history to absolute timestamps
        abs_times = []
        types_idx = []
        t = 0.0
        for e in history:
            t += e["time_since_last_event"]
            abs_times.append(t)
            idx = self.type_to_idx.get(e["type_event"])
            types_idx.append(idx if idx is not None else 0)

        current_t = abs_times[-1]

        # Compute conditional intensities at current time (use last 50 events)
        intensities = self.mu.copy()
        for j in range(max(0, len(abs_times) - 50), len(abs_times)):
            dt = current_t - abs_times[j]
            if dt > 0:
                intensities += self.alpha[types_idx[j], :] * np.exp(-self.beta * dt)

        # Predict type with highest intensity
        pred_idx = int(np.argmax(intensities))
        pred_type = self.type_list[pred_idx]

        # Predict time: expected inter-arrival = 1 / total intensity
        total_intensity = float(np.sum(intensities))
        pred_time = max(0.0, 1.0 / total_intensity) if total_intensity > 0 else 0.0

        return pred_type, pred_time, f"hawkes: λ={intensities[pred_idx]:.3f}, Σλ={total_intensity:.3f}"
