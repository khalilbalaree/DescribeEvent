# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Statistical baseline methods: most_common, last_event, markov, knn_subseq."""

from collections import Counter, defaultdict
import numpy as np


class BaselineMethod:
    """Base class for all baseline methods."""

    def fit(self, all_histories):
        """Train on pooled histories (no-op for statistical methods)."""
        pass

    def predict(self, history):
        """Given a sequence's current history, return (pred_type, pred_time, description)."""
        raise NotImplementedError


class MostCommon(BaselineMethod):
    """Predict the most frequent type in history. Time = mean of last 10 inter-event times."""

    def predict(self, history):
        types = [e["type_event"] for e in history]
        type_counts = Counter(types)
        pred_type = type_counts.most_common(1)[0][0]
        recent_times = [e["time_since_last_event"] for e in history[-10:]]
        pred_time = float(np.mean(recent_times)) if recent_times else 0.0
        return pred_type, pred_time, "most_common: majority type + mean of last 10 times"


class LastEvent(BaselineMethod):
    """Predict same type and time as the previous event."""

    def predict(self, history):
        last = history[-1]
        return (last["type_event"], last["time_since_last_event"],
                "last_event: repeat previous event type and time")


class Markov(BaselineMethod):
    """First-order Markov chain on event types with mean transition times."""

    def predict(self, history):
        # Build transition counts and times
        transition_counts = defaultdict(Counter)
        transition_times = defaultdict(list)
        for i in range(1, len(history)):
            src = history[i - 1]["type_event"]
            dst = history[i]["type_event"]
            transition_counts[src][dst] += 1
            transition_times[(src, dst)].append(history[i]["time_since_last_event"])

        current_type = history[-1]["type_event"]
        if current_type in transition_counts:
            pred_type = transition_counts[current_type].most_common(1)[0][0]
            times = transition_times[(current_type, pred_type)]
            pred_time = float(np.mean(times))
            desc = f"markov: P({pred_type}|{current_type}) transition"
        else:
            # Fallback to most common
            types = [e["type_event"] for e in history]
            pred_type = Counter(types).most_common(1)[0][0]
            recent_times = [e["time_since_last_event"] for e in history[-10:]]
            pred_time = float(np.mean(recent_times))
            desc = "markov: fallback to most_common (unseen source type)"
        return pred_type, pred_time, desc


class KNNSubseq(BaselineMethod):
    """K-nearest subsequence matching within the sequence's own history."""

    def __init__(self, window_size=5, k=3, type_weight=0.7):
        self.W = window_size
        self.k = k
        self.type_weight = type_weight
        self.time_weight = 1.0 - type_weight

    def predict(self, history):
        W = self.W
        if len(history) < W + 1:
            # Not enough history for subsequence matching, fallback
            types = [e["type_event"] for e in history]
            pred_type = Counter(types).most_common(1)[0][0]
            pred_time = float(np.mean([e["time_since_last_event"] for e in history[-10:]]))
            return pred_type, pred_time, "knn_subseq: fallback (insufficient history)"

        # Current query window = last W events
        query_types = [e["type_event"] for e in history[-W:]]
        query_times = [e["time_since_last_event"] for e in history[-W:]]

        # Compute time stats for normalization
        all_times = [e["time_since_last_event"] for e in history]
        time_std = float(np.std(all_times)) if len(all_times) > 1 else 1.0
        if time_std == 0:
            time_std = 1.0

        # Slide window over history (excluding the last W events which are the query)
        candidates = []
        for i in range(len(history) - W - 1):
            window_types = [e["type_event"] for e in history[i:i + W]]
            window_times = [e["time_since_last_event"] for e in history[i:i + W]]

            # Type similarity: fraction of matching types
            type_sim = sum(1 for a, b in zip(query_types, window_types) if a == b) / W
            # Time similarity: negative normalized distance
            time_dist = sum(abs(a - b) for a, b in zip(query_times, window_times)) / (W * time_std)
            time_sim = 1.0 / (1.0 + time_dist)

            score = self.type_weight * type_sim + self.time_weight * time_sim
            # The event following this window
            next_event = history[i + W]
            candidates.append((score, next_event))

        if not candidates:
            types = [e["type_event"] for e in history]
            pred_type = Counter(types).most_common(1)[0][0]
            pred_time = float(np.mean([e["time_since_last_event"] for e in history[-10:]]))
            return pred_type, pred_time, "knn_subseq: fallback (no candidates)"

        # Top-k by score
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_k = candidates[:self.k]

        # Majority vote for type
        type_votes = Counter(e["type_event"] for _, e in top_k)
        pred_type = type_votes.most_common(1)[0][0]
        # Mean time of top-k
        pred_time = float(np.mean([e["time_since_last_event"] for _, e in top_k]))

        return pred_type, pred_time, f"knn_subseq: top-{len(top_k)} match (best={top_k[0][0]:.3f})"
