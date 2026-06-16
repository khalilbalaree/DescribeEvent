# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""ARIMA baseline for inter-event time prediction with most-common type prediction."""

import warnings
import numpy as np
from collections import Counter
from baselines.statistical import BaselineMethod


class ARIMA(BaselineMethod):
    """ARIMA on inter-event times + most-common type prediction.

    Fits a low-order ARIMA model on the sequence's inter-event times.
    Type prediction uses the most frequent type in the sequence's history.
    Refits after each ground-truth append (handled externally by the prediction loop).
    """

    def __init__(self):
        self._best_order = None

    def predict(self, history):
        times = [e["time_since_last_event"] for e in history]
        pred_time = self._arima_forecast(times)
        types = [e["type_event"] for e in history]
        pred_type = Counter(types).most_common(1)[0][0]
        desc = f"arima: time={self._best_order}, type=most_common({pred_type})"
        return pred_type, pred_time, desc

    def _arima_forecast(self, times):
        """Fit ARIMA and forecast one step ahead."""
        try:
            from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
        except ImportError:
            # Fallback: just return mean
            return float(np.mean(times[-10:])) if times else 0.0

        if len(times) < 5:
            return float(np.mean(times)) if times else 0.0

        series = np.array(times, dtype=np.float64)
        best_aic = float('inf')
        best_forecast = float(np.mean(times[-10:]))
        self._best_order = "fallback"

        for p in [1, 2]:
            for d in [0, 1]:
                for q in [0, 1]:
                    if p == 0 and q == 0:
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = StatsARIMA(series, order=(p, d, q))
                            fit = model.fit(method_kwargs={"maxiter": 50})
                            if fit.aic < best_aic:
                                best_aic = fit.aic
                                forecast = fit.forecast(steps=1)[0]
                                best_forecast = max(0.0, float(forecast))
                                self._best_order = f"({p},{d},{q})"
                    except Exception:
                        continue

        return best_forecast
