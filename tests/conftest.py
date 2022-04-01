from typing import Union

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def record_event_monkeypatch(monkeypatch):
    def record_event_monkeypatch_inner(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
        *,
        calling_method=None,
        inference_identifier=None,
        data_summary_min_rows: int = 100,
    ):
        return None

    monkeypatch.setattr(
        "ml_performance_monitoring.monitor.MLPerformanceMonitoring._record_event",
        record_event_monkeypatch_inner,
    )
