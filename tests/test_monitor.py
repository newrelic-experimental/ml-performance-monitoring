import os
import uuid
from unittest import mock

import numpy as np
import pytest

from ml_performance_monitoring.monitor import MLPerformanceMonitoring

metadata = {"environment": "aws", "dataset": "iris", "version": "1.0"}

monitor = MLPerformanceMonitoring(
    insert_key="NRII-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    model_name="Iris RandomForestClassifier",
    model_version="1.0.0",
    metadata=metadata,
)

monitor._record_events = mock.Mock()  # type: ignore


def setup_function(function):
    pass


def teardown_function(function):
    monitor._record_events.reset_mock()


def test_init_insert_key():
    if os.environ.get("NEW_RELIC_INSERT_KEY") is not None:
        del os.environ["NEW_RELIC_INSERT_KEY"]

    if os.environ.get("NEW_RELIC_LICENSE_KEY") is not None:
        del os.environ["NEW_RELIC_LICENSE_KEY"]

    with pytest.raises(Exception) as insert_key_type:
        MLPerformanceMonitoring(
            insert_key=123456789,
            model_name="Iris RandomForestClassifier",
            model_version="1.0.0",
            metadata=metadata,
        )
    assert (
        insert_key_type.value.args[0]
        == "insert_key instance type must be str and not None"
    )

    with pytest.raises(Exception) as insert_key_missing:
        MLPerformanceMonitoring(
            model_name="Iris RandomForestClassifier",
            model_version="1.0.0",
            metadata=metadata,
        )
    assert (
        insert_key_missing.value.args[0]
        == "insert_key instance type must be str and not None"
    )


def test_init_model_name():
    with pytest.raises(Exception) as model_name_missing:
        MLPerformanceMonitoring(
            insert_key="NRII-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            model_version="1.0.0",
            metadata=metadata,
        )
    assert (
        model_name_missing.value.args[0]
        == "MLPerformanceMonitoring.__init__() missing 1 required positional argument: 'model_name'"
    )

    with pytest.raises(Exception) as model_name_type:
        MLPerformanceMonitoring(
            insert_key="NRII-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            model_name=123456789,
            model_version="1.0.0",
            metadata=metadata,
        )
    assert (
        model_name_type.value.args[0]
        == "model_name instance type must be str and not empty"
    )

    with pytest.raises(Exception) as model_name_type:
        MLPerformanceMonitoring(
            insert_key="NRII-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            model_name="",
            model_version="1.0.0",
            metadata=metadata,
        )
    assert (
        model_name_type.value.args[0]
        == "model_name instance type must be str and not empty"
    )


def test_init_model_version():
    with pytest.raises(Exception) as model_version_missing:
        MLPerformanceMonitoring(
            insert_key="NRII-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            model_name="Iris RandomForestClassifier",
            metadata=metadata,
        )
    assert model_version_missing.value.args[0] == (
        "MLPerformanceMonitoring.__init__() missing 1 required positional argument: "
        "'model_version'"
    )

    with pytest.raises(Exception) as model_version_type:
        MLPerformanceMonitoring(
            insert_key="NRII-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            model_name="Iris RandomForestClassifier",
            model_version=123456789,
            metadata=metadata,
        )
    assert (
        model_version_type.value.args[0]
        == "model_version instance type must be str and not empty"
    )

    with pytest.raises(Exception) as model_version_type:
        MLPerformanceMonitoring(
            insert_key="NRII-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            model_name="Iris RandomForestClassifier",
            model_version="",
            metadata=metadata,
        )
    assert (
        model_version_type.value.args[0]
        == "model_version instance type must be str and not empty"
    )


def test_init_metadata():
    with pytest.raises(Exception) as metadata_type:
        MLPerformanceMonitoring(
            insert_key="NRII-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            model_name="Iris RandomForestClassifier",
            model_version="1.0.0",
            metadata=123456789,
        )
    assert (
        metadata_type.value.args[0]
        == "metadata instance type must be Dict[str, Any] or None"
    )


def test_init_output_type():
    assert isinstance(
        MLPerformanceMonitoring(
            insert_key="NRII-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            model_name="Iris RandomForestClassifier",
            model_version="1.0.0",
        ),
        MLPerformanceMonitoring,
    )


def test_record_inference_data_x_y_missing():
    with pytest.raises(Exception) as missing_X_y:
        monitor.record_inference_data()
    assert missing_X_y.value.args[0] == (
        "MLPerformanceMonitoring.record_inference_data() missing 2 required "
        "positional arguments: 'X' and 'y'"
    )

    with pytest.raises(Exception) as missing_y:
        monitor.record_inference_data(
            X=np.array([[11, 12, 5, 2], [15, 1, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]])
        )
    assert missing_y.value.args[0] == (
        "MLPerformanceMonitoring.record_inference_data() missing 1 required "
        "positional argument: 'y'"
    )

    with pytest.raises(Exception) as missing_X:
        monitor.record_inference_data(y=[11, 12, 5, 2, 4])
    assert missing_X.value.args[0] == (
        "MLPerformanceMonitoring.record_inference_data() missing 1 required "
        "positional argument: 'X'"
    )


def test_record_inference_data_x_y_type():
    with pytest.raises(Exception) as X_type:
        monitor.record_inference_data(X=123456789, y=np.array([11, 12, 5, 2, 4]))
    assert X_type.value.args[0] == "X instance type must be pd.DataFrame or np.ndarray"

    with pytest.raises(Exception) as y_type:
        monitor.record_inference_data(
            X=np.array(
                [[11, 12, 5, 2], [1, 15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]
            ),
            y=123456789,
        )
    assert y_type.value.args[0] == "y instance type must be pd.DataFrame or np.ndarray"


def test_record_inference_data_x_y_same_length():
    with pytest.raises(Exception) as X_y_length:
        monitor.record_inference_data(
            X=np.array(
                [[11, 12, 5, 2], [1, 15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]
            ),
            y=np.array([123456789]),
        )
    assert X_y_length.value.args[0] == "X and y must have the same length"


def test_record_inference_data_x_y_inference_metadata_same_length():
    with pytest.raises(Exception) as X_y_metadata_length:
        monitor.record_inference_data(
            X=np.array(
                [[11, 12, 5, 2], [1, 15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]
            ),
            y=np.array([11, 12, 5, 2]),
            inference_metadata=[{"index": 1}, {"index": 2}, {"index": 3}],
        )
    assert (
        X_y_metadata_length.value.args[0]
        == "inference_metadata must have the same length as X and y or have a length of 0"
    )


def test_record_inference_data_missing_inference_metadata():
    monitor.record_inference_data(
        X=np.array([[11, 12, 5, 2], [1, 15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]),
        y=np.array([11, 12, 5, 2]),
    )

    events = monitor._record_events.call_args[0][0]
    assert len(events) == 4


def test_record_inference_data():

    monitor.record_inference_data(
        X=np.array(
            [
                [11, 12, 5, 2],
                [1, 15, 6, 10],
                ["third", "third", "third", 2.3],
                [12, 15, 8, 6],
            ]
        ),
        y=np.array([11, 12, 5, 2]),
        inference_metadata=[
            {"index": 1},
            {"index": 2},
            {"index": 3, "test": "result"},
            {"index": 4},
        ],
    )

    events = monitor._record_events.call_args[0][0]

    assert len(events) == 4

    # makes sure that inference metadata is assigned properly
    assert (
        events[0]["index"] == 1
        and events[1]["index"] == 2
        and events[2]["index"] == 3
        and events[2]["test"] == "result"
        and events[3]["index"] == 4
    )

    # make sure each label is assigned to the correct features
    assert events[0]["feature.0"] == "11" and events[0]["label.0"] == 11
    assert events[1]["feature.0"] == "1" and events[1]["label.0"] == 12
    assert events[2]["feature.0"] == "third" and events[2]["label.0"] == 5
    assert events[3]["feature.0"] == "12" and events[3]["label.0"] == 2


def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False


def test_uuid_as_inference_id():
    prep_events_mock = mock.Mock()
    monitor.prepare_events = prep_events_mock

    monitor.record_inference_data(
        X=np.array([[11, 12, 5, 2], [1, 15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]),
        y=np.array([11, 12, 5, 2]),
    )

    inference_data = prep_events_mock.mock_calls[0][1][0]

    assert is_valid_uuid(inference_data["inference_id"].astype("str").iloc[0])
    assert len(inference_data["inference_id"].unique()) == 4


def test_metric_value(capsys):
    prep_events_mock = mock.Mock()
    monitor.prepare_events = prep_events_mock
    metrics = {
        "Accuracy": 0.9812,
        "Recall": "sadasd",
    }
    monitor.record_metrics(metrics=metrics)
    captured = capsys.readouterr()
    assert captured.out == (
        "Sending failed for metric Recall: value instance type must be int or float "
        "and not <class 'str'>\n"
    )
