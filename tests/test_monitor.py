import numpy as np
import pytest

from ml_performance_monitoring.monitor import MLPerformanceMonitoring

metadata = {"environment": "aws", "dataset": "iris", "version": "1.0"}

monitor = MLPerformanceMonitoring(
    insert_key="NRII-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    model_name="Iris RandomForestClassifier",
    metadata=metadata,
    label_type="categorical",
)


def test_init_insert_key():
    with pytest.raises(Exception) as insert_key_type:
        MLPerformanceMonitoring(
            insert_key=123456789,
            model_name="Iris RandomForestClassifier",
            metadata=metadata,
            label_type="categorical",
        )
    assert (
        insert_key_type.value.args[0]
        == "insert_key instance type must str and not None"
    )

    with pytest.raises(Exception) as insert_key_missing:
        MLPerformanceMonitoring(
            model_name="Iris RandomForestClassifier",
            metadata=metadata,
            label_type="categorical",
        )
    assert (
        insert_key_missing.value.args[0]
        == "insert_key instance type must str and not None"
    )


def test_init_model_name():
    with pytest.raises(Exception) as model_name_missing:
        MLPerformanceMonitoring(
            insert_key="NRII-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            metadata=metadata,
            label_type="categorical",
        )
    assert (
        model_name_missing.value.args[0]
        == "__init__() missing 1 required positional argument: 'model_name'"
    )

    with pytest.raises(Exception) as model_name_type:
        MLPerformanceMonitoring(
            insert_key="NRII-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            model_name=123456789,
            metadata=metadata,
            label_type="categorical",
        )
    assert model_name_type.value.args[0] == "model_name instance type must be str"


def test_init_metadata():
    with pytest.raises(Exception) as metadata_type:
        MLPerformanceMonitoring(
            insert_key="NRII-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            model_name="Iris RandomForestClassifier",
            metadata=123456789,
            label_type="categorical",
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
            label_type="categorical",
        ),
        MLPerformanceMonitoring,
    )


def test_record_inference_data_x_y_missing():
    with pytest.raises(Exception) as missing_X_y:
        monitor.record_inference_data()
    assert (
        missing_X_y.value.args[0]
        == "record_inference_data() missing 2 required positional arguments: 'X' and 'y'"
    )

    with pytest.raises(Exception) as missing_y:
        monitor.record_inference_data(
            X=np.array([[11, 12, 5, 2], [15, 1, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]])
        )
    assert (
        missing_y.value.args[0]
        == "record_inference_data() missing 1 required positional argument: 'y'"
    )

    with pytest.raises(Exception) as missing_X:
        monitor.record_inference_data(y=[11, 12, 5, 2, 4])
    assert (
        missing_X.value.args[0]
        == "record_inference_data() missing 1 required positional argument: 'X'"
    )


def test_record_inference_data_x_y_type():
    with pytest.raises(Exception) as X_type:
        monitor.record_inference_data(X=123456789, y=np.array([11, 12, 5, 2, 4]))
    assert (
        X_type.value.args[0]
        == "X instance type must be pd.core.frame.DataFrame or np.ndarray"
    )

    with pytest.raises(Exception) as y_type:
        monitor.record_inference_data(
            X=np.array(
                [[11, 12, 5, 2], [1, 15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]
            ),
            y=123456789,
        )
    assert (
        y_type.value.args[0]
        == "y instance type must be pd.core.frame.DataFrame or np.ndarray"
    )


def test_record_inference_data_x_y_same_length():
    with pytest.raises(Exception) as X_y_length:
        monitor.record_inference_data(
            X=np.array(
                [[11, 12, 5, 2], [1, 15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]
            ),
            y=np.array([123456789]),
        )
    assert X_y_length.value.args[0] == "X and y must have the same length"


def test_record_inference_data():
    monitor.record_inference_data(
        X=np.array([[11, 12, 5, 2], [1, 15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]),
        y=np.array([11, 12, 5, 2]),
    )
