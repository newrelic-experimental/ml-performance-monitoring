import atexit
import datetime
import logging
import os
import uuid
import warnings
from enum import Enum, EnumMeta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from newrelic_telemetry_sdk import (
    EventBatch,
    EventClient,
    GaugeMetric,
    Harvester,
    MetricBatch,
    MetricClient,
)
from newrelic_telemetry_sdk.event import Event

logger = logging.getLogger("ml_performance_monitoring")


class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)  # type: ignore
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    pass


class LabelType(str, BaseEnum):
    categorical = "categorical"
    numeric = "numeric"


FEATURE_TYPE = {
    "bool": 1,
    "numeric": 2,
    "datetime": 3,
    "categorical": 4,
}

EventName = "InferenceData"


class MLPerformanceMonitoring:

    # this class uses the telemetry SDK to record metrics to new relic, please see https://github.com/newrelic/newrelic-telemetry-sdk-python
    def __init__(
        self,
        model_name: str,
        model_version: str,
        insert_key: Optional[str] = None,
        metadata: Dict[str, Any] = {},
        staging: bool = False,
        model=None,
        send_inference_data=True,
        send_data_metrics=False,
        features_columns: Optional[List[str]] = None,
        labels_columns: Optional[List[str]] = None,
        label_type: Optional[str] = None,
        event_client_host: Optional[str] = None,
        metric_client_host: Optional[str] = None,
        use_logger: Optional[bool] = None,
    ):

        if not isinstance(model_name, str) or not model_name:
            raise TypeError("model_name instance type must be str and not empty")
        if not isinstance(model_version, str) or not model_version:
            raise TypeError("model_version instance type must be str and not empty")
        if not isinstance(metadata, Dict) and metadata is not None:
            raise TypeError("metadata instance type must be Dict[str, Any] or None")
        if not isinstance(staging, bool):
            raise TypeError("staging instance type must be bool")
        if not isinstance(features_columns, List) and features_columns is not None:
            raise TypeError("features_columns instance type must be List[str]")
        if not isinstance(labels_columns, List) and labels_columns is not None:
            raise TypeError("labels_columns instance type must be List[str]")
        if not isinstance(event_client_host, str) and event_client_host is not None:
            raise TypeError("event_client_host instance type must be str or None")
        if not isinstance(metric_client_host, str) and metric_client_host is not None:
            raise TypeError("metric_client_host instance type must be str or None")
        if label_type not in LabelType:
            raise TypeError(
                f"label_type instance must be one of the values: {[e.value for e in LabelType]}"
            )

        self.event_client_host = event_client_host or os.getenv(
            "EVENT_CLIENT_HOST", EventClient.HOST
        )

        self.metric_client_host = metric_client_host or os.getenv(
            "METRIC_CLIENT_HOST", MetricClient.HOST
        )

        self._set_insert_key(insert_key)
        self.model = model
        self.model_version = model_version
        self.send_inference_data = send_inference_data
        self.send_data_metrics = send_data_metrics
        self.model_name = model_name
        self.static_metadata = metadata
        self.features_columns = features_columns
        self.labels_columns = labels_columns
        self.label_type = label_type
        self.use_logger = use_logger if use_logger is not None else False

    def _set_insert_key(
        self,
        insert_key: Optional[str] = None,
    ):
        self.insert_key = insert_key or os.getenv("NEW_RELIC_INSERT_KEY")  # type: ignore

        if (
            not isinstance(self.insert_key, str) and self.insert_key is not None
        ) or self.insert_key is None:
            raise TypeError("insert_key instance type must be str and not None")
        self._start()

    def _log(self, msg: str):
        if self.use_logger:
            logger.info(msg)
        else:
            print(msg)

    # initialize event thread
    def _start(self):

        # the API requires the new relic insert key https://docs.newrelic.com/docs/apis/intro-apis/new-relic-api-keys/#insights-insert-key

        # define a Metric client & batch for sending gauge metrics, you usually want to do this high volume inference in production. Metrics are historically aggregated in NRDB.
        # for metric data types please see https://docs.newrelic.com/docs/telemetry-data-platform/understand-data/metric-data/metric-data-type/

        # Client is the HTTP connection to New Relic (HTTP API level)
        self.metric_client = MetricClient(
            self.insert_key,
            host=self.metric_client_host,
        )

        # Storage for metrics + aggregation (I1(v1) + I1(v2) = I1(v1+v2))
        self.metric_batch = MetricBatch()

        # Background thread that flushes the batch every 1 seconds into the client, harvest_interval=5 as a default
        # the 100K limit may not be the first one we hit, we may hit the 1MB of compressed data, so clearing the memory faster
        # can be done with a higher harvest interval, i.e., 1 second instead of 5.
        self.metric_harvester = Harvester(
            self.metric_client, self.metric_batch, harvest_interval=5
        )

        # In this tutorial we will use Custom Events.
        # For more information please see https://docs.newrelic.com/docs/telemetry-data-platform/ingest-apis/introduction-event-api/
        self.event_client = EventClient(
            self.insert_key,
            host=self.event_client_host,
        )
        self.event_batch = EventBatch()

        # Background thread that flushes the batch
        self.event_harvester = Harvester(self.event_client, self.event_batch)

        # This starts the thread
        self.metric_harvester.start()
        self.event_harvester.start()

        # When the process exits, run the harvester.stop() method before terminating the process
        # Why? To send the remaining data...
        atexit.register(self.metric_harvester.stop)
        atexit.register(self.event_harvester.stop)

    # custom events can be queried by using the following NRQL query template "SELECT * FROM MyTable" i.e., where MyTable==table_name
    def _record_events(self, events, table: str):
        for event in events:
            event["eventType"] = table
            self.event_batch.record(event)

    def _record_metrics(self, metrics):
        for metric in metrics:
            self.metric_batch.record_gauge(metric["name"], metric.value, metric.tags)

    def set_model(self, model):
        self.model = model

    def _calc_descriptive_statistics(self, df):
        df_statistics = df.describe(include="all")
        df_statistics = df_statistics.append(
            df.nunique()
            .rename("uniques")
            .to_frame()
            .assign(missing=df.isna().sum())
            .assign(
                missing_perc=lambda x: (x.missing / len(df) * 100),
                types=self._calc_columns_types(df),
            )
            .T
        )
        df_statistics = df_statistics.drop(["unique", "top", "freq"], errors="ignore")
        return df_statistics

    def _calc_columns_types(self, df):
        columns_types = (
            df.dtypes.apply(str)
            .replace(
                {r"^(float|int).*": "numeric", "object": "categorical"}, regex=True
            )
            .rename("types")
        )
        return columns_types

    def get_suffix(self) -> str:
        return str(uuid.uuid4())[:4]

    def get_request_id(self) -> str:
        return str(uuid.uuid4())[:18]

    def get_version(self) -> str:
        return datetime.datetime.today().strftime("%Y.%m.%d")

    def tuple_to_event(
        self,
        t: Tuple[Any, ...],
        columns: Sequence[str],
        request_id: str,
        metadata: Dict[str, Any],
        timestamp: Optional[int] = None,
        event_name: str = EventName,
        params: Optional[Dict[str, Any]] = None,
    ) -> Event:
        d = dict(zip(columns, t))
        d["instrumentation.provider"] = "nr_performance_monitoring"

        d["request_id"] = request_id
        d["model_version"] = self.model_version
        d.update(metadata)
        if params is not None:
            for k in params:
                d[f"parmas_{k}"] = params[k]

        return Event(event_name, d, timestamp_ms=timestamp)

    def prepare_events(
        self,
        flat: pd.DataFrame,
        y: pd.DataFrame,
        metadata: Dict[str, Any],
        inference_id_to_inference_metadata: Dict[str, Dict[str, str]],
        timestamp: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Event]:
        events: List[Event] = []

        request_id = self.get_request_id()
        for t in flat.itertuples(index=False, name=None):
            curr_inference_metadata = (
                inference_id_to_inference_metadata[t[0]]
                if t[0] in inference_id_to_inference_metadata
                else {}
            )
            events.append(
                self.tuple_to_event(
                    t,
                    flat.columns.to_list(),
                    request_id,
                    {**metadata, **curr_inference_metadata},
                    timestamp,
                )
            )

        for s in y.itertuples(index=False, name=None):
            events.append(
                self.tuple_to_event(
                    s,
                    y.columns.to_list(),
                    request_id,
                    metadata,
                    timestamp,
                    params=params,
                )
            )

        return events

    def record_inference_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
        *,
        inference_metadata: List[Dict[str, str]] = [],
        data_summary_min_rows: int = 100,
        timestamp: Optional[int] = None,
    ):
        """Send inference data to the New Relic Database.
        Tha data could be viewed on the "Model performence" page and in the table "InferenceData" in New Relic NRDB


        Args:
            X (Union[pd.DataFrame, np.ndarray]): A list of features
            y (Union[pd.DataFrame, np.ndarray]): A list of the resulting labels
            inference_metadata (List[Dict[str, str]], optional): Additional data about the each inference, the
                data will be added to each inference in the NRDB. Each Item in the list is a Dict of key-value items, describing
                A single inference. Defaults to None.
        """
        self.static_metadata.update(
            {
                "modelName": self.model_name,
                "instrumentation.provider": "nr_performance_monitoring",
            }
        )
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X instance type must be pd.DataFrame or np.ndarray")
        if not isinstance(y, (pd.DataFrame, np.ndarray)):
            raise TypeError("y instance type must be pd.DataFrame or np.ndarray")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        elif len(inference_metadata) > 0 and len(inference_metadata) != len(X):
            raise ValueError(
                "inference_metadata must have the same length as X and y or have a length of 0"
            )
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                list(map(np.ravel, X)),
                columns=self.features_columns
                if self.features_columns
                else [str(int) for int in range(len(X[0]))],
            )

        columns_types = self._calc_columns_types(
            X.drop(columns=["index"], errors="ignore")
        )

        X_df = X.stack().reset_index()
        X_df.columns = ["inference_id", "feature_name", "feature_value"]
        X_df["feature_type"] = X_df["feature_name"].map(columns_types)
        infid_to_uuid = {
            infid: str(uuid.uuid4()) for infid in X_df["inference_id"].unique()
        }
        inference_id_to_inference_metadata = dict(
            zip(infid_to_uuid.values(), inference_metadata)
        )
        X_df["inference_id"] = X_df["inference_id"].apply(lambda x: infid_to_uuid[x])
        X_df["batch.index"] = X_df.groupby("inference_id").cumcount()

        if not isinstance(y, pd.DataFrame):
            labels_columns = (
                self.labels_columns
                if self.labels_columns
                else list(str(int) for int in range(1 if y.ndim == 1 else y.shape[-1]))
            )

            y = pd.DataFrame(
                list(map(np.ravel, y)),
                columns=[str(sub) for sub in labels_columns],
            )
        y_df = y.stack().reset_index()
        y_df.columns = ["inference_id", "label_name", "label_value"]
        y_df["label_type"] = self.label_type
        y_df["inference_id"] = y_df["inference_id"].apply(lambda x: infid_to_uuid[x])
        y_df["batch.index"] = y_df.groupby("inference_id").cumcount()
        inference_data = pd.concat([X, y], axis=1)
        if self.send_data_metrics:
            if len(inference_data) >= data_summary_min_rows:
                self.df_statistics = self._calc_descriptive_statistics(inference_data)
                for name, metrics in self.df_statistics.to_dict().items():
                    metadata = {**self.static_metadata, "name": name}
                    metrics["types"] = FEATURE_TYPE.get(metrics["types"])
                    self.record_metrics(
                        metrics=metrics,
                        metadata=metadata,
                        data_metric=True,
                    )
                self._log("data_metric sent successfully")

            else:
                warnings.warn(
                    f"send_data_metrics occurs only when there are at least {data_summary_min_rows} rows. To change this constraint please update the variable 'data_summary_min_rows' in the fuction call"
                )
        if not self.send_inference_data:
            warnings.warn(
                "send_inference_data parameter is False, please turn it to True to send the inference_data"
            )
            return
        inference_data.reset_index(level=0, inplace=True)

        events = self.prepare_events(
            X_df,
            y_df,
            metadata=self.static_metadata,
            inference_id_to_inference_metadata=inference_id_to_inference_metadata,
            timestamp=timestamp,
        )
        try:
            self._record_events(events, EventName)
        except Exception as e:
            self._log(str(e))

        self._log("inference data sent successfully")

    def record_metrics(
        self,
        metrics: Dict[str, float],
        timestamp: int = None,
        metadata: Optional[Dict[str, Any]] = None,
        data_metric: bool = False,
        feature_name: Optional[str] = None,
    ):
        """This method send metrics to the table "Metric" in New Relic NRDB"""
        metric_type = "data_metric" if data_metric else "model_metric"
        metadata = metadata if metadata else {**self.static_metadata}

        metadata.update(
            {
                "metricType": metric_type,
            }
        )
        if feature_name is not None:
            metadata.update({"feature_name": feature_name})
        if self.model_version is not None:
            metadata.update({"model_version": self.model_version})
        metrics_batch: List[GaugeMetric] = []
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)):
                self._log(
                    f"Sending failed for metric {metric}: value instance type must be int or float and not {type(value)}"
                )
                continue
            metrics_batch.append(
                GaugeMetric(metric, value, metadata, end_time_ms=timestamp)
                if timestamp
                else GaugeMetric(metric, value, metadata)
            )
        try:
            self._record_metrics(metrics_batch)
        except Exception as e:
            self._log(str(e))

    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs):
        """This method call the model 'predict' method and also call 'record_inference_data' method to send  inference data to the table "InferenceData" in New Relic NRDB"""
        y_pred = self.model.predict(X)
        self.record_inference_data(X, y_pred, **kwargs)
        return y_pred

    def predict_log_proba(self, X: Union[pd.DataFrame, np.ndarray], **kwargs):
        """This method call the model 'predict_log_proba' method and also call 'record_inference_data' method to send  inference data to the table "InferenceData" in New Relic NRDB"""
        y_pred = self.model.predict_log_proba(X)
        self.record_inference_data(X, y_pred, **kwargs)
        return y_pred

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray], **kwargs):
        """This method call the model 'predict_log_proba' method and also call 'record_inference_data' method to send metrics to the table "InferenceData" in New Relic NRDB"""
        y_pred = self.model.predict_proba(X)
        self.record_inference_data(X, y_pred, **kwargs)
        return y_pred

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y):
        """This method call the model 'fit' method"""
        return self.model.fit(X, y)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y, **kwargs):
        """This method call the model 'fit_transform' method"""
        return self.model.fit_transform(X, y)

    def fit_predict(self, X: Union[pd.DataFrame, np.ndarray], y, **kwargs):
        """This method call the model 'fit_predict' method and also call 'record_inference_data' method to send inference data to the table "InferenceData" in New Relic NRDB"""
        y_pred = self.model.fit_predict(X, y)
        self.record_inference_data(X, y_pred, **kwargs)
        return y_pred


def wrap_model(
    model_name: str,
    model_version: str,
    insert_key: Optional[str] = None,
    metadata: Dict[str, Any] = {},
    staging: bool = False,
    model=None,
    send_inference_data=True,
    send_data_metrics=False,
    features_columns: Optional[List[str]] = None,
    labels_columns: Optional[List[str]] = None,
    label_type: Optional[str] = None,
    event_client_host: Optional[str] = None,
    metric_client_host: Optional[str] = None,
) -> MLPerformanceMonitoring:
    """This is a wrapper function that extends the model/pipeline methods with the functionality of sending the inference data to the table "InferenceData" in New Relic NRDB"""
    return MLPerformanceMonitoring(
        model_name,
        model_version,
        insert_key,
        metadata,
        staging,
        model,
        send_inference_data,
        send_data_metrics,
        features_columns,
        labels_columns,
        label_type,
        event_client_host,
        metric_client_host,
    )
