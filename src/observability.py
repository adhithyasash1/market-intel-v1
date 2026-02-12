import logging
import json
from typing import Dict, Any


# ─── Structured Logging ───
class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for structured logging systems (ELK, Splunk, etc).
    """
    def format(self, record):
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "process_id": record.process,
        }
        # Merge extra properties if present
        if hasattr(record, "props") and isinstance(record.props, dict):  # type: ignore
            log_obj.update(record.props)  # type: ignore

        # Include exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def configure_logging(level=logging.INFO):
    """Configures the root logger to output JSON to stdout."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    # Remove existing handlers to prevent duplicate logs
    if root_logger.handlers:
        root_logger.handlers = []

    root_logger.addHandler(handler)
    root_logger.setLevel(level)


# ─── Prometheus Metrics ───
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server

    # Metrics Definitions
    BACKTEST_DURATION = Histogram(
        'backtest_duration_seconds',
        'Time spent running backtest simulation',
        buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0)
    )

    SNAPSHOT_FETCH_DURATION = Histogram(
        'snapshot_fetch_duration_seconds',
        'Time spent fetching TradingView snapshot',
        buckets=(1.0, 5.0, 10.0, 30.0)
    )

    SNAPSHOT_FETCH_ERRORS = Counter(
        'snapshot_fetch_errors_total',
        'Total number of failed snapshot fetches'
    )

    LAST_SNAPSHOT_TIMESTAMP = Gauge(
        'last_snapshot_timestamp_seconds',
        'Unix timestamp of the last successful snapshot'
    )

    FAILED_JOBS_TOTAL = Counter(
        'failed_jobs_total',
        'Total number of failed worker jobs',
        ['job_type']
    )

    def start_metrics_server(port=8000):
        """Starts a background thread to serve Prometheus metrics."""
        try:
            start_http_server(port)
            logging.info(f"Metrics server started on port {port}")
        except Exception as e:
            logging.error(f"Failed to start metrics server: {e}")

except ImportError:
    # Dummy implementations for environments without prometheus_client (e.g. CI)
    logging.getLogger(__name__).warning("prometheus_client not found. Metrics will be no-ops.")

    class DummyMetric:
        def observe(self, x): pass

        def inc(self, amount=1): pass

        def set(self, x): pass

        def time(self): return self

        def __enter__(self): return self

        def __exit__(self, exc_type, exc_val, exc_tb): pass

        def labels(self, **kwargs): return self

    BACKTEST_DURATION = DummyMetric()
    SNAPSHOT_FETCH_DURATION = DummyMetric()
    SNAPSHOT_FETCH_ERRORS = DummyMetric()
    LAST_SNAPSHOT_TIMESTAMP = DummyMetric()
    FAILED_JOBS_TOTAL = DummyMetric()

    def start_metrics_server(port=8000): pass
