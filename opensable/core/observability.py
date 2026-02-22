"""
Observability - Distributed Tracing and Log Aggregation.

Features:
- OpenTelemetry integration
- Distributed tracing across services
- Span creation and management
- ELK/Loki log aggregation
- Structured logging
- Log shipping and querying
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import traceback


class SpanKind(Enum):
    """OpenTelemetry span kinds."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class LogLevel(Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class Span:
    """Distributed tracing span."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: SpanKind
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span."""
        self.events.append({"name": name, "timestamp": time.time(), "attributes": attributes or {}})

    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value

    def set_error(self, error: Exception):
        """Mark span as error."""
        self.status = "error"
        self.error = str(error)
        self.attributes["error.type"] = type(error).__name__
        self.attributes["error.message"] = str(error)
        self.attributes["error.stacktrace"] = traceback.format_exc()

    def end(self):
        """End the span."""
        if not self.end_time:
            self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class Trace:
    """Complete distributed trace."""

    trace_id: str
    spans: List[Span] = field(default_factory=list)

    def add_span(self, span: Span):
        """Add span to trace."""
        self.spans.append(span)

    def get_root_span(self) -> Optional[Span]:
        """Get root span."""
        for span in self.spans:
            if not span.parent_span_id:
                return span
        return None

    def get_duration_ms(self) -> float:
        """Get total trace duration."""
        if not self.spans:
            return 0.0

        start = min(span.start_time for span in self.spans)
        end = max(span.end_time or span.start_time for span in self.spans)
        return (end - start) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "duration_ms": self.get_duration_ms(),
            "spans": [span.to_dict() for span in self.spans],
        }


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "logger_name": self.logger_name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "attributes": self.attributes,
            "exception": self.exception,
        }


class DistributedTracer:
    """
    Distributed tracing implementation.

    Features:
    - Trace and span creation
    - Context propagation
    - Automatic instrumentation
    - Export to backends (Jaeger, Zipkin)
    """

    def __init__(self, service_name: str = "opensable"):
        """
        Initialize distributed tracer.

        Args:
            service_name: Name of the service
        """
        self.service_name = service_name
        self.traces: Dict[str, Trace] = {}
        self.active_spans: Dict[str, Span] = {}

    def create_trace(self) -> str:
        """Create a new trace."""
        import secrets

        trace_id = secrets.token_hex(16)
        self.traces[trace_id] = Trace(trace_id=trace_id)
        return trace_id

    def start_span(
        self,
        name: str,
        trace_id: str,
        parent_span_id: Optional[str] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new span."""
        import secrets

        span_id = secrets.token_hex(8)

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time=time.time(),
            attributes=attributes or {},
        )

        # Add service name
        span.set_attribute("service.name", self.service_name)

        # Add to trace
        if trace_id in self.traces:
            self.traces[trace_id].add_span(span)

        # Track as active
        self.active_spans[span_id] = span

        return span

    def end_span(self, span_id: str):
        """End a span."""
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.end()
            del self.active_spans[span_id]

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID."""
        return self.traces.get(trace_id)

    def export_traces(self, backend: str = "jaeger") -> List[Dict[str, Any]]:
        """
        Export traces to backend.

        Args:
            backend: Export backend (jaeger, zipkin, etc.)

        Returns:
            List of trace data
        """
        traces_data = []

        for trace in self.traces.values():
            if backend == "jaeger":
                traces_data.append(self._to_jaeger_format(trace))
            elif backend == "zipkin":
                traces_data.append(self._to_zipkin_format(trace))
            else:
                traces_data.append(trace.to_dict())

        return traces_data

    def _to_jaeger_format(self, trace: Trace) -> Dict[str, Any]:
        """Convert trace to Jaeger format."""
        return {
            "traceID": trace.trace_id,
            "spans": [
                {
                    "traceID": span.trace_id,
                    "spanID": span.span_id,
                    "parentSpanID": span.parent_span_id,
                    "operationName": span.name,
                    "startTime": int(span.start_time * 1_000_000),  # microseconds
                    "duration": int(span.duration_ms * 1000),  # microseconds
                    "tags": [
                        {"key": k, "type": "string", "value": str(v)}
                        for k, v in span.attributes.items()
                    ],
                    "logs": [
                        {
                            "timestamp": int(event["timestamp"] * 1_000_000),
                            "fields": [
                                {"key": k, "type": "string", "value": str(v)}
                                for k, v in event.get("attributes", {}).items()
                            ],
                        }
                        for event in span.events
                    ],
                }
                for span in trace.spans
            ],
        }

    def _to_zipkin_format(self, trace: Trace) -> List[Dict[str, Any]]:
        """Convert trace to Zipkin format."""
        return [
            {
                "traceId": span.trace_id,
                "id": span.span_id,
                "parentId": span.parent_span_id,
                "name": span.name,
                "timestamp": int(span.start_time * 1_000_000),  # microseconds
                "duration": int(span.duration_ms * 1000),  # microseconds
                "kind": span.kind.value.upper(),
                "tags": span.attributes,
                "annotations": [
                    {"timestamp": int(event["timestamp"] * 1_000_000), "value": event["name"]}
                    for event in span.events
                ],
            }
            for span in trace.spans
        ]


class LogAggregator:
    """
    Log aggregation and shipping.

    Features:
    - Structured logging
    - Log buffering and batching
    - Export to ELK stack or Loki
    - Log querying and filtering
    """

    def __init__(self, storage_dir: Optional[str] = None, buffer_size: int = 1000):
        """
        Initialize log aggregator.

        Args:
            storage_dir: Directory for log storage
            buffer_size: Buffer size before flushing
        """
        self.storage_dir = Path(storage_dir) if storage_dir else Path.home() / ".opensable" / "logs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.buffer_size = buffer_size
        self.log_buffer: List[LogEntry] = []

        # Setup Python logging integration
        self._setup_logging()

    def _setup_logging(self):
        """Setup Python logging integration."""
        handler = StructuredLogHandler(self)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)

    def log(
        self,
        level: LogLevel,
        message: str,
        logger_name: str = "opensable",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
    ):
        """Add log entry."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            logger_name=logger_name,
            trace_id=trace_id,
            span_id=span_id,
            attributes=attributes or {},
            exception=traceback.format_exc() if exception else None,
        )

        self.log_buffer.append(entry)

        # Flush if buffer full
        if len(self.log_buffer) >= self.buffer_size:
            asyncio.create_task(self.flush())

    async def flush(self):
        """Flush log buffer to storage."""
        if not self.log_buffer:
            return

        # Write to daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = self.storage_dir / f"{date_str}.jsonl"

        with open(log_file, "a") as f:
            for entry in self.log_buffer:
                f.write(json.dumps(entry.to_dict()) + "\n")

        self.log_buffer.clear()

    async def query(
        self,
        level: Optional[LogLevel] = None,
        logger_name: Optional[str] = None,
        trace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        search: Optional[str] = None,
        limit: int = 100,
    ) -> List[LogEntry]:
        """
        Query logs with filters.

        Args:
            level: Filter by log level
            logger_name: Filter by logger name
            trace_id: Filter by trace ID
            start_time: Filter by start time
            end_time: Filter by end time
            search: Text search in message
            limit: Maximum results

        Returns:
            List of matching log entries
        """
        results = []

        # Search in log files
        for log_file in sorted(self.storage_dir.glob("*.jsonl"), reverse=True):
            if len(results) >= limit:
                break

            with open(log_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        entry = self._dict_to_log_entry(data)

                        # Apply filters
                        if level and entry.level != level:
                            continue
                        if logger_name and entry.logger_name != logger_name:
                            continue
                        if trace_id and entry.trace_id != trace_id:
                            continue
                        if start_time and entry.timestamp < start_time:
                            continue
                        if end_time and entry.timestamp > end_time:
                            continue
                        if search and search.lower() not in entry.message.lower():
                            continue

                        results.append(entry)

                        if len(results) >= limit:
                            break
                    except Exception:
                        continue

        return results

    def _dict_to_log_entry(self, data: Dict[str, Any]) -> LogEntry:
        """Convert dictionary to LogEntry."""
        return LogEntry(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=LogLevel(data["level"]),
            message=data["message"],
            logger_name=data["logger_name"],
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            attributes=data.get("attributes", {}),
            exception=data.get("exception"),
        )

    def export_to_elk(self) -> List[Dict[str, Any]]:
        """Export logs in Elasticsearch format."""
        logs = []

        for log_file in self.storage_dir.glob("*.jsonl"):
            with open(log_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # ELK format
                        logs.append(
                            {
                                "@timestamp": data["timestamp"],
                                "level": data["level"],
                                "message": data["message"],
                                "logger": data["logger_name"],
                                "trace": {
                                    "id": data.get("trace_id"),
                                    "span_id": data.get("span_id"),
                                },
                                "attributes": data.get("attributes", {}),
                                "exception": data.get("exception"),
                            }
                        )
                    except Exception:
                        continue

        return logs

    def export_to_loki(self) -> Dict[str, Any]:
        """Export logs in Loki format."""
        streams = {}

        for log_file in self.storage_dir.glob("*.jsonl"):
            with open(log_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)

                        # Create label set
                        labels = {"level": data["level"], "logger": data["logger_name"]}
                        label_str = json.dumps(labels, sort_keys=True)

                        if label_str not in streams:
                            streams[label_str] = {"stream": labels, "values": []}

                        # Add log line
                        timestamp_ns = int(
                            datetime.fromisoformat(data["timestamp"]).timestamp() * 1_000_000_000
                        )
                        streams[label_str]["values"].append([str(timestamp_ns), data["message"]])
                    except Exception:
                        continue

        return {"streams": list(streams.values())}


class StructuredLogHandler(logging.Handler):
    """Python logging handler that integrates with LogAggregator."""

    def __init__(self, aggregator: LogAggregator):
        """Initialize handler."""
        super().__init__()
        self.aggregator = aggregator

    def emit(self, record: logging.LogRecord):
        """Emit log record."""
        level_map = {
            logging.DEBUG: LogLevel.DEBUG,
            logging.INFO: LogLevel.INFO,
            logging.WARNING: LogLevel.WARNING,
            logging.ERROR: LogLevel.ERROR,
            logging.CRITICAL: LogLevel.CRITICAL,
        }

        level = level_map.get(record.levelno, LogLevel.INFO)

        self.aggregator.log(
            level=level,
            message=record.getMessage(),
            logger_name=record.name,
            attributes={
                "filename": record.filename,
                "lineno": record.lineno,
                "funcName": record.funcName,
            },
            exception=record.exc_info[1] if record.exc_info else None,
        )


# Example usage
async def main():
    """Example observability features."""

    print("=" * 50)
    print("Observability Examples")
    print("=" * 50)

    # Distributed Tracing
    print("\n1. Distributed Tracing")
    tracer = DistributedTracer(service_name="opensable-api")

    # Create trace
    trace_id = tracer.create_trace()
    print(f"  Created trace: {trace_id}")

    # Create spans
    root_span = tracer.start_span(
        "handle_request",
        trace_id=trace_id,
        kind=SpanKind.SERVER,
        attributes={"http.method": "POST", "http.url": "/api/agents"},
    )

    # Child span
    db_span = tracer.start_span(
        "database_query",
        trace_id=trace_id,
        parent_span_id=root_span.span_id,
        kind=SpanKind.CLIENT,
        attributes={"db.system": "postgresql", "db.statement": "SELECT * FROM agents"},
    )

    # Simulate work
    await asyncio.sleep(0.1)
    db_span.add_event("query_executed", {"rows": 5})
    tracer.end_span(db_span.span_id)

    await asyncio.sleep(0.05)
    tracer.end_span(root_span.span_id)

    # Get trace
    trace = tracer.get_trace(trace_id)
    if trace:
        print(f"  Trace duration: {trace.get_duration_ms():.2f}ms")
        print(f"  Spans: {len(trace.spans)}")
        for span in trace.spans:
            print(f"    - {span.name}: {span.duration_ms:.2f}ms")

    # Log Aggregation
    print("\n2. Log Aggregation")
    aggregator = LogAggregator()

    # Add logs
    aggregator.log(
        LogLevel.INFO,
        "Application started",
        logger_name="opensable.main",
        attributes={"version": "1.0.0"},
    )

    aggregator.log(
        LogLevel.DEBUG,
        "Processing request",
        logger_name="opensable.api",
        trace_id=trace_id,
        span_id=root_span.span_id,
        attributes={"user_id": "user123"},
    )

    aggregator.log(
        LogLevel.WARNING,
        "Slow query detected",
        logger_name="opensable.db",
        trace_id=trace_id,
        span_id=db_span.span_id,
        attributes={"duration_ms": 150},
    )

    print(f"  Logged {len(aggregator.log_buffer)} entries")

    # Flush to storage
    await aggregator.flush()
    print("  Flushed logs to storage")

    # Query logs
    logs = await aggregator.query(trace_id=trace_id)
    print(f"\n  Query results: {len(logs)} logs for trace {trace_id[:8]}...")
    for log in logs:
        print(f"    - [{log.level.value}] {log.message}")

    # Python logging integration
    print("\n3. Python Logging Integration")
    logger = logging.getLogger("opensable.test")

    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")

    print(f"  Logged {len(aggregator.log_buffer)} entries via Python logging")

    # Export formats
    print("\n4. Export Formats")

    # Jaeger
    jaeger_traces = tracer.export_traces(backend="jaeger")
    print(f"  Jaeger format: {len(jaeger_traces)} traces")

    # ELK
    elk_logs = aggregator.export_to_elk()
    print(f"  ELK format: {len(elk_logs)} log entries")

    # Loki
    loki_logs = aggregator.export_to_loki()
    print(f"  Loki format: {len(loki_logs.get('streams', []))} streams")

    print("\n✅ Observability examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
