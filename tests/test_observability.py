"""
Tests for Observability - Distributed Tracing and Log Aggregation.
"""

import pytest
import time
from opensable.core.observability import (
    DistributedTracer, LogAggregator, Span, Trace,
    SpanKind, LogLevel, LogEntry
)


class TestSpanKind:
    """Test SpanKind enum."""

    def test_members(self):
        assert SpanKind.INTERNAL.value == "internal"
        assert SpanKind.SERVER.value == "server"
        assert SpanKind.CLIENT.value == "client"
        assert SpanKind.PRODUCER.value == "producer"
        assert SpanKind.CONSUMER.value == "consumer"


class TestLogLevel:
    """Test LogLevel enum."""

    def test_members(self):
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestLogEntry:
    """Test LogEntry dataclass."""

    def test_create(self):
        from datetime import datetime
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="test msg",
            logger_name="test",
        )
        assert entry.level == LogLevel.INFO
        assert entry.message == "test msg"
        assert entry.logger_name == "test"

    def test_to_dict(self):
        from datetime import datetime
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="oops",
            logger_name="app",
            trace_id="t1",
        )
        d = entry.to_dict()
        assert d["level"] == "ERROR"
        assert d["message"] == "oops"
        assert d["trace_id"] == "t1"

    def test_defaults(self):
        from datetime import datetime
        entry = LogEntry(timestamp=datetime.now(), level=LogLevel.DEBUG, message="x", logger_name="l")
        assert entry.trace_id is None
        assert entry.span_id is None
        assert entry.attributes == {}
        assert entry.exception is None


class TestSpan:
    """Test Span dataclass."""

    def test_create(self):
        s = Span(
            trace_id="t1", span_id="s1", parent_span_id=None,
            name="op", kind=SpanKind.INTERNAL, start_time=time.time()
        )
        assert s.trace_id == "t1"
        assert s.name == "op"
        assert s.status == "ok"

    def test_add_event(self):
        s = Span(trace_id="t", span_id="s", parent_span_id=None,
                 name="op", kind=SpanKind.SERVER, start_time=time.time())
        s.add_event("checkpoint", {"key": "val"})
        assert len(s.events) == 1
        assert s.events[0]["name"] == "checkpoint"

    def test_set_attribute(self):
        s = Span(trace_id="t", span_id="s", parent_span_id=None,
                 name="op", kind=SpanKind.CLIENT, start_time=time.time())
        s.set_attribute("http.method", "GET")
        assert s.attributes["http.method"] == "GET"

    def test_set_error(self):
        s = Span(trace_id="t", span_id="s", parent_span_id=None,
                 name="op", kind=SpanKind.INTERNAL, start_time=time.time())
        s.set_error(ValueError("bad"))
        assert s.status == "error"
        assert s.error == "bad"

    def test_end_and_duration(self):
        s = Span(trace_id="t", span_id="s", parent_span_id=None,
                 name="op", kind=SpanKind.INTERNAL, start_time=time.time())
        time.sleep(0.01)
        s.end()
        assert s.end_time is not None
        assert s.duration_ms > 0

    def test_to_dict(self):
        s = Span(trace_id="t", span_id="s", parent_span_id=None,
                 name="op", kind=SpanKind.INTERNAL, start_time=1000.0)
        d = s.to_dict()
        assert d["trace_id"] == "t"
        assert d["kind"] == "internal"


class TestDistributedTracer:
    """Test distributed tracer."""

    @pytest.fixture
    def tracer(self):
        return DistributedTracer(service_name="test-svc")

    def test_init(self, tracer):
        assert tracer.service_name == "test-svc"

    def test_create_trace(self, tracer):
        tid = tracer.create_trace()
        assert isinstance(tid, str)
        assert tid in tracer.traces

    def test_start_span(self, tracer):
        tid = tracer.create_trace()
        span = tracer.start_span("operation", trace_id=tid)
        assert isinstance(span, Span)
        assert span.name == "operation"
        assert span.trace_id == tid

    def test_end_span(self, tracer):
        tid = tracer.create_trace()
        span = tracer.start_span("op", trace_id=tid)
        tracer.end_span(span.span_id)
        assert span.span_id not in tracer.active_spans
        assert span.end_time is not None

    def test_child_span(self, tracer):
        tid = tracer.create_trace()
        parent = tracer.start_span("parent", trace_id=tid)
        child = tracer.start_span("child", trace_id=tid, parent_span_id=parent.span_id)
        assert child.parent_span_id == parent.span_id

    def test_get_trace(self, tracer):
        tid = tracer.create_trace()
        tracer.start_span("op", trace_id=tid)
        trace = tracer.get_trace(tid)
        assert trace is not None
        assert len(trace.spans) == 1

    def test_span_kind(self, tracer):
        tid = tracer.create_trace()
        span = tracer.start_span("http", trace_id=tid, kind=SpanKind.SERVER)
        assert span.kind == SpanKind.SERVER

    def test_export_traces(self, tracer):
        tid = tracer.create_trace()
        tracer.start_span("op", trace_id=tid)
        exported = tracer.export_traces(backend="jaeger")
        assert len(exported) >= 1


class TestTrace:
    """Test Trace dataclass."""

    def test_create(self):
        t = Trace(trace_id="abc")
        assert t.trace_id == "abc"
        assert t.spans == []

    def test_add_span(self):
        t = Trace(trace_id="abc")
        s = Span(trace_id="abc", span_id="s1", parent_span_id=None,
                 name="op", kind=SpanKind.INTERNAL, start_time=time.time())
        t.add_span(s)
        assert len(t.spans) == 1

    def test_get_root_span(self):
        t = Trace(trace_id="abc")
        s = Span(trace_id="abc", span_id="s1", parent_span_id=None,
                 name="root", kind=SpanKind.INTERNAL, start_time=time.time())
        t.add_span(s)
        root = t.get_root_span()
        assert root is not None
        assert root.name == "root"


class TestLogAggregator:
    """Test log aggregator."""

    @pytest.fixture
    def aggregator(self, tmp_path):
        return LogAggregator(storage_dir=str(tmp_path / "logs"))

    def test_init(self, aggregator):
        assert aggregator.log_buffer == []
        assert aggregator.buffer_size == 1000

    def test_log_entry(self, aggregator):
        aggregator.log(LogLevel.INFO, "hello world", logger_name="test")
        assert len(aggregator.log_buffer) == 1
        assert aggregator.log_buffer[0].message == "hello world"

    def test_log_with_context(self, aggregator):
        aggregator.log(
            LogLevel.ERROR, "failed",
            logger_name="app",
            trace_id="t123",
            attributes={"key": "val"},
        )
        entry = aggregator.log_buffer[0]
        assert entry.trace_id == "t123"
        assert entry.attributes["key"] == "val"

    def test_multiple_levels(self, aggregator):
        aggregator.log(LogLevel.DEBUG, "d", logger_name="l")
        aggregator.log(LogLevel.WARNING, "w", logger_name="l")
        assert len(aggregator.log_buffer) == 2

    @pytest.mark.asyncio
    async def test_flush(self, aggregator):
        aggregator.log(LogLevel.INFO, "flush me", logger_name="t")
        await aggregator.flush()
        assert len(aggregator.log_buffer) == 0

    @pytest.mark.asyncio
    async def test_query(self, aggregator):
        aggregator.log(LogLevel.INFO, "findme", logger_name="t")
        await aggregator.flush()
        results = await aggregator.query(search="findme")
        assert len(results) >= 1
        assert results[0].message == "findme"
