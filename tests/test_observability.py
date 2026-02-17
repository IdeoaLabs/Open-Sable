"""
Tests for Observability - Distributed tracing and log aggregation.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from core.observability import (
    TracingProvider, LogAggregator,
    Span, SpanKind, SpanStatus, LogLevel
)


class TestTracingProvider:
    """Test distributed tracing"""
    
    @pytest.fixture
    def tracer(self):
        return TracingProvider(
            service_name="test-service",
            exporter="jaeger",
            endpoint="http://localhost:14268/api/traces"
        )
    
    def test_create_span(self, tracer):
        """Test span creation"""
        span = tracer.start_span("test_operation", SpanKind.INTERNAL)
        
        assert span.name == "test_operation"
        assert span.kind == SpanKind.INTERNAL
        assert span.trace_id is not None
        assert span.span_id is not None
    
    def test_span_with_attributes(self, tracer):
        """Test span with attributes"""
        span = tracer.start_span(
            "http_request",
            SpanKind.SERVER,
            attributes={"http.method": "GET", "http.status": 200}
        )
        
        assert span.attributes["http.method"] == "GET"
        assert span.attributes["http.status"] == 200
    
    def test_nested_spans(self, tracer):
        """Test parent-child span relationships"""
        parent = tracer.start_span("parent_op", SpanKind.INTERNAL)
        child = tracer.start_span("child_op", SpanKind.INTERNAL, parent=parent)
        
        assert child.parent_span_id == parent.span_id
        assert child.trace_id == parent.trace_id  # Same trace
    
    def test_span_status(self, tracer):
        """Test span status setting"""
        span = tracer.start_span("test_op", SpanKind.INTERNAL)
        
        span.set_status(SpanStatus.OK, "Operation successful")
        
        assert span.status == SpanStatus.OK
        assert span.status_message == "Operation successful"
    
    def test_span_error(self, tracer):
        """Test error recording in span"""
        span = tracer.start_span("failing_op", SpanKind.INTERNAL)
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            span.record_exception(e)
            span.set_status(SpanStatus.ERROR, str(e))
        
        assert span.status == SpanStatus.ERROR
        assert "Test error" in span.status_message
    
    def test_span_events(self, tracer):
        """Test adding events to spans"""
        span = tracer.start_span("event_test", SpanKind.INTERNAL)
        
        span.add_event("event1", {"key": "value1"})
        span.add_event("event2", {"key": "value2"})
        
        assert len(span.events) == 2
        assert span.events[0]["name"] == "event1"
    
    def test_context_propagation(self, tracer):
        """Test trace context extraction and propagation"""
        span = tracer.start_span("service_a", SpanKind.CLIENT)
        
        context = tracer.extract_context(span)
        
        assert "traceparent" in context
        assert span.trace_id in context["traceparent"]
    
    def test_span_duration(self, tracer):
        """Test span duration tracking"""
        import time
        
        span = tracer.start_span("timed_op", SpanKind.INTERNAL)
        time.sleep(0.1)  # 100ms
        tracer.end_span(span)
        
        assert span.duration_ms >= 100


class TestLogAggregator:
    """Test log aggregation"""
    
    @pytest.fixture
    def logger(self):
        return LogAggregator(
            service_name="test-service",
            backend="elasticsearch",
            endpoint="http://localhost:9200"
        )
    
    @pytest.mark.asyncio
    async def test_log_message(self, logger):
        """Test basic logging"""
        await logger.log(
            level=LogLevel.INFO,
            message="Test log message",
            context={"key": "value"}
        )
        
        assert len(logger.logs) == 1
        assert logger.logs[0].message == "Test log message"
        assert logger.logs[0].level == LogLevel.INFO
    
    @pytest.mark.asyncio
    async def test_log_levels(self, logger):
        """Test different log levels"""
        await logger.log(LogLevel.DEBUG, "Debug message")
        await logger.log(LogLevel.INFO, "Info message")
        await logger.log(LogLevel.WARNING, "Warning message")
        await logger.log(LogLevel.ERROR, "Error message")
        
        assert len(logger.logs) == 4
    
    @pytest.mark.asyncio
    async def test_structured_logging(self, logger):
        """Test structured log context"""
        await logger.log(
            LogLevel.INFO,
            "User action",
            context={
                "user_id": "123",
                "action": "login",
                "ip": "192.168.1.1"
            }
        )
        
        log = logger.logs[0]
        assert log.context["user_id"] == "123"
        assert log.context["action"] == "login"
    
    @pytest.mark.asyncio
    async def test_query_by_level(self, logger):
        """Test querying logs by level"""
        await logger.log(LogLevel.INFO, "Info 1")
        await logger.log(LogLevel.ERROR, "Error 1")
        await logger.log(LogLevel.INFO, "Info 2")
        
        errors = await logger.query(level=LogLevel.ERROR)
        
        assert len(errors) == 1
        assert errors[0].message == "Error 1"
    
    @pytest.mark.asyncio
    async def test_query_by_time(self, logger):
        """Test querying logs by time range"""
        now = datetime.now()
        
        await logger.log(LogLevel.INFO, "Recent log")
        
        recent = await logger.query(
            start_time=now - timedelta(minutes=1),
            end_time=now + timedelta(minutes=1)
        )
        
        assert len(recent) >= 1
    
    @pytest.mark.asyncio
    async def test_query_with_filters(self, logger):
        """Test querying with context filters"""
        await logger.log(
            LogLevel.INFO,
            "User action",
            context={"user_id": "123", "action": "login"}
        )
        await logger.log(
            LogLevel.INFO,
            "Other action",
            context={"user_id": "456", "action": "logout"}
        )
        
        user_logs = await logger.query(filters={"user_id": "123"})
        
        assert len(user_logs) == 1
        assert user_logs[0].context["user_id"] == "123"
    
    @pytest.mark.asyncio
    async def test_trace_correlation(self, logger):
        """Test trace ID in logs for correlation"""
        await logger.log(
            LogLevel.INFO,
            "Traced operation",
            context={
                "trace_id": "abc123",
                "span_id": "span456"
            }
        )
        
        log = logger.logs[0]
        assert log.context["trace_id"] == "abc123"


class TestObservabilityIntegration:
    """Test tracing + logging integration"""
    
    @pytest.fixture
    def tracer(self):
        return TracingProvider(service_name="test")
    
    @pytest.fixture
    def logger(self):
        return LogAggregator(service_name="test")
    
    @pytest.mark.asyncio
    async def test_correlated_trace_and_logs(self, tracer, logger):
        """Test correlating traces with logs"""
        span = tracer.start_span("operation", SpanKind.INTERNAL)
        
        # Log with trace context
        await logger.log(
            LogLevel.INFO,
            "Operation in progress",
            context={
                "trace_id": span.trace_id,
                "span_id": span.span_id
            }
        )
        
        tracer.end_span(span)
        
        # Verify correlation
        log = logger.logs[0]
        assert log.context["trace_id"] == span.trace_id
        assert log.context["span_id"] == span.span_id
    
    def test_multiple_exporters(self):
        """Test different trace exporters"""
        jaeger = TracingProvider(service_name="test", exporter="jaeger")
        zipkin = TracingProvider(service_name="test", exporter="zipkin")
        
        assert jaeger.exporter == "jaeger"
        assert zipkin.exporter == "zipkin"
    
    def test_multiple_log_backends(self):
        """Test different log backends"""
        elk = LogAggregator(service_name="test", backend="elasticsearch")
        loki = LogAggregator(service_name="test", backend="loki")
        
        assert elk.backend == "elasticsearch"
        assert loki.backend == "loki"
