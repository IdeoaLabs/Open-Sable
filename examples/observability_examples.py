"""
Observability Examples - Distributed tracing and log aggregation.

Demonstrates OpenTelemetry tracing with Jaeger/Zipkin and structured logging with ELK/Loki.
"""

import asyncio
import time
from core.observability import (
    TracingProvider, LogAggregator,
    Span, SpanKind, SpanStatus, LogLevel
)


async def main():
    """Run observability examples."""
    
    print("=" * 60)
    print("Observability Examples")
    print("=" * 60)
    
    # Example 1: Distributed tracing setup
    print("\n1. Distributed Tracing Setup")
    print("-" * 40)
    
    tracer = TracingProvider(
        service_name="opensable-demo",
        exporter="jaeger",
        endpoint="http://localhost:14268/api/traces"
    )
    
    print(f"Initialized tracer for: {tracer.service_name}")
    print(f"Exporter: {tracer.exporter}")
    print(f"Endpoint: {tracer.endpoint}")
    
    # Example 2: Basic span
    print("\n2. Basic Span Creation")
    print("-" * 40)
    
    span = tracer.start_span(
        name="process_request",
        kind=SpanKind.SERVER,
        attributes={"http.method": "POST", "http.url": "/api/agents"}
    )
    
    # Simulate work
    await asyncio.sleep(0.1)
    
    span.set_status(SpanStatus.OK, "Request processed successfully")
    tracer.end_span(span)
    
    print(f"Created span: {span.name}")
    print(f"Trace ID: {span.trace_id}")
    print(f"Span ID: {span.span_id}")
    print(f"Duration: {span.duration_ms:.2f}ms")
    
    # Example 3: Nested spans (parent-child)
    print("\n3. Nested Spans (Parent-Child)")
    print("-" * 40)
    
    parent = tracer.start_span("handle_request", SpanKind.SERVER)
    
    # Child span 1
    auth_span = tracer.start_span(
        "authenticate_user",
        SpanKind.INTERNAL,
        parent=parent,
        attributes={"user.id": "user_123"}
    )
    await asyncio.sleep(0.05)
    auth_span.set_status(SpanStatus.OK)
    tracer.end_span(auth_span)
    
    # Child span 2
    db_span = tracer.start_span(
        "query_database",
        SpanKind.CLIENT,
        parent=parent,
        attributes={"db.system": "postgresql", "db.query": "SELECT * FROM agents"}
    )
    await asyncio.sleep(0.08)
    db_span.set_status(SpanStatus.OK)
    tracer.end_span(db_span)
    
    # Child span 3
    render_span = tracer.start_span(
        "render_response",
        SpanKind.INTERNAL,
        parent=parent
    )
    await asyncio.sleep(0.03)
    render_span.set_status(SpanStatus.OK)
    tracer.end_span(render_span)
    
    tracer.end_span(parent)
    
    print(f"Parent span: {parent.name}")
    print(f"  - Child 1: {auth_span.name} ({auth_span.duration_ms:.2f}ms)")
    print(f"  - Child 2: {db_span.name} ({db_span.duration_ms:.2f}ms)")
    print(f"  - Child 3: {render_span.name} ({render_span.duration_ms:.2f}ms)")
    print(f"Total duration: {parent.duration_ms:.2f}ms")
    
    # Example 4: Error tracking in spans
    print("\n4. Error Tracking")
    print("-" * 40)
    
    error_span = tracer.start_span("risky_operation", SpanKind.INTERNAL)
    
    try:
        # Simulate an error
        await asyncio.sleep(0.02)
        raise ValueError("Database connection timeout")
    except Exception as e:
        error_span.record_exception(e)
        error_span.set_status(SpanStatus.ERROR, str(e))
    finally:
        tracer.end_span(error_span)
    
    print(f"Error span: {error_span.name}")
    print(f"Status: {error_span.status}")
    print(f"Error message: {error_span.status_message}")
    
    # Example 5: Span events
    print("\n5. Span Events")
    print("-" * 40)
    
    event_span = tracer.start_span("workflow_execution", SpanKind.INTERNAL)
    
    # Add events
    event_span.add_event("workflow_started", {"workflow.id": "wf_123"})
    await asyncio.sleep(0.05)
    
    event_span.add_event("step_completed", {"step": "data_validation", "status": "success"})
    await asyncio.sleep(0.03)
    
    event_span.add_event("step_completed", {"step": "data_processing", "status": "success"})
    await asyncio.sleep(0.04)
    
    event_span.add_event("workflow_completed", {"duration_ms": 120})
    
    tracer.end_span(event_span)
    
    print(f"Span with events: {event_span.name}")
    print(f"Events recorded: {len(event_span.events)}")
    for event in event_span.events:
        print(f"  - {event['name']}: {event.get('attributes', {})}")
    
    # Example 6: Trace context propagation
    print("\n6. Trace Context Propagation")
    print("-" * 40)
    
    # Service A starts a trace
    service_a_span = tracer.start_span("service_a_call", SpanKind.CLIENT)
    
    # Extract context for propagation
    context = tracer.extract_context(service_a_span)
    print(f"Propagation context:")
    print(f"  traceparent: {context.get('traceparent', 'N/A')[:50]}...")
    
    # Service B receives the context and continues the trace
    service_b_span = tracer.start_span(
        "service_b_process",
        SpanKind.SERVER,
        context=context
    )
    
    await asyncio.sleep(0.06)
    
    tracer.end_span(service_b_span)
    tracer.end_span(service_a_span)
    
    print(f"Service A span ID: {service_a_span.span_id}")
    print(f"Service B span ID: {service_b_span.span_id}")
    print(f"Same trace: {service_a_span.trace_id == service_b_span.trace_id}")
    
    # Example 7: Log aggregation setup
    print("\n7. Log Aggregation Setup")
    print("-" * 40)
    
    logger = LogAggregator(
        service_name="opensable-demo",
        backend="elasticsearch",
        endpoint="http://localhost:9200",
        index_name="opensable-logs"
    )
    
    print(f"Initialized log aggregator for: {logger.service_name}")
    print(f"Backend: {logger.backend}")
    print(f"Index: {logger.index_name}")
    
    # Example 8: Structured logging
    print("\n8. Structured Logging")
    print("-" * 40)
    
    # Different log levels
    await logger.log(
        level=LogLevel.INFO,
        message="Application started",
        context={
            "version": "1.0.0",
            "environment": "production",
            "region": "us-west-2"
        }
    )
    
    await logger.log(
        level=LogLevel.DEBUG,
        message="Processing request",
        context={
            "request_id": "req_abc123",
            "user_id": "user_456",
            "endpoint": "/api/agents",
            "method": "POST"
        }
    )
    
    await logger.log(
        level=LogLevel.WARNING,
        message="High memory usage detected",
        context={
            "memory_used_mb": 850,
            "memory_limit_mb": 1024,
            "usage_percent": 83
        }
    )
    
    await logger.log(
        level=LogLevel.ERROR,
        message="Database query failed",
        context={
            "query": "SELECT * FROM workflows WHERE id = ?",
            "error": "Connection timeout",
            "retry_count": 3
        }
    )
    
    print(f"Logged {len(logger.logs)} events")
    
    # Example 9: Log queries
    print("\n9. Log Queries")
    print("-" * 40)
    
    # Query by level
    errors = await logger.query(level=LogLevel.ERROR)
    print(f"Error logs: {len(errors)}")
    
    # Query by time range
    from datetime import datetime, timedelta
    
    recent = await logger.query(
        start_time=datetime.now() - timedelta(minutes=5),
        end_time=datetime.now()
    )
    print(f"Recent logs (last 5 min): {len(recent)}")
    
    # Query by context filter
    user_logs = await logger.query(
        filters={"user_id": "user_456"}
    )
    print(f"Logs for user_456: {len(user_logs)}")
    
    # Example 10: Correlation with traces
    print("\n10. Trace-Log Correlation")
    print("-" * 40)
    
    correlated_span = tracer.start_span("correlated_operation", SpanKind.INTERNAL)
    
    # Log with trace context
    await logger.log(
        level=LogLevel.INFO,
        message="Operation in progress",
        context={
            "trace_id": correlated_span.trace_id,
            "span_id": correlated_span.span_id,
            "operation": "data_sync"
        }
    )
    
    await asyncio.sleep(0.05)
    
    tracer.end_span(correlated_span)
    
    print(f"Trace ID: {correlated_span.trace_id}")
    print("Logs can be correlated using trace_id")
    
    # Example 11: Export to different backends
    print("\n11. Multi-Backend Export")
    print("-" * 40)
    
    # Jaeger exporter
    jaeger_tracer = TracingProvider(
        service_name="opensable",
        exporter="jaeger",
        endpoint="http://localhost:14268/api/traces"
    )
    
    # Zipkin exporter
    zipkin_tracer = TracingProvider(
        service_name="opensable",
        exporter="zipkin",
        endpoint="http://localhost:9411/api/v2/spans"
    )
    
    print("Configured exporters:")
    print(f"  - Jaeger: {jaeger_tracer.endpoint}")
    print(f"  - Zipkin: {zipkin_tracer.endpoint}")
    
    # ELK stack
    elk_logger = LogAggregator(
        service_name="opensable",
        backend="elasticsearch",
        endpoint="http://localhost:9200"
    )
    
    # Loki
    loki_logger = LogAggregator(
        service_name="opensable",
        backend="loki",
        endpoint="http://localhost:3100"
    )
    
    print("Configured log backends:")
    print(f"  - Elasticsearch: {elk_logger.endpoint}")
    print(f"  - Loki: {loki_logger.endpoint}")
    
    # Example 12: Performance monitoring
    print("\n12. Performance Monitoring")
    print("-" * 40)
    
    # Trace multiple operations
    operations = ["query_db", "call_api", "process_data", "render_response"]
    
    for op in operations:
        span = tracer.start_span(op, SpanKind.INTERNAL)
        await asyncio.sleep(0.02 + (hash(op) % 50) / 1000)  # Vary timing
        tracer.end_span(span)
    
    # Calculate stats
    total_ops = len(operations)
    avg_duration = sum(s.duration_ms for s in tracer.spans[-total_ops:]) / total_ops
    
    print(f"Operations traced: {total_ops}")
    print(f"Average duration: {avg_duration:.2f}ms")
    print(f"\nDetailed timings:")
    for span in tracer.spans[-total_ops:]:
        print(f"  {span.name}: {span.duration_ms:.2f}ms")
    
    # Log performance summary
    await logger.log(
        level=LogLevel.INFO,
        message="Performance summary",
        context={
            "total_operations": total_ops,
            "avg_duration_ms": avg_duration,
            "min_duration_ms": min(s.duration_ms for s in tracer.spans[-total_ops:]),
            "max_duration_ms": max(s.duration_ms for s in tracer.spans[-total_ops:])
        }
    )
    
    print("\n" + "=" * 60)
    print("✅ Observability examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
