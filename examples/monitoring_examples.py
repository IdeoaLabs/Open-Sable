"""
Monitoring Examples - Prometheus metrics and health checks.

Demonstrates metrics collection, health checks, and Grafana dashboard generation.
"""

import asyncio
from core.monitoring import MetricsCollector, HealthChecker


async def main():
    """Run monitoring examples."""
    
    print("=" * 60)
    print("Monitoring Examples")
    print("=" * 60)
    
    collector = MetricsCollector()
    health_checker = HealthChecker()
    
    # Example 1: Counter metrics
    print("\n1. Counter Metrics")
    print("-" * 40)
    
    # Increment counters
    for i in range(10):
        collector.increment_counter("requests_total", labels={"endpoint": "/api/chat"})
        collector.increment_counter("requests_total", labels={"endpoint": "/api/agents"})
    
    collector.increment_counter("errors_total", labels={"type": "timeout"}, value=3)
    collector.increment_counter("errors_total", labels={"type": "validation"}, value=5)
    
    print("Incremented request and error counters")
    
    # Example 2: Gauge metrics
    print("\n2. Gauge Metrics")
    print("-" * 40)
    
    # Set gauge values
    collector.set_gauge("active_sessions", 42)
    collector.set_gauge("queue_size", 15)
    collector.set_gauge("memory_usage_bytes", 1024 * 1024 * 256)  # 256 MB
    
    print("Set gauge values:")
    print(f"  Active sessions: 42")
    print(f"  Queue size: 15")
    print(f"  Memory usage: 256 MB")
    
    # Increase/decrease
    collector.increment_gauge("active_sessions", 5)
    collector.decrement_gauge("queue_size", 3)
    
    print("Updated gauges: +5 sessions, -3 queue items")
    
    # Example 3: Histogram metrics
    print("\n3. Histogram Metrics")
    print("-" * 40)
    
    # Record response times
    response_times = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.8, 1.2, 2.0]
    
    for rt in response_times:
        collector.observe_histogram(
            "request_duration_seconds",
            rt,
            labels={"method": "GET"}
        )
    
    print(f"Recorded {len(response_times)} response times")
    print(f"Range: {min(response_times)}s - {max(response_times)}s")
    
    # Example 4: Summary metrics
    print("\n4. Summary Metrics")
    print("-" * 40)
    
    # Record values for summary
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    for v in values:
        collector.observe_summary(
            "processing_time_ms",
            v,
            labels={"worker": "worker1"}
        )
    
    print(f"Recorded {len(values)} processing times")
    
    # Example 5: Health checks
    print("\n5. Health Checks")
    print("-" * 40)
    
    # Register health checks
    async def check_database():
        # Simulate database check
        await asyncio.sleep(0.1)
        return True, "Database connected"
    
    async def check_redis():
        # Simulate Redis check
        await asyncio.sleep(0.05)
        return True, "Redis available"
    
    async def check_api():
        # Simulate API check
        await asyncio.sleep(0.08)
        return False, "API endpoint timeout"
    
    health_checker.register_check("database", check_database)
    health_checker.register_check("redis", check_redis)
    health_checker.register_check("api", check_api)
    
    print("Registered 3 health checks")
    
    # Run health checks
    results = await health_checker.run_all_checks()
    
    print(f"\nHealth check results:")
    for name, result in results.items():
        status = "✅" if result["healthy"] else "❌"
        print(f"  {status} {name}: {result['status']}")
    
    # Example 6: Liveness and readiness
    print("\n6. Liveness and Readiness Probes")
    print("-" * 40)
    
    is_alive = await health_checker.liveness()
    is_ready = await health_checker.readiness()
    
    print(f"Liveness: {'✅' if is_alive else '❌'}")
    print(f"Readiness: {'✅' if is_ready else '❌'}")
    
    # Example 7: Custom metrics
    print("\n7. Custom Metrics")
    print("-" * 40)
    
    # Track custom business metrics
    collector.increment_counter(
        "messages_processed_total",
        labels={"interface": "telegram", "status": "success"}
    )
    
    collector.increment_counter(
        "messages_processed_total",
        labels={"interface": "discord", "status": "success"}
    )
    
    collector.set_gauge("active_users", 127)
    collector.set_gauge("pending_tasks", 8)
    
    print("Recorded custom business metrics")
    
    # Example 8: Performance monitoring
    print("\n8. Performance Monitoring")
    print("-" * 40)
    
    # Simulate request processing
    import time
    
    start = time.time()
    await asyncio.sleep(0.2)  # Simulate work
    duration = time.time() - start
    
    collector.observe_histogram(
        "agent_execution_seconds",
        duration,
        labels={"agent_type": "researcher"}
    )
    
    print(f"Recorded agent execution time: {duration:.3f}s")
    
    # Example 9: Export metrics
    print("\n9. Export Metrics")
    print("-" * 40)
    
    metrics_output = collector.export_metrics()
    
    print(f"Exported metrics (first 500 chars):")
    print(metrics_output[:500])
    print("...")
    
    # Example 10: Grafana dashboard
    print("\n10. Grafana Dashboard Generation")
    print("-" * 40)
    
    dashboard = collector.generate_grafana_dashboard()
    
    print(f"Generated Grafana dashboard:")
    print(f"  Title: {dashboard['dashboard']['title']}")
    print(f"  Panels: {len(dashboard['dashboard']['panels'])}")
    print(f"  Time range: {dashboard['dashboard']['time']['from']} to {dashboard['dashboard']['time']['to']}")
    
    # Example 11: Metrics server
    print("\n11. Metrics Server")
    print("-" * 40)
    
    print("Metrics server can be started with:")
    print("  collector.start_server(port=9090)")
    print("  Metrics available at: http://localhost:9090/metrics")
    
    # Example 12: Alert rules
    print("\n12. Alert Rules Example")
    print("-" * 40)
    
    print("Example Prometheus alert rules:")
    print("""
  - alert: HighErrorRate
    expr: rate(errors_total[5m]) > 0.05
    labels:
      severity: warning
    annotations:
      summary: High error rate detected
    
  - alert: HighMemoryUsage
    expr: memory_usage_bytes > 1073741824
    labels:
      severity: critical
    annotations:
      summary: Memory usage above 1GB
    """)
    
    print("\n" + "=" * 60)
    print("✅ Monitoring examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
