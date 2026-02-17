"""
Monitoring and Observability - Prometheus metrics, Grafana dashboards, and tracing.

Features:
- Prometheus metrics export
- Custom metrics (counters, gauges, histograms)
- Health checks and readiness probes
- Performance monitoring
- Request/response tracking
- Error rate monitoring
- Grafana dashboard templates
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from enum import Enum
import json
from pathlib import Path

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, generate_latest,
        CONTENT_TYPE_LATEST, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Stub types so class definitions and type hints don't crash
    Counter = Gauge = Histogram = Summary = object
    CollectorRegistry = generate_latest = None
    CONTENT_TYPE_LATEST = ""
    start_http_server = None


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Metric definition."""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class HealthStatus:
    """Health check status."""
    healthy: bool
    status: str  # healthy, degraded, unhealthy
    checks: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "healthy": self.healthy,
            "status": self.status,
            "checks": self.checks,
            "timestamp": self.timestamp.isoformat()
        }


class MetricsCollector:
    """
    Collect and export Prometheus metrics.
    
    Features:
    - Standard metrics (requests, errors, latency)
    - Custom metrics
    - Label support
    - Automatic HTTP server for scraping
    """
    
    def __init__(self, namespace: str = "opensable", enable_defaults: bool = True):
        """
        Initialize metrics collector.
        
        Args:
            namespace: Metric namespace prefix
            enable_defaults: Enable default metrics
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client not installed: pip install prometheus-client"
            )
        
        self.namespace = namespace
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        
        if enable_defaults:
            self._setup_default_metrics()
    
    def _setup_default_metrics(self):
        """Setup default metrics."""
        # Request metrics
        self.metrics["requests_total"] = Counter(
            f"{self.namespace}_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
            registry=self.registry
        )
        
        self.metrics["request_duration_seconds"] = Histogram(
            f"{self.namespace}_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Error metrics
        self.metrics["errors_total"] = Counter(
            f"{self.namespace}_errors_total",
            "Total number of errors",
            ["error_type"],
            registry=self.registry
        )
        
        # Agent metrics
        self.metrics["agent_active"] = Gauge(
            f"{self.namespace}_agent_active",
            "Number of active agents",
            registry=self.registry
        )
        
        self.metrics["agent_tasks_total"] = Counter(
            f"{self.namespace}_agent_tasks_total",
            "Total number of agent tasks",
            ["agent_type", "status"],
            registry=self.registry
        )
        
        # Memory metrics
        self.metrics["memory_usage_bytes"] = Gauge(
            f"{self.namespace}_memory_usage_bytes",
            "Memory usage in bytes",
            ["type"],
            registry=self.registry
        )
        
        # Database metrics
        self.metrics["db_connections_active"] = Gauge(
            f"{self.namespace}_db_connections_active",
            "Active database connections",
            ["database"],
            registry=self.registry
        )
        
        self.metrics["db_query_duration_seconds"] = Histogram(
            f"{self.namespace}_db_query_duration_seconds",
            "Database query duration in seconds",
            ["operation"],
            registry=self.registry
        )
    
    def create_counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Counter:
        """Create a custom counter metric."""
        full_name = f"{self.namespace}_{name}"
        counter = Counter(
            full_name,
            description,
            labels or [],
            registry=self.registry
        )
        self.metrics[name] = counter
        return counter
    
    def create_gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Gauge:
        """Create a custom gauge metric."""
        full_name = f"{self.namespace}_{name}"
        gauge = Gauge(
            full_name,
            description,
            labels or [],
            registry=self.registry
        )
        self.metrics[name] = gauge
        return gauge
    
    def create_histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ) -> Histogram:
        """Create a custom histogram metric."""
        full_name = f"{self.namespace}_{name}"
        histogram = Histogram(
            full_name,
            description,
            labels or [],
            buckets=buckets,
            registry=self.registry
        )
        self.metrics[name] = histogram
        return histogram
    
    def inc_requests(self, method: str, endpoint: str, status: int):
        """Increment request counter."""
        self.metrics["requests_total"].labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
    
    def observe_request_duration(self, method: str, endpoint: str, duration: float):
        """Observe request duration."""
        self.metrics["request_duration_seconds"].labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def inc_errors(self, error_type: str):
        """Increment error counter."""
        self.metrics["errors_total"].labels(error_type=error_type).inc()
    
    def set_active_agents(self, count: int):
        """Set active agents count."""
        self.metrics["agent_active"].set(count)
    
    def inc_agent_tasks(self, agent_type: str, status: str):
        """Increment agent task counter."""
        self.metrics["agent_tasks_total"].labels(
            agent_type=agent_type,
            status=status
        ).inc()
    
    def set_memory_usage(self, memory_type: str, bytes_used: int):
        """Set memory usage."""
        self.metrics["memory_usage_bytes"].labels(type=memory_type).set(bytes_used)
    
    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry)
    
    def start_server(self, port: int = 9090):
        """Start HTTP server for metrics scraping."""
        start_http_server(port, registry=self.registry)


class HealthChecker:
    """
    Health check system for monitoring service status.
    
    Features:
    - Multiple health checks
    - Liveness probes
    - Readiness probes
    - Dependency checks
    """
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable[[], bool]] = {}
        self._cache: Optional[HealthStatus] = None
        self._cache_ttl = 5  # seconds
        self._last_check = None
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Function that returns True if healthy
        """
        self.checks[name] = check_func
    
    async def check_health(self, use_cache: bool = True) -> HealthStatus:
        """
        Run all health checks.
        
        Args:
            use_cache: Use cached result if available
        
        Returns:
            HealthStatus
        """
        # Check cache
        if use_cache and self._cache and self._last_check:
            elapsed = (datetime.now() - self._last_check).total_seconds()
            if elapsed < self._cache_ttl:
                return self._cache
        
        # Run checks
        check_results = {}
        all_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                check_results[name] = {
                    "healthy": result,
                    "status": "ok" if result else "failed"
                }
                
                if not result:
                    all_healthy = False
                    
            except Exception as e:
                check_results[name] = {
                    "healthy": False,
                    "status": "error",
                    "error": str(e)
                }
                all_healthy = False
        
        # Determine overall status
        if all_healthy:
            status = "healthy"
        elif any(check["healthy"] for check in check_results.values()):
            status = "degraded"
        else:
            status = "unhealthy"
        
        health_status = HealthStatus(
            healthy=all_healthy,
            status=status,
            checks=check_results
        )
        
        # Update cache
        self._cache = health_status
        self._last_check = datetime.now()
        
        return health_status
    
    async def liveness_probe(self) -> bool:
        """
        Liveness probe - check if service is alive.
        
        Returns:
            True if alive
        """
        return True  # If this code runs, service is alive
    
    async def readiness_probe(self) -> bool:
        """
        Readiness probe - check if service is ready to accept traffic.
        
        Returns:
            True if ready
        """
        health = await self.check_health()
        return health.status != "unhealthy"


class PerformanceMonitor:
    """
    Monitor performance metrics.
    
    Features:
    - Request tracking
    - Latency monitoring
    - Throughput measurement
    - Resource usage tracking
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Rolling window size for metrics
        """
        self.window_size = window_size
        self.request_times: List[float] = []
        self.request_sizes: List[int] = []
        self.error_count = 0
        self.total_requests = 0
        self.start_time = time.time()
    
    def record_request(
        self,
        duration: float,
        size: int = 0,
        success: bool = True
    ):
        """
        Record a request.
        
        Args:
            duration: Request duration in seconds
            size: Response size in bytes
            success: Whether request was successful
        """
        self.total_requests += 1
        
        # Track in rolling window
        self.request_times.append(duration)
        if len(self.request_times) > self.window_size:
            self.request_times.pop(0)
        
        self.request_sizes.append(size)
        if len(self.request_sizes) > self.window_size:
            self.request_sizes.pop(0)
        
        if not success:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.request_times:
            return {
                "requests_total": 0,
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "error_rate": 0,
                "throughput_rps": 0
            }
        
        # Calculate percentiles
        sorted_times = sorted(self.request_times)
        n = len(sorted_times)
        
        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        
        # Calculate throughput
        elapsed = time.time() - self.start_time
        throughput = self.total_requests / elapsed if elapsed > 0 else 0
        
        return {
            "requests_total": self.total_requests,
            "avg_latency_ms": sum(self.request_times) / len(self.request_times) * 1000,
            "p50_latency_ms": sorted_times[p50_idx] * 1000,
            "p95_latency_ms": sorted_times[p95_idx] * 1000,
            "p99_latency_ms": sorted_times[p99_idx] * 1000,
            "error_rate": (self.error_count / self.total_requests * 100) if self.total_requests > 0 else 0,
            "throughput_rps": throughput,
            "avg_response_size_bytes": sum(self.request_sizes) / len(self.request_sizes) if self.request_sizes else 0
        }


class GrafanaDashboard:
    """
    Generate Grafana dashboard JSON.
    
    Creates dashboards for Open-Sable metrics.
    """
    
    @staticmethod
    def create_dashboard(title: str = "Open-Sable Metrics") -> Dict[str, Any]:
        """Create Grafana dashboard JSON."""
        return {
            "dashboard": {
                "title": title,
                "tags": ["opensable", "ai", "agents"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 0,
                "panels": [
                    # Requests panel
                    {
                        "id": 1,
                        "title": "Request Rate",
                        "type": "graph",
                        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                        "targets": [{
                            "expr": "rate(opensable_requests_total[5m])",
                            "legendFormat": "{{method}} {{endpoint}}"
                        }]
                    },
                    # Latency panel
                    {
                        "id": 2,
                        "title": "Request Latency (p95)",
                        "type": "graph",
                        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
                        "targets": [{
                            "expr": "histogram_quantile(0.95, rate(opensable_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "{{method}} {{endpoint}}"
                        }]
                    },
                    # Error rate panel
                    {
                        "id": 3,
                        "title": "Error Rate",
                        "type": "graph",
                        "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
                        "targets": [{
                            "expr": "rate(opensable_errors_total[5m])",
                            "legendFormat": "{{error_type}}"
                        }]
                    },
                    # Active agents panel
                    {
                        "id": 4,
                        "title": "Active Agents",
                        "type": "stat",
                        "gridPos": {"x": 12, "y": 8, "w": 6, "h": 4},
                        "targets": [{
                            "expr": "opensable_agent_active"
                        }]
                    },
                    # Agent tasks panel
                    {
                        "id": 5,
                        "title": "Agent Tasks",
                        "type": "graph",
                        "gridPos": {"x": 0, "y": 16, "w": 24, "h": 8},
                        "targets": [{
                            "expr": "rate(opensable_agent_tasks_total[5m])",
                            "legendFormat": "{{agent_type}} - {{status}}"
                        }]
                    },
                    # Memory usage panel
                    {
                        "id": 6,
                        "title": "Memory Usage",
                        "type": "graph",
                        "gridPos": {"x": 0, "y": 24, "w": 12, "h": 8},
                        "targets": [{
                            "expr": "opensable_memory_usage_bytes",
                            "legendFormat": "{{type}}"
                        }]
                    },
                    # DB connections panel
                    {
                        "id": 7,
                        "title": "Database Connections",
                        "type": "stat",
                        "gridPos": {"x": 12, "y": 24, "w": 6, "h": 4},
                        "targets": [{
                            "expr": "opensable_db_connections_active"
                        }]
                    }
                ]
            }
        }
    
    @staticmethod
    def save_dashboard(dashboard: Dict[str, Any], output_file: str):
        """Save dashboard to file."""
        Path(output_file).write_text(json.dumps(dashboard, indent=2))


# Example usage
async def main():
    """Example monitoring usage."""
    
    print("=" * 50)
    print("Monitoring and Observability Example")
    print("=" * 50)
    
    # Initialize metrics collector
    print("\n1. Metrics Collection")
    metrics = MetricsCollector()
    
    # Simulate some requests
    for i in range(10):
        method = "GET" if i % 2 == 0 else "POST"
        endpoint = "/api/chat" if i % 3 == 0 else "/api/agents"
        status = 200 if i % 5 != 0 else 500
        duration = 0.1 + (i * 0.05)
        
        metrics.inc_requests(method, endpoint, status)
        metrics.observe_request_duration(method, endpoint, duration)
        
        if status == 500:
            metrics.inc_errors("internal_error")
    
    # Set agent metrics
    metrics.set_active_agents(5)
    metrics.inc_agent_tasks("researcher", "completed")
    metrics.inc_agent_tasks("writer", "completed")
    
    print("  Recorded metrics for 10 requests")
    
    # Export metrics
    prometheus_data = metrics.export_metrics()
    print(f"  Exported {len(prometheus_data)} bytes of metrics")
    
    # Health checks
    print("\n2. Health Checks")
    health_checker = HealthChecker()
    
    # Register some checks
    health_checker.register_check(
        "database",
        lambda: True  # Simulate healthy database
    )
    
    health_checker.register_check(
        "redis",
        lambda: True  # Simulate healthy Redis
    )
    
    health_checker.register_check(
        "api",
        lambda: True  # Simulate healthy API
    )
    
    health_status = await health_checker.check_health()
    print(f"  Overall status: {health_status.status}")
    print(f"  Healthy: {health_status.healthy}")
    print(f"  Checks:")
    for name, result in health_status.checks.items():
        print(f"    - {name}: {result['status']}")
    
    # Performance monitoring
    print("\n3. Performance Monitoring")
    perf_monitor = PerformanceMonitor()
    
    # Simulate requests
    import random
    for _ in range(100):
        duration = random.uniform(0.01, 0.5)
        size = random.randint(100, 10000)
        success = random.random() > 0.05  # 5% error rate
        
        perf_monitor.record_request(duration, size, success)
    
    stats = perf_monitor.get_stats()
    print(f"  Total requests: {stats['requests_total']}")
    print(f"  Avg latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
    print(f"  P99 latency: {stats['p99_latency_ms']:.2f}ms")
    print(f"  Error rate: {stats['error_rate']:.2f}%")
    print(f"  Throughput: {stats['throughput_rps']:.2f} req/s")
    
    # Generate Grafana dashboard
    print("\n4. Grafana Dashboard")
    dashboard = GrafanaDashboard.create_dashboard()
    dashboard_file = "/tmp/opensable_dashboard.json"
    GrafanaDashboard.save_dashboard(dashboard, dashboard_file)
    print(f"  Dashboard saved to {dashboard_file}")
    print(f"  Panels: {len(dashboard['dashboard']['panels'])}")
    
    print("\n✅ Monitoring examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
