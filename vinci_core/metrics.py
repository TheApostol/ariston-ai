from prometheus_client import Counter, Histogram, start_http_server

REQUEST_COUNT = Counter(
    'vinci_requests_total',
    'Total number of requests processed',
    ['model_name', 'layer']
)

REQUEST_LATENCY = Histogram(
    'vinci_request_latency_seconds',
    'Latency of AI requests',
    ['model_name']
)

FALLBACK_COUNT = Counter(
    'vinci_fallbacks_total',
    'Total number of fallback routes taken'
)

def start_metrics_server(port: int = 9090):
    """Start the Prometheus metrics endpoint in the background."""
    start_http_server(port)
    print(f"Metrics server started on port {port}")
