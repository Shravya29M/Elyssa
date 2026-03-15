from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "elyssa_response_requests_total",
    "Total requests to response service",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "elyssa_response_request_duration_seconds",
    "Request latency in seconds",
)
