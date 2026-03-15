from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "elyssa_gateway_requests_total",
    "Total requests to gateway",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "elyssa_gateway_request_duration_seconds",
    "End-to-end request latency in seconds",
)
