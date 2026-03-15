from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "elyssa_inference_requests_total",
    "Total requests to inference service",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "elyssa_inference_request_duration_seconds",
    "Request latency in seconds",
)
