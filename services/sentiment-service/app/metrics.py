from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "elyssa_sentiment_requests_total",
    "Total requests to sentiment service",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "elyssa_sentiment_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
)
