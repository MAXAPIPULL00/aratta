"""
Resilience metrics — counters, gauges, histograms for monitoring.

Exportable to Prometheus text format or JSON.
"""

import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

logger = logging.getLogger("aratta.metrics")


@dataclass
class CounterMetric:
    name: str
    description: str
    value: int = 0
    labels: dict[tuple, int] = field(default_factory=dict)

    def inc(self, labels: dict[str, str] = None, value: int = 1):
        self.value += value
        if labels:
            key = tuple(sorted(labels.items()))
            self.labels[key] = self.labels.get(key, 0) + value


@dataclass
class GaugeMetric:
    name: str
    description: str
    value: float = 0.0

    def set(self, value: float):
        self.value = value


@dataclass
class HistogramMetric:
    name: str
    description: str
    observations: list[float] = field(default_factory=list)
    sum_value: float = 0.0
    count: int = 0

    def observe(self, value: float):
        self.observations.append(value)
        self.sum_value += value
        self.count += 1
        if len(self.observations) > 1000:
            self.observations = self.observations[-1000:]

    def percentile(self, p: float) -> float:
        if not self.observations:
            return 0.0
        s = sorted(self.observations)
        return s[min(int(len(s) * p), len(s) - 1)]


class ResilienceMetrics:
    """Thread-safe metrics collector."""

    def __init__(self):
        self._lock = Lock()
        self.provider_errors = CounterMetric("aratta_provider_errors_total", "Provider errors by type")
        self.circuit_opens = CounterMetric("aratta_circuit_opens_total", "Circuit breaker opens")
        self.heal_requests = CounterMetric("aratta_heal_requests_total", "Heal requests sent")
        self.open_circuits = GaugeMetric("aratta_open_circuits", "Currently open circuits")
        self.heal_duration = HistogramMetric("aratta_heal_duration_seconds", "Healing cycle duration")

    def record_provider_error(self, provider: str, error_type: str):
        with self._lock:
            self.provider_errors.inc({"provider": provider, "error_type": error_type})

    def record_circuit_open(self, provider: str):
        with self._lock:
            self.circuit_opens.inc({"provider": provider})

    def record_heal_request(self, provider: str, error_type: str):
        with self._lock:
            self.heal_requests.inc({"provider": provider, "error_type": error_type})

    def set_open_circuits(self, count: int):
        with self._lock:
            self.open_circuits.set(count)

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "provider_errors": self.provider_errors.value,
                "circuit_opens": self.circuit_opens.value,
                "heal_requests": self.heal_requests.value,
                "open_circuits": self.open_circuits.value,
            }


_metrics: ResilienceMetrics | None = None


def get_metrics() -> ResilienceMetrics:
    global _metrics
    if _metrics is None:
        _metrics = ResilienceMetrics()
    return _metrics
