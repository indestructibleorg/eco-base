"""MetricsEngine — in-memory time series storage with aggregation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from observability_platform.domain.entities import Metric
from observability_platform.domain.value_objects import MetricType
from observability_platform.domain.events import MetricRecorded
from observability_platform.domain.exceptions import MetricNotFoundError


@dataclass(frozen=True, slots=True)
class TimeSeriesPoint:
    """A single data point in a time series."""
    timestamp: float
    value: float
    tags: dict[str, str] = field(default_factory=dict)


class MetricsEngine:
    """In-memory time series store with query and aggregation capabilities.

    Stores metric data points indexed by name, supports time-range queries,
    and computes aggregations (avg, max, min, p99).
    """

    def __init__(self) -> None:
        self._series: dict[str, list[TimeSeriesPoint]] = {}
        self._metric_types: dict[str, MetricType] = {}

    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: dict[str, str] | None = None,
        timestamp: float | None = None,
    ) -> Metric:
        """Record a metric data point.

        For COUNTER types, the value is accumulated on top of the last known value.
        For GAUGE and HISTOGRAM types, the value is stored as-is.
        """
        ts = timestamp or time.time()
        resolved_tags = tags or {}

        if name not in self._series:
            self._series[name] = []
            self._metric_types[name] = metric_type

        if metric_type == MetricType.COUNTER and self._series[name]:
            # Counters accumulate
            last = self._series[name][-1].value
            actual_value = last + value
        else:
            actual_value = value

        point = TimeSeriesPoint(timestamp=ts, value=actual_value, tags=resolved_tags)
        self._series[name].append(point)

        metric = Metric(
            name=name,
            value=actual_value,
            metric_type=metric_type,
            timestamp=ts,
            tags=resolved_tags,
        )
        return metric

    def query(
        self,
        name: str,
        time_range: tuple[float, float] | None = None,
    ) -> list[TimeSeriesPoint]:
        """Query data points for a metric, optionally filtered by time range.

        Args:
            name: Metric name.
            time_range: Optional (start, end) timestamps to filter points.

        Returns:
            List of TimeSeriesPoint within the range.

        Raises:
            MetricNotFoundError: If the metric name does not exist.
        """
        if name not in self._series:
            raise MetricNotFoundError(name)

        points = self._series[name]
        if time_range is None:
            return list(points)

        start, end = time_range
        return [p for p in points if start <= p.timestamp <= end]

    def get_latest(self, name: str) -> TimeSeriesPoint | None:
        """Get the most recent data point for a metric."""
        if name not in self._series or not self._series[name]:
            return None
        return self._series[name][-1]

    def aggregate(
        self,
        name: str,
        function: str,
        time_range: tuple[float, float] | None = None,
    ) -> float:
        """Compute an aggregation over metric data points.

        Args:
            name: Metric name.
            function: One of 'avg', 'max', 'min', 'sum', 'count', 'p99'.
            time_range: Optional (start, end) timestamps.

        Returns:
            The computed aggregation value.
        """
        points = self.query(name, time_range)
        if not points:
            return 0.0

        values = [p.value for p in points]

        if function == "avg":
            return sum(values) / len(values)
        elif function == "max":
            return max(values)
        elif function == "min":
            return min(values)
        elif function == "sum":
            return sum(values)
        elif function == "count":
            return float(len(values))
        elif function == "p99":
            return self._percentile(values, 99)
        else:
            raise ValueError(f"Unknown aggregation function: {function}")

    @property
    def metric_names(self) -> list[str]:
        """Return all known metric names."""
        return list(self._series.keys())

    def clear(self, name: str | None = None) -> None:
        """Clear data points. If name is None, clear everything."""
        if name is None:
            self._series.clear()
            self._metric_types.clear()
        elif name in self._series:
            del self._series[name]
            self._metric_types.pop(name, None)

    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        """Compute the given percentile from a list of values."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        k = (pct / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)
