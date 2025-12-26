"""
DuckDB + Parquet time series storage for high-performance analytical queries.

Columnar storage enables fast queries like:
- Moving average of last 10k steps
- Gradient statistics over time windows
- Cross-run metric comparisons
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from atlas_sdk.types import Metric, MetricType


class TimeSeriesStore:
    """
    DuckDB-backed time series store with Parquet persistence.

    Uses DuckDB for fast analytical queries and Parquet for
    efficient columnar storage.
    """

    # Batch size before flushing to Parquet
    BATCH_SIZE = 1000

    def __init__(self, data_dir: Path, run_id: str, read_only: bool = False):
        self.data_dir = data_dir
        self.run_id = run_id
        self.read_only = read_only
        self.metrics_dir = data_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffer for batching
        self._buffer: List[Dict[str, Any]] = []

        # DuckDB connection
        db_path = str(data_dir / f"metrics_{run_id}.duckdb") if not read_only else ":memory:"
        self.conn = duckdb.connect(db_path, read_only=read_only)
        self._init_schema()

        # Load existing Parquet files if in read-only mode
        if read_only:
            self._load_parquet_files()

    def _init_schema(self) -> None:
        """Initialize DuckDB schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                name VARCHAR NOT NULL,
                value DOUBLE,
                step BIGINT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                metric_type VARCHAR DEFAULT 'scalar',
                metadata JSON
            )
        """)

        # Indices for common query patterns
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_name_step
            ON metrics (name, step)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_step
            ON metrics (step)
        """)

    def _load_parquet_files(self) -> None:
        """Load existing Parquet files into DuckDB."""
        pattern = self.metrics_dir / f"{self.run_id}_*.parquet"
        parquet_files = list(self.metrics_dir.glob(f"{self.run_id}_*.parquet"))

        for pq_file in parquet_files:
            try:
                self.conn.execute(f"""
                    INSERT INTO metrics
                    SELECT * FROM read_parquet('{pq_file}')
                """)
            except duckdb.Error:
                pass  # Skip corrupted files

    def log(self, metric: Metric) -> None:
        """Log a single metric."""
        if self.read_only:
            raise RuntimeError("Cannot log in read-only mode")

        record = {
            "name": metric.name,
            "value": float(metric.value) if isinstance(metric.value, (int, float)) else None,
            "step": metric.step,
            "timestamp": metric.timestamp,
            "metric_type": metric.metric_type.value,
            "metadata": json.dumps(metric.metadata) if metric.metadata else None,
        }

        self._buffer.append(record)

        if len(self._buffer) >= self.BATCH_SIZE:
            self._flush()

    def log_batch(self, metrics: List[Metric]) -> None:
        """Log multiple metrics at once."""
        if self.read_only:
            raise RuntimeError("Cannot log in read-only mode")

        for metric in metrics:
            record = {
                "name": metric.name,
                "value": float(metric.value) if isinstance(metric.value, (int, float)) else None,
                "step": metric.step,
                "timestamp": metric.timestamp,
                "metric_type": metric.metric_type.value,
                "metadata": json.dumps(metric.metadata) if metric.metadata else None,
            }
            self._buffer.append(record)

        if len(self._buffer) >= self.BATCH_SIZE:
            self._flush()

    def _flush(self) -> None:
        """Flush buffer to DuckDB and Parquet."""
        if not self._buffer:
            return

        # Create Arrow table from buffer
        table = pa.Table.from_pylist(self._buffer)

        # Insert into DuckDB
        self.conn.execute("INSERT INTO metrics SELECT * FROM table")

        # Also write to Parquet for persistence
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        parquet_path = self.metrics_dir / f"{self.run_id}_{timestamp}.parquet"
        pq.write_table(table, parquet_path, compression="snappy")

        self._buffer.clear()

    def query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results."""
        result = self.conn.execute(sql).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]

    def get_metric(
        self,
        name: str,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get metric values by name with optional step range."""
        # Flush any pending data first
        if not self.read_only:
            self._flush()

        query = f"SELECT * FROM metrics WHERE name = '{name}'"

        if start_step is not None:
            query += f" AND step >= {start_step}"
        if end_step is not None:
            query += f" AND step <= {end_step}"

        query += " ORDER BY step"

        if limit is not None:
            query += f" LIMIT {limit}"

        return self.query(query)

    def get_latest(self, name: str, n: int = 1) -> List[Dict[str, Any]]:
        """Get the N most recent values for a metric."""
        if not self.read_only:
            self._flush()

        return self.query(f"""
            SELECT * FROM metrics
            WHERE name = '{name}'
            ORDER BY step DESC
            LIMIT {n}
        """)

    def get_moving_average(
        self,
        name: str,
        window_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """Calculate moving average for a metric."""
        if not self.read_only:
            self._flush()

        return self.query(f"""
            SELECT
                step,
                value,
                AVG(value) OVER (
                    ORDER BY step
                    ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW
                ) as moving_avg
            FROM metrics
            WHERE name = '{name}'
            ORDER BY step
        """)

    def get_statistics(self, name: str) -> Dict[str, Any]:
        """Get statistical summary for a metric."""
        if not self.read_only:
            self._flush()

        result = self.query(f"""
            SELECT
                COUNT(*) as count,
                MIN(value) as min,
                MAX(value) as max,
                AVG(value) as mean,
                STDDEV(value) as std,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median
            FROM metrics
            WHERE name = '{name}'
        """)
        return result[0] if result else {}

    def list_metrics(self) -> List[str]:
        """List all metric names."""
        if not self.read_only:
            self._flush()

        result = self.query("SELECT DISTINCT name FROM metrics ORDER BY name")
        return [r["name"] for r in result]

    def get_step_range(self) -> Dict[str, int]:
        """Get min and max step values."""
        if not self.read_only:
            self._flush()

        result = self.query("SELECT MIN(step) as min_step, MAX(step) as max_step FROM metrics")
        return result[0] if result else {"min_step": 0, "max_step": 0}

    def compare_runs(
        self,
        other_store: "TimeSeriesStore",
        metric_name: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Compare a metric between two runs."""
        return {
            self.run_id: self.get_metric(metric_name),
            other_store.run_id: other_store.get_metric(metric_name),
        }

    def export_to_parquet(self, output_path: Path) -> None:
        """Export all metrics to a single Parquet file."""
        if not self.read_only:
            self._flush()

        result = self.conn.execute("SELECT * FROM metrics ORDER BY step").fetchall()
        columns = [desc[0] for desc in self.conn.description]

        if result:
            table = pa.Table.from_pydict({
                col: [row[i] for row in result]
                for i, col in enumerate(columns)
            })
            pq.write_table(table, output_path, compression="snappy")

    def close(self) -> None:
        """Close the store and flush any pending data."""
        if not self.read_only:
            self._flush()
        self.conn.close()


def create_comparison_view(
    stores: List[TimeSeriesStore],
    metric_name: str,
) -> pa.Table:
    """Create a unified Arrow table for comparing metrics across runs."""
    all_data = []

    for store in stores:
        data = store.get_metric(metric_name)
        for row in data:
            row["run_id"] = store.run_id
            all_data.append(row)

    return pa.Table.from_pylist(all_data)
