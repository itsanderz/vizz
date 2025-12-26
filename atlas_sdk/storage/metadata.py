"""
SQLite-based metadata storage with WAL mode for concurrent access.

Stores:
- Run metadata (name, config, tags, status)
- User votes on insights
- Experiment configurations
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from uuid import UUID

from atlas_sdk.types import Artifact, Insight, InsightVote, RunConfig, RunStatus


class MetadataStore:
    """
    SQLite metadata store with Write-Ahead Logging (WAL) mode.

    WAL mode allows concurrent reads from the UI while training writes,
    providing ACID compliance without blocking.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database transactions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._transaction() as cursor:
            # Schema version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)

            # Runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT 'default',
                    status TEXT NOT NULL DEFAULT 'running',
                    config TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_seconds REAL
                )
            """)

            # Tags table (many-to-many)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS run_tags (
                    run_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (run_id, tag),
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)

            # Artifacts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER DEFAULT 0,
                    checksum TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)

            # Insights table (AI-generated)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT DEFAULT 'info',
                    related_metrics TEXT,
                    visualization_spec TEXT,
                    timestamp TEXT NOT NULL,
                    votes_up INTEGER DEFAULT 0,
                    votes_down INTEGER DEFAULT 0,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)

            # Votes table (user feedback)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS votes (
                    id TEXT PRIMARY KEY,
                    insight_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    vote INTEGER NOT NULL,
                    feedback TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (insight_id) REFERENCES insights(id),
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)

            # Indices for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_run ON insights(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_votes_insight ON votes(insight_id)")

    # ==================== Run Operations ====================

    def create_run(
        self,
        run_id: str,
        config: RunConfig,
    ) -> None:
        """Create a new run record."""
        now = datetime.utcnow().isoformat()
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO runs (id, name, project, status, config, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    config.name,
                    config.project,
                    RunStatus.RUNNING.value,
                    json.dumps(config.config),
                    config.notes,
                    now,
                    now,
                ),
            )

            # Insert tags
            for tag in config.tags:
                cursor.execute(
                    "INSERT OR IGNORE INTO run_tags (run_id, tag) VALUES (?, ?)",
                    (run_id, tag),
                )

    def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Update run status and optionally duration."""
        now = datetime.utcnow().isoformat()
        with self._transaction() as cursor:
            if status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INTERRUPTED):
                cursor.execute(
                    """
                    UPDATE runs
                    SET status = ?, updated_at = ?, finished_at = ?, duration_seconds = ?
                    WHERE id = ?
                    """,
                    (status.value, now, now, duration_seconds, run_id),
                )
            else:
                cursor.execute(
                    "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
                    (status.value, now, run_id),
                )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run by ID."""
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                run = dict(row)
                run["config"] = json.loads(run["config"]) if run["config"] else {}
                # Get tags
                cursor.execute("SELECT tag FROM run_tags WHERE run_id = ?", (run_id,))
                run["tags"] = [r["tag"] for r in cursor.fetchall()]
                return run
            return None

    def list_runs(
        self,
        project: Optional[str] = None,
        status: Optional[RunStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List runs with optional filtering."""
        with self._transaction() as cursor:
            query = "SELECT * FROM runs WHERE 1=1"
            params: List[Any] = []

            if project:
                query += " AND project = ?"
                params.append(project)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            runs = []
            for row in cursor.fetchall():
                run = dict(row)
                run["config"] = json.loads(run["config"]) if run["config"] else {}
                runs.append(run)

            return runs

    # ==================== Artifact Operations ====================

    def create_artifact(self, run_id: str, artifact: Artifact) -> None:
        """Record an artifact."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO artifacts (id, run_id, name, artifact_type, path, size_bytes, checksum, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(artifact.id),
                    run_id,
                    artifact.name,
                    artifact.artifact_type.value,
                    artifact.path,
                    artifact.size_bytes,
                    artifact.checksum,
                    json.dumps(artifact.metadata),
                    artifact.created_at.isoformat(),
                ),
            )

    def get_artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all artifacts for a run."""
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM artifacts WHERE run_id = ?", (run_id,))
            return [dict(row) for row in cursor.fetchall()]

    # ==================== Insight Operations ====================

    def create_insight(self, insight: Insight) -> None:
        """Store an AI-generated insight."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO insights (id, run_id, insight_type, title, description, severity, related_metrics, visualization_spec, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(insight.id),
                    str(insight.run_id),
                    insight.insight_type,
                    insight.title,
                    insight.description,
                    insight.severity,
                    json.dumps(insight.related_metrics),
                    json.dumps(insight.visualization_spec) if insight.visualization_spec else None,
                    insight.timestamp.isoformat(),
                ),
            )

    def get_insights(
        self,
        run_id: str,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get insights for a run."""
        with self._transaction() as cursor:
            query = "SELECT * FROM insights WHERE run_id = ?"
            params: List[Any] = [run_id]

            if severity:
                query += " AND severity = ?"
                params.append(severity)

            query += " ORDER BY timestamp DESC"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    # ==================== Vote Operations ====================

    def record_vote(self, vote: InsightVote) -> None:
        """Record a user vote on an insight."""
        with self._transaction() as cursor:
            # Insert vote
            cursor.execute(
                """
                INSERT INTO votes (id, insight_id, run_id, vote, feedback, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(vote.id),
                    str(vote.insight_id),
                    str(vote.run_id),
                    vote.vote,
                    vote.feedback,
                    vote.timestamp.isoformat(),
                ),
            )

            # Update insight vote counts
            if vote.vote > 0:
                cursor.execute(
                    "UPDATE insights SET votes_up = votes_up + 1 WHERE id = ?",
                    (str(vote.insight_id),),
                )
            else:
                cursor.execute(
                    "UPDATE insights SET votes_down = votes_down + 1 WHERE id = ?",
                    (str(vote.insight_id),),
                )

    def get_vote_stats(self) -> Dict[str, Any]:
        """Get aggregated vote statistics for recommendation training."""
        with self._transaction() as cursor:
            cursor.execute("""
                SELECT
                    i.insight_type,
                    SUM(CASE WHEN v.vote > 0 THEN 1 ELSE 0 END) as upvotes,
                    SUM(CASE WHEN v.vote < 0 THEN 1 ELSE 0 END) as downvotes
                FROM insights i
                LEFT JOIN votes v ON i.id = v.insight_id
                GROUP BY i.insight_type
            """)
            return {row["insight_type"]: {"up": row["upvotes"], "down": row["downvotes"]}
                    for row in cursor.fetchall()}

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
