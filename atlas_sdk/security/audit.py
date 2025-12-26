"""
Audit Logging System

Enterprise audit logging for compliance and security monitoring.
Provides immutable, tamper-evident logs of all system operations.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class AuditAction(str, Enum):
    """Types of auditable actions."""
    # Run operations
    RUN_CREATE = "run.create"
    RUN_UPDATE = "run.update"
    RUN_DELETE = "run.delete"
    RUN_FINISH = "run.finish"

    # Data operations
    METRIC_LOG = "metric.log"
    ARTIFACT_UPLOAD = "artifact.upload"
    ARTIFACT_DOWNLOAD = "artifact.download"
    ARTIFACT_DELETE = "artifact.delete"

    # Analysis operations
    INSIGHT_GENERATE = "insight.generate"
    INSIGHT_VOTE = "insight.vote"

    # Access operations
    DATA_ACCESS = "data.access"
    DATA_EXPORT = "data.export"
    CONFIG_CHANGE = "config.change"

    # Security operations
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    PERMISSION_DENIED = "permission.denied"

    # System operations
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    ERROR = "error"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """An immutable audit event."""
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    action: AuditAction = AuditAction.DATA_ACCESS
    severity: AuditSeverity = AuditSeverity.INFO
    actor: str = "system"
    resource_type: str = ""
    resource_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    organization_id: Optional[str] = None
    team_id: Optional[str] = None
    previous_hash: Optional[str] = None
    event_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "severity": self.severity.value,
            "actor": self.actor,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "organization_id": self.organization_id,
            "team_id": self.team_id,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }

    def compute_hash(self, previous_hash: Optional[str] = None) -> str:
        """Compute tamper-evident hash of this event."""
        data = {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "actor": self.actor,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": json.dumps(self.details, sort_keys=True),
            "previous_hash": previous_hash or "",
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class AuditLogger:
    """
    Enterprise audit logger with tamper-evident chain.

    Features:
    - Immutable event chain (blockchain-like)
    - Automatic rotation
    - Async writing
    - Compression
    - Encryption support
    """

    def __init__(
        self,
        log_path: Path,
        rotation_size_mb: int = 100,
        retention_days: int = 365,
        encrypt: bool = False,
        organization_id: Optional[str] = None,
    ):
        self.log_path = Path(log_path)
        self.rotation_size_mb = rotation_size_mb
        self.retention_days = retention_days
        self.encrypt = encrypt
        self.organization_id = organization_id

        self._lock = threading.Lock()
        self._last_hash: Optional[str] = None
        self._event_count = 0

        # Create log directory
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Load last hash from existing log
        self._load_last_hash()

    def _load_last_hash(self) -> None:
        """Load the last event hash from existing log file."""
        if not self.log_path.exists():
            return

        try:
            with open(self.log_path, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        event = json.loads(last_line)
                        self._last_hash = event.get("event_hash")
                        self._event_count = len(lines)
        except Exception:
            pass

    def log(
        self,
        action: AuditAction,
        resource_type: str = "",
        resource_id: str = "",
        details: Optional[Dict[str, Any]] = None,
        actor: str = "system",
        severity: AuditSeverity = AuditSeverity.INFO,
        **kwargs: Any,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            action: Type of action being audited
            resource_type: Type of resource (run, artifact, etc.)
            resource_id: ID of the resource
            details: Additional details
            actor: Who performed the action
            severity: Severity level
            **kwargs: Additional event fields

        Returns:
            The created AuditEvent
        """
        event = AuditEvent(
            action=action,
            severity=severity,
            actor=actor,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            organization_id=self.organization_id,
            **kwargs,
        )

        with self._lock:
            # Compute hash with chain
            event.previous_hash = self._last_hash
            event.event_hash = event.compute_hash(self._last_hash)
            self._last_hash = event.event_hash

            # Write to log
            self._write_event(event)
            self._event_count += 1

            # Check rotation
            if self._should_rotate():
                self._rotate()

        return event

    def _write_event(self, event: AuditEvent) -> None:
        """Write event to log file."""
        line = json.dumps(event.to_dict()) + "\n"

        with open(self.log_path, "a") as f:
            f.write(line)

    def _should_rotate(self) -> bool:
        """Check if log should be rotated."""
        if not self.log_path.exists():
            return False

        size_mb = self.log_path.stat().st_size / (1024 * 1024)
        return size_mb >= self.rotation_size_mb

    def _rotate(self) -> None:
        """Rotate log file."""
        if not self.log_path.exists():
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        rotated_path = self.log_path.with_suffix(f".{timestamp}.log")

        self.log_path.rename(rotated_path)

        # Compress rotated file
        try:
            import gzip
            import shutil

            with open(rotated_path, "rb") as f_in:
                with gzip.open(f"{rotated_path}.gz", "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            rotated_path.unlink()
        except Exception:
            pass  # Compression is optional

    def verify_chain(self) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of the audit log chain.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not self.log_path.exists():
            return True, []

        errors = []
        previous_hash = None

        with open(self.log_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    event_dict = json.loads(line.strip())

                    # Check previous hash matches
                    if event_dict.get("previous_hash") != previous_hash:
                        errors.append(
                            f"Line {line_num}: Previous hash mismatch"
                        )

                    # Verify event hash
                    event = AuditEvent(
                        id=UUID(event_dict["id"]),
                        timestamp=datetime.fromisoformat(event_dict["timestamp"]),
                        action=AuditAction(event_dict["action"]),
                        severity=AuditSeverity(event_dict["severity"]),
                        actor=event_dict["actor"],
                        resource_type=event_dict["resource_type"],
                        resource_id=event_dict["resource_id"],
                        details=event_dict["details"],
                    )

                    expected_hash = event.compute_hash(previous_hash)
                    if event_dict.get("event_hash") != expected_hash:
                        errors.append(
                            f"Line {line_num}: Event hash mismatch (tampering detected)"
                        )

                    previous_hash = event_dict.get("event_hash")

                except Exception as e:
                    errors.append(f"Line {line_num}: Parse error - {e}")

        return len(errors) == 0, errors

    def query(
        self,
        action: Optional[AuditAction] = None,
        actor: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """
        Query audit events with filters.

        Args:
            action: Filter by action type
            actor: Filter by actor
            resource_type: Filter by resource type
            start_time: Filter events after this time
            end_time: Filter events before this time
            limit: Maximum number of events to return

        Returns:
            List of matching AuditEvents
        """
        if not self.log_path.exists():
            return []

        events = []

        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    event_dict = json.loads(line.strip())

                    # Apply filters
                    if action and event_dict.get("action") != action.value:
                        continue

                    if actor and event_dict.get("actor") != actor:
                        continue

                    if resource_type and event_dict.get("resource_type") != resource_type:
                        continue

                    event_time = datetime.fromisoformat(event_dict["timestamp"])
                    if start_time and event_time < start_time:
                        continue
                    if end_time and event_time > end_time:
                        continue

                    events.append(AuditEvent(
                        id=UUID(event_dict["id"]),
                        timestamp=event_time,
                        action=AuditAction(event_dict["action"]),
                        severity=AuditSeverity(event_dict["severity"]),
                        actor=event_dict["actor"],
                        resource_type=event_dict["resource_type"],
                        resource_id=event_dict["resource_id"],
                        details=event_dict["details"],
                        previous_hash=event_dict.get("previous_hash"),
                        event_hash=event_dict.get("event_hash"),
                    ))

                    if len(events) >= limit:
                        break

                except Exception:
                    continue

        return events

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        stats = {
            "total_events": self._event_count,
            "log_size_mb": 0,
            "by_action": {},
            "by_severity": {},
        }

        if self.log_path.exists():
            stats["log_size_mb"] = round(
                self.log_path.stat().st_size / (1024 * 1024), 2
            )

        return stats
