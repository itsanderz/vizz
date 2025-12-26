"""
Compliance Module

Support for enterprise compliance requirements:
- GDPR (data privacy, right to deletion)
- HIPAA (healthcare data protection)
- SOC 2 (security controls)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class ComplianceStandard(str, Enum):
    """Supported compliance standards."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"


class PIIType(str, Enum):
    """Types of Personally Identifiable Information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"


@dataclass
class PIIDetection:
    """A detected PII instance."""
    pii_type: PIIType
    location: str  # Field or path where found
    value_preview: str  # Redacted preview
    confidence: float


@dataclass
class ComplianceViolation:
    """A compliance violation."""
    standard: ComplianceStandard
    rule_id: str
    description: str
    severity: str  # low, medium, high, critical
    location: str
    remediation: str


class PIIDetector:
    """
    Detects PII in data for compliance checking.

    Uses pattern matching and heuristics to identify
    potentially sensitive data.
    """

    # Regex patterns for common PII
    PATTERNS = {
        PIIType.EMAIL: re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ),
        PIIType.PHONE: re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
        ),
        PIIType.SSN: re.compile(
            r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
        ),
        PIIType.CREDIT_CARD: re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
        ),
        PIIType.IP_ADDRESS: re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
    }

    def scan_text(self, text: str, location: str = "") -> List[PIIDetection]:
        """Scan text for PII."""
        detections = []

        for pii_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                value = match.group()
                # Redact the value for the preview
                redacted = self._redact(value, pii_type)

                detections.append(PIIDetection(
                    pii_type=pii_type,
                    location=location,
                    value_preview=redacted,
                    confidence=0.9,
                ))

        return detections

    def scan_dict(
        self,
        data: Dict[str, Any],
        path: str = "",
    ) -> List[PIIDetection]:
        """Recursively scan a dictionary for PII."""
        detections = []

        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, str):
                detections.extend(self.scan_text(value, current_path))
            elif isinstance(value, dict):
                detections.extend(self.scan_dict(value, current_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        detections.extend(
                            self.scan_text(item, f"{current_path}[{i}]")
                        )
                    elif isinstance(item, dict):
                        detections.extend(
                            self.scan_dict(item, f"{current_path}[{i}]")
                        )

        return detections

    def _redact(self, value: str, pii_type: PIIType) -> str:
        """Redact a PII value for safe display."""
        if len(value) <= 4:
            return "****"

        if pii_type == PIIType.EMAIL:
            parts = value.split("@")
            return f"{parts[0][:2]}***@{parts[1]}" if len(parts) == 2 else "***"

        if pii_type == PIIType.CREDIT_CARD:
            return f"****-****-****-{value[-4:]}"

        if pii_type == PIIType.SSN:
            return f"***-**-{value[-4:]}"

        # Default: show first 2 and last 2 characters
        return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


class ComplianceChecker:
    """
    Checks data and configurations against compliance standards.

    Supports GDPR, HIPAA, and SOC 2 requirements.
    """

    def __init__(
        self,
        standards: Optional[List[ComplianceStandard]] = None,
        pii_detection: bool = True,
    ):
        self.standards = standards or []
        self.pii_detection = pii_detection
        self.pii_detector = PIIDetector() if pii_detection else None

    def check_run_config(
        self,
        config: Dict[str, Any],
    ) -> List[ComplianceViolation]:
        """Check run configuration for compliance issues."""
        violations = []

        # Check for PII in config
        if self.pii_detector:
            pii_found = self.pii_detector.scan_dict(config)
            for detection in pii_found:
                violations.append(ComplianceViolation(
                    standard=ComplianceStandard.GDPR,
                    rule_id="GDPR-PII-001",
                    description=f"PII detected in configuration: {detection.pii_type.value}",
                    severity="high",
                    location=detection.location,
                    remediation="Remove or anonymize PII before logging",
                ))

        # HIPAA-specific checks
        if ComplianceStandard.HIPAA in self.standards:
            phi_keywords = ["patient", "diagnosis", "medical", "health", "ssn"]
            for key in config.keys():
                if any(kw in key.lower() for kw in phi_keywords):
                    violations.append(ComplianceViolation(
                        standard=ComplianceStandard.HIPAA,
                        rule_id="HIPAA-PHI-001",
                        description=f"Potential PHI field name: {key}",
                        severity="high",
                        location=key,
                        remediation="Ensure PHI is properly encrypted and access-controlled",
                    ))

        return violations

    def check_data_retention(
        self,
        retention_days: int,
    ) -> List[ComplianceViolation]:
        """Check data retention settings."""
        violations = []

        # GDPR: Purpose limitation
        if ComplianceStandard.GDPR in self.standards:
            if retention_days > 365 * 3:  # 3 years
                violations.append(ComplianceViolation(
                    standard=ComplianceStandard.GDPR,
                    rule_id="GDPR-RET-001",
                    description=f"Data retention ({retention_days} days) may exceed necessity",
                    severity="medium",
                    location="data_retention_days",
                    remediation="Review retention period for proportionality",
                ))

        return violations

    def check_encryption(
        self,
        encryption_enabled: bool,
        encryption_algorithm: Optional[str] = None,
    ) -> List[ComplianceViolation]:
        """Check encryption settings."""
        violations = []

        if not encryption_enabled:
            if ComplianceStandard.HIPAA in self.standards:
                violations.append(ComplianceViolation(
                    standard=ComplianceStandard.HIPAA,
                    rule_id="HIPAA-ENC-001",
                    description="Encryption at rest is required for PHI",
                    severity="critical",
                    location="encryption",
                    remediation="Enable AES-256 encryption",
                ))

            if ComplianceStandard.SOC2 in self.standards:
                violations.append(ComplianceViolation(
                    standard=ComplianceStandard.SOC2,
                    rule_id="SOC2-CC6.1",
                    description="Data encryption is recommended",
                    severity="medium",
                    location="encryption",
                    remediation="Enable encryption for sensitive data",
                ))

        elif encryption_algorithm:
            # Check algorithm strength
            weak_algorithms = ["des", "3des", "rc4", "md5"]
            if any(alg in encryption_algorithm.lower() for alg in weak_algorithms):
                violations.append(ComplianceViolation(
                    standard=ComplianceStandard.SOC2,
                    rule_id="SOC2-CC6.7",
                    description=f"Weak encryption algorithm: {encryption_algorithm}",
                    severity="high",
                    location="encryption_algorithm",
                    remediation="Use AES-256-GCM or equivalent",
                ))

        return violations

    def check_audit_logging(
        self,
        audit_enabled: bool,
        log_retention_days: int,
    ) -> List[ComplianceViolation]:
        """Check audit logging configuration."""
        violations = []

        if not audit_enabled:
            if ComplianceStandard.SOC2 in self.standards:
                violations.append(ComplianceViolation(
                    standard=ComplianceStandard.SOC2,
                    rule_id="SOC2-CC7.2",
                    description="Audit logging is required",
                    severity="high",
                    location="audit_logging",
                    remediation="Enable audit logging",
                ))

            if ComplianceStandard.HIPAA in self.standards:
                violations.append(ComplianceViolation(
                    standard=ComplianceStandard.HIPAA,
                    rule_id="HIPAA-AUD-001",
                    description="Audit controls required for PHI access",
                    severity="critical",
                    location="audit_logging",
                    remediation="Enable comprehensive audit logging",
                ))

        if audit_enabled and log_retention_days < 90:
            if ComplianceStandard.HIPAA in self.standards:
                violations.append(ComplianceViolation(
                    standard=ComplianceStandard.HIPAA,
                    rule_id="HIPAA-AUD-002",
                    description=f"Audit log retention ({log_retention_days} days) below minimum",
                    severity="high",
                    location="audit_log_retention",
                    remediation="Retain audit logs for at least 6 years (HIPAA)",
                ))

        return violations

    def generate_compliance_report(
        self,
        violations: List[ComplianceViolation],
    ) -> str:
        """Generate a compliance report."""
        report = ["# Atlas Compliance Report", ""]
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append(f"Standards Checked: {', '.join(s.value for s in self.standards)}")
        report.append("")

        if not violations:
            report.append("## Status: COMPLIANT")
            report.append("No violations detected.")
        else:
            # Group by severity
            by_severity = {"critical": [], "high": [], "medium": [], "low": []}
            for v in violations:
                by_severity[v.severity].append(v)

            report.append(f"## Status: {len(violations)} VIOLATIONS FOUND")
            report.append("")

            for severity in ["critical", "high", "medium", "low"]:
                if by_severity[severity]:
                    report.append(f"### {severity.upper()} ({len(by_severity[severity])})")
                    for v in by_severity[severity]:
                        report.append(f"- **{v.rule_id}**: {v.description}")
                        report.append(f"  - Location: `{v.location}`")
                        report.append(f"  - Remediation: {v.remediation}")
                    report.append("")

        return "\n".join(report)


def check_compliance(
    config: Dict[str, Any],
    standards: List[ComplianceStandard],
) -> Tuple[bool, str]:
    """
    Convenience function to check compliance.

    Returns:
        Tuple of (is_compliant, report)
    """
    checker = ComplianceChecker(standards=standards)

    all_violations = []
    all_violations.extend(checker.check_run_config(config))
    all_violations.extend(
        checker.check_encryption(
            config.get("encryption_enabled", False),
            config.get("encryption_algorithm"),
        )
    )
    all_violations.extend(
        checker.check_audit_logging(
            config.get("audit_enabled", False),
            config.get("audit_retention_days", 90),
        )
    )

    report = checker.generate_compliance_report(all_violations)
    is_compliant = len(all_violations) == 0

    return is_compliant, report


# Import fix
from typing import Tuple
