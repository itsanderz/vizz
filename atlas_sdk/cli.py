"""
Atlas CLI

Command-line interface for Atlas experiment management.

Usage:
    atlas runs list
    atlas runs show <run_id>
    atlas runs compare <run_id_1> <run_id_2>
    atlas runs export <run_id> --output ./export
    atlas metrics list <run_id>
    atlas metrics query "SELECT * FROM metrics WHERE name = 'loss'"
    atlas insights list <run_id>
    atlas report generate <run_id>
    atlas config show
    atlas compliance check
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Atlas - The Sovereign Experiment Engine CLI",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=None,
        help="Path to .atlas directory",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Runs commands
    runs_parser = subparsers.add_parser("runs", help="Manage runs")
    runs_sub = runs_parser.add_subparsers(dest="runs_command")

    # runs list
    runs_list = runs_sub.add_parser("list", help="List runs")
    runs_list.add_argument("--project", help="Filter by project")
    runs_list.add_argument("--status", help="Filter by status")
    runs_list.add_argument("--limit", type=int, default=20)
    runs_list.add_argument("--tag", action="append", help="Filter by tag")

    # runs show
    runs_show = runs_sub.add_parser("show", help="Show run details")
    runs_show.add_argument("run_id", help="Run ID")

    # runs compare
    runs_compare = runs_sub.add_parser("compare", help="Compare runs")
    runs_compare.add_argument("run_ids", nargs="+", help="Run IDs to compare")
    runs_compare.add_argument("--metric", default="loss", help="Metric to compare")

    # runs export
    runs_export = runs_sub.add_parser("export", help="Export run data")
    runs_export.add_argument("run_id", help="Run ID")
    runs_export.add_argument("--output", "-o", type=Path, required=True)
    runs_export.add_argument("--include-artifacts", action="store_true")

    # runs delete
    runs_delete = runs_sub.add_parser("delete", help="Delete a run")
    runs_delete.add_argument("run_id", help="Run ID")
    runs_delete.add_argument("--force", "-f", action="store_true")

    # Metrics commands
    metrics_parser = subparsers.add_parser("metrics", help="Query metrics")
    metrics_sub = metrics_parser.add_subparsers(dest="metrics_command")

    # metrics list
    metrics_list = metrics_sub.add_parser("list", help="List metric names")
    metrics_list.add_argument("run_id", help="Run ID")

    # metrics get
    metrics_get = metrics_sub.add_parser("get", help="Get metric values")
    metrics_get.add_argument("run_id", help="Run ID")
    metrics_get.add_argument("metric_name", help="Metric name")
    metrics_get.add_argument("--start", type=int, help="Start step")
    metrics_get.add_argument("--end", type=int, help="End step")
    metrics_get.add_argument("--limit", type=int, default=100)

    # metrics query
    metrics_query = metrics_sub.add_parser("query", help="Run SQL query")
    metrics_query.add_argument("sql", help="SQL query")
    metrics_query.add_argument("--run-id", help="Run ID for context")

    # metrics stats
    metrics_stats = metrics_sub.add_parser("stats", help="Get metric statistics")
    metrics_stats.add_argument("run_id", help="Run ID")
    metrics_stats.add_argument("metric_name", help="Metric name")

    # Insights commands
    insights_parser = subparsers.add_parser("insights", help="AI insights")
    insights_sub = insights_parser.add_subparsers(dest="insights_command")

    insights_list = insights_sub.add_parser("list", help="List insights")
    insights_list.add_argument("run_id", help="Run ID")
    insights_list.add_argument("--severity", choices=["info", "warning", "critical"])

    # Report commands
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_sub = report_parser.add_subparsers(dest="report_command")

    report_gen = report_sub.add_parser("generate", help="Generate report")
    report_gen.add_argument("run_id", help="Run ID")
    report_gen.add_argument("--output", "-o", type=Path)
    report_gen.add_argument("--format", choices=["markdown", "html", "pdf"], default="markdown")

    # Config commands
    config_parser = subparsers.add_parser("config", help="Configuration")
    config_sub = config_parser.add_subparsers(dest="config_command")

    config_sub.add_parser("show", help="Show current configuration")
    config_sub.add_parser("init", help="Initialize .atlas directory")

    config_set = config_sub.add_parser("set", help="Set configuration value")
    config_set.add_argument("key", help="Configuration key")
    config_set.add_argument("value", help="Configuration value")

    # Compliance commands
    compliance_parser = subparsers.add_parser("compliance", help="Compliance checking")
    compliance_sub = compliance_parser.add_subparsers(dest="compliance_command")

    compliance_check = compliance_sub.add_parser("check", help="Run compliance check")
    compliance_check.add_argument(
        "--standard",
        action="append",
        choices=["gdpr", "hipaa", "soc2"],
        help="Standards to check",
    )

    compliance_sub.add_parser("report", help="Generate compliance report")

    # Audit commands
    audit_parser = subparsers.add_parser("audit", help="Audit log management")
    audit_sub = audit_parser.add_subparsers(dest="audit_command")

    audit_list = audit_sub.add_parser("list", help="List audit events")
    audit_list.add_argument("--action", help="Filter by action")
    audit_list.add_argument("--actor", help="Filter by actor")
    audit_list.add_argument("--limit", type=int, default=50)

    audit_sub.add_parser("verify", help="Verify audit log integrity")

    return parser


def format_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Format data as a table."""
    if not rows:
        return "No data"

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Build table
    lines = []

    # Header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-+-".join("-" * w for w in widths))

    # Rows
    for row in rows:
        row_line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(row_line)

    return "\n".join(lines)


def format_output(data: Any, fmt: str) -> str:
    """Format output in the specified format."""
    if fmt == "json":
        return json.dumps(data, indent=2, default=str)

    if fmt == "csv":
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                headers = list(data[0].keys())
                lines = [",".join(headers)]
                for row in data:
                    lines.append(",".join(str(row.get(h, "")) for h in headers))
                return "\n".join(lines)

    # Default: table format
    if isinstance(data, list) and data:
        if isinstance(data[0], dict):
            headers = list(data[0].keys())
            rows = [[row.get(h, "") for h in headers] for row in data]
            return format_table(headers, rows)

    return str(data)


def cmd_runs_list(args: argparse.Namespace, storage: Any) -> int:
    """List runs."""
    kwargs = {"limit": args.limit}
    if args.project:
        kwargs["project"] = args.project
    if args.status:
        kwargs["status"] = args.status

    runs = storage.list_runs(**kwargs)

    if args.format == "json":
        print(json.dumps(runs, indent=2, default=str))
    else:
        headers = ["ID", "Name", "Project", "Status", "Created"]
        rows = [
            [r["id"][:8], r["name"], r["project"], r["status"], r["created_at"][:19]]
            for r in runs
        ]
        print(format_table(headers, rows))

    return 0


def cmd_runs_show(args: argparse.Namespace, storage: Any) -> int:
    """Show run details."""
    run = storage.get_run(args.run_id)

    if not run:
        print(f"Run not found: {args.run_id}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(run, indent=2, default=str))
    else:
        print(f"Run: {run['name']}")
        print(f"  ID: {run['id']}")
        print(f"  Project: {run['project']}")
        print(f"  Status: {run['status']}")
        print(f"  Created: {run['created_at']}")
        if run.get("duration_seconds"):
            print(f"  Duration: {run['duration_seconds']:.1f}s")
        if run.get("tags"):
            print(f"  Tags: {', '.join(run['tags'])}")
        print(f"  Config:")
        for k, v in run.get("config", {}).items():
            print(f"    {k}: {v}")

    return 0


def cmd_metrics_list(args: argparse.Namespace, storage: Any) -> int:
    """List metric names."""
    names = storage.list_metrics(args.run_id)

    if args.format == "json":
        print(json.dumps(names))
    else:
        for name in names:
            print(name)

    return 0


def cmd_metrics_stats(args: argparse.Namespace, storage: Any) -> int:
    """Get metric statistics."""
    stats = storage.get_metric_statistics(args.run_id, args.metric_name)

    if args.format == "json":
        print(json.dumps(stats, indent=2))
    else:
        print(f"Statistics for {args.metric_name}:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

    return 0


def cmd_report_generate(args: argparse.Namespace, storage: Any) -> int:
    """Generate a report."""
    report = storage.generate_report(args.run_id)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)

    return 0


def cmd_compliance_check(args: argparse.Namespace, storage: Any) -> int:
    """Run compliance check."""
    from atlas_sdk.security.compliance import ComplianceChecker, ComplianceStandard

    standards = []
    if args.standard:
        for s in args.standard:
            standards.append(ComplianceStandard(s))

    checker = ComplianceChecker(standards=standards)

    # Check current configuration
    from atlas_sdk.config import get_config
    config = get_config()

    violations = []
    violations.extend(checker.check_run_config(config.model_dump()))
    violations.extend(checker.check_encryption(
        encryption_enabled=False,  # Would check actual setting
    ))
    violations.extend(checker.check_audit_logging(
        audit_enabled=config.enterprise.audit_logging_enabled,
        log_retention_days=config.enterprise.data_retention_days,
    ))

    report = checker.generate_compliance_report(violations)
    print(report)

    return 0 if not violations else 1


def cmd_audit_verify(args: argparse.Namespace, storage: Any) -> int:
    """Verify audit log integrity."""
    from atlas_sdk.security.audit import AuditLogger

    atlas_dir = args.atlas_dir or Path.cwd() / ".atlas"
    log_path = atlas_dir / "audit.log"

    if not log_path.exists():
        print("No audit log found")
        return 1

    logger = AuditLogger(log_path)
    is_valid, errors = logger.verify_chain()

    if is_valid:
        print("✓ Audit log integrity verified")
        return 0
    else:
        print("✗ Audit log integrity check FAILED")
        for error in errors:
            print(f"  - {error}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    # Initialize storage
    atlas_dir = args.atlas_dir or Path.cwd() / ".atlas"

    if not atlas_dir.exists() and args.command != "config":
        print(f"Atlas directory not found: {atlas_dir}", file=sys.stderr)
        print("Run 'atlas config init' to initialize.", file=sys.stderr)
        return 1

    try:
        from atlas_sdk.storage.engine import StorageEngine
        storage = StorageEngine(base_dir=atlas_dir.parent, read_only=True)
    except Exception as e:
        if args.command not in ("config", "compliance"):
            print(f"Failed to open Atlas storage: {e}", file=sys.stderr)
            return 1
        storage = None

    # Dispatch commands
    try:
        if args.command == "runs":
            if args.runs_command == "list":
                return cmd_runs_list(args, storage)
            elif args.runs_command == "show":
                return cmd_runs_show(args, storage)

        elif args.command == "metrics":
            if args.metrics_command == "list":
                return cmd_metrics_list(args, storage)
            elif args.metrics_command == "stats":
                return cmd_metrics_stats(args, storage)

        elif args.command == "report":
            if args.report_command == "generate":
                return cmd_report_generate(args, storage)

        elif args.command == "compliance":
            if args.compliance_command == "check":
                return cmd_compliance_check(args, storage)

        elif args.command == "audit":
            if args.audit_command == "verify":
                return cmd_audit_verify(args, storage)

        elif args.command == "config":
            if args.config_command == "show":
                from atlas_sdk.config import get_config
                config = get_config()
                print(json.dumps(config.model_dump(), indent=2, default=str))
                return 0

            elif args.config_command == "init":
                atlas_dir.mkdir(parents=True, exist_ok=True)
                print(f"Initialized Atlas directory at {atlas_dir}")
                return 0

        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    finally:
        if storage:
            storage.close()


if __name__ == "__main__":
    sys.exit(main())
