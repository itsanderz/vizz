/**
 * Atlas Storage Interface
 *
 * Connects to SQLite (metadata) and DuckDB (time series) for data access.
 */

import * as path from "path";
import * as fs from "fs";

// Types
export interface Run {
  id: string;
  name: string;
  project: string;
  status: "running" | "completed" | "failed" | "interrupted";
  config: Record<string, unknown>;
  tags: string[];
  notes?: string;
  created_at: string;
  updated_at: string;
  finished_at?: string;
  duration_seconds?: number;
}

export interface Metric {
  name: string;
  value: number;
  step: number;
  timestamp: string;
  metric_type: string;
  metadata?: Record<string, unknown>;
}

export interface Insight {
  id: string;
  run_id: string;
  insight_type: string;
  title: string;
  description: string;
  severity: "info" | "warning" | "critical";
  related_metrics: string[];
  visualization_spec?: Record<string, unknown>;
  timestamp: string;
  votes_up: number;
  votes_down: number;
}

export interface Artifact {
  id: string;
  run_id: string;
  name: string;
  artifact_type: string;
  path: string;
  size_bytes: number;
  checksum?: string;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export class AtlasStorage {
  private atlasPath: string;
  private sqliteDb: any; // better-sqlite3 instance
  private duckDb: any; // duckdb instance

  constructor(atlasPath: string) {
    this.atlasPath = atlasPath;
  }

  async initialize(): Promise<void> {
    const dbPath = path.join(this.atlasPath, "atlas.db");

    if (!fs.existsSync(dbPath)) {
      throw new Error(`Atlas database not found at ${dbPath}`);
    }

    // Initialize SQLite connection
    try {
      const Database = require("better-sqlite3");
      this.sqliteDb = new Database(dbPath, { readonly: true });
      this.sqliteDb.pragma("journal_mode = WAL");
    } catch (error) {
      console.error("Failed to initialize SQLite:", error);
      throw error;
    }

    // Initialize DuckDB connection
    try {
      const duckdb = require("duckdb");
      this.duckDb = new duckdb.Database(":memory:", { access_mode: "READ_ONLY" });
    } catch (error) {
      console.error("Failed to initialize DuckDB:", error);
      // DuckDB is optional for basic functionality
    }
  }

  // ==================== Run Operations ====================

  async listRuns(options?: {
    project?: string;
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<Run[]> {
    let query = "SELECT * FROM runs WHERE 1=1";
    const params: any[] = [];

    if (options?.project) {
      query += " AND project = ?";
      params.push(options.project);
    }

    if (options?.status) {
      query += " AND status = ?";
      params.push(options.status);
    }

    query += " ORDER BY created_at DESC";

    if (options?.limit) {
      query += ` LIMIT ${options.limit}`;
    }

    if (options?.offset) {
      query += ` OFFSET ${options.offset}`;
    }

    const rows = this.sqliteDb.prepare(query).all(...params);
    return rows.map((row: any) => this.parseRun(row));
  }

  async getRun(runId: string): Promise<Run | null> {
    const row = this.sqliteDb
      .prepare("SELECT * FROM runs WHERE id = ?")
      .get(runId);

    if (!row) {
      return null;
    }

    const run = this.parseRun(row);

    // Get tags
    const tags = this.sqliteDb
      .prepare("SELECT tag FROM run_tags WHERE run_id = ?")
      .all(runId);
    run.tags = tags.map((t: any) => t.tag);

    return run;
  }

  private parseRun(row: any): Run {
    return {
      id: row.id,
      name: row.name,
      project: row.project,
      status: row.status,
      config: row.config ? JSON.parse(row.config) : {},
      tags: [],
      notes: row.notes,
      created_at: row.created_at,
      updated_at: row.updated_at,
      finished_at: row.finished_at,
      duration_seconds: row.duration_seconds,
    };
  }

  // ==================== Metrics Operations ====================

  async getMetrics(
    runId: string,
    metricName: string,
    options?: {
      startStep?: number;
      endStep?: number;
      limit?: number;
    }
  ): Promise<Metric[]> {
    // Try to read from DuckDB file
    const duckDbPath = path.join(this.atlasPath, `metrics_${runId}.duckdb`);

    if (fs.existsSync(duckDbPath) && this.duckDb) {
      return this.getMetricsFromDuckDB(duckDbPath, metricName, options);
    }

    // Fallback to reading Parquet files
    return this.getMetricsFromParquet(runId, metricName, options);
  }

  private async getMetricsFromDuckDB(
    dbPath: string,
    metricName: string,
    options?: {
      startStep?: number;
      endStep?: number;
      limit?: number;
    }
  ): Promise<Metric[]> {
    return new Promise((resolve, reject) => {
      const duckdb = require("duckdb");
      const db = new duckdb.Database(dbPath, { access_mode: "READ_ONLY" });

      let query = `SELECT * FROM metrics WHERE name = '${metricName}'`;

      if (options?.startStep !== undefined) {
        query += ` AND step >= ${options.startStep}`;
      }
      if (options?.endStep !== undefined) {
        query += ` AND step <= ${options.endStep}`;
      }

      query += " ORDER BY step";

      if (options?.limit) {
        query += ` LIMIT ${options.limit}`;
      }

      db.all(query, (err: any, rows: any[]) => {
        db.close();
        if (err) {
          reject(err);
        } else {
          resolve(rows || []);
        }
      });
    });
  }

  private async getMetricsFromParquet(
    runId: string,
    metricName: string,
    options?: {
      startStep?: number;
      endStep?: number;
      limit?: number;
    }
  ): Promise<Metric[]> {
    // Read Parquet files using DuckDB's parquet reader
    const metricsDir = path.join(this.atlasPath, "metrics");
    const pattern = `${metricsDir}/${runId}_*.parquet`;

    return new Promise((resolve, reject) => {
      if (!this.duckDb) {
        resolve([]);
        return;
      }

      let query = `
        SELECT * FROM read_parquet('${pattern}')
        WHERE name = '${metricName}'
      `;

      if (options?.startStep !== undefined) {
        query += ` AND step >= ${options.startStep}`;
      }
      if (options?.endStep !== undefined) {
        query += ` AND step <= ${options.endStep}`;
      }

      query += " ORDER BY step";

      if (options?.limit) {
        query += ` LIMIT ${options.limit}`;
      }

      this.duckDb.all(query, (err: any, rows: any[]) => {
        if (err) {
          // Parquet files might not exist yet
          resolve([]);
        } else {
          resolve(rows || []);
        }
      });
    });
  }

  async listMetricNames(runId: string): Promise<string[]> {
    const duckDbPath = path.join(this.atlasPath, `metrics_${runId}.duckdb`);

    return new Promise((resolve, reject) => {
      if (!fs.existsSync(duckDbPath)) {
        resolve([]);
        return;
      }

      const duckdb = require("duckdb");
      const db = new duckdb.Database(duckDbPath, { access_mode: "READ_ONLY" });

      db.all(
        "SELECT DISTINCT name FROM metrics ORDER BY name",
        (err: any, rows: any[]) => {
          db.close();
          if (err) {
            resolve([]);
          } else {
            resolve(rows?.map((r) => r.name) || []);
          }
        }
      );
    });
  }

  async getMetricStatistics(
    runId: string,
    metricName: string
  ): Promise<{
    count: number;
    min: number;
    max: number;
    mean: number;
    std: number;
  }> {
    const duckDbPath = path.join(this.atlasPath, `metrics_${runId}.duckdb`);

    return new Promise((resolve, reject) => {
      if (!fs.existsSync(duckDbPath)) {
        resolve({ count: 0, min: 0, max: 0, mean: 0, std: 0 });
        return;
      }

      const duckdb = require("duckdb");
      const db = new duckdb.Database(duckDbPath, { access_mode: "READ_ONLY" });

      const query = `
        SELECT
          COUNT(*) as count,
          MIN(value) as min,
          MAX(value) as max,
          AVG(value) as mean,
          STDDEV(value) as std
        FROM metrics
        WHERE name = '${metricName}'
      `;

      db.all(query, (err: any, rows: any[]) => {
        db.close();
        if (err || !rows?.length) {
          resolve({ count: 0, min: 0, max: 0, mean: 0, std: 0 });
        } else {
          resolve(rows[0]);
        }
      });
    });
  }

  // ==================== Insight Operations ====================

  async getInsights(
    runId: string,
    options?: { severity?: string }
  ): Promise<Insight[]> {
    let query = "SELECT * FROM insights WHERE run_id = ?";
    const params: any[] = [runId];

    if (options?.severity) {
      query += " AND severity = ?";
      params.push(options.severity);
    }

    query += " ORDER BY timestamp DESC";

    const rows = this.sqliteDb.prepare(query).all(...params);
    return rows.map((row: any) => ({
      id: row.id,
      run_id: row.run_id,
      insight_type: row.insight_type,
      title: row.title,
      description: row.description,
      severity: row.severity,
      related_metrics: row.related_metrics
        ? JSON.parse(row.related_metrics)
        : [],
      visualization_spec: row.visualization_spec
        ? JSON.parse(row.visualization_spec)
        : undefined,
      timestamp: row.timestamp,
      votes_up: row.votes_up,
      votes_down: row.votes_down,
    }));
  }

  async voteInsight(
    insightId: string,
    runId: string,
    vote: 1 | -1,
    feedback?: string
  ): Promise<void> {
    // Note: This would need a writable connection for actual voting
    console.log(`Vote recorded: ${insightId} = ${vote}`);
  }

  // ==================== Artifact Operations ====================

  async getArtifacts(runId: string): Promise<Artifact[]> {
    const rows = this.sqliteDb
      .prepare("SELECT * FROM artifacts WHERE run_id = ?")
      .all(runId);

    return rows.map((row: any) => ({
      id: row.id,
      run_id: row.run_id,
      name: row.name,
      artifact_type: row.artifact_type,
      path: row.path,
      size_bytes: row.size_bytes,
      checksum: row.checksum,
      metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
      created_at: row.created_at,
    }));
  }

  // ==================== Report Generation ====================

  async generateReport(runId: string): Promise<string> {
    const run = await this.getRun(runId);
    if (!run) {
      throw new Error(`Run ${runId} not found`);
    }

    const metricNames = await this.listMetricNames(runId);
    const insights = await this.getInsights(runId);

    let report = `# Atlas Run Report: ${run.name}\n\n`;
    report += `**ID:** ${run.id}\n`;
    report += `**Project:** ${run.project}\n`;
    report += `**Status:** ${run.status}\n`;
    report += `**Created:** ${run.created_at}\n`;

    if (run.duration_seconds) {
      report += `**Duration:** ${Math.round(run.duration_seconds)}s\n`;
    }

    if (run.tags.length > 0) {
      report += `**Tags:** ${run.tags.join(", ")}\n`;
    }

    report += "\n## Configuration\n\n```json\n";
    report += JSON.stringify(run.config, null, 2);
    report += "\n```\n\n";

    report += "## Metrics Summary\n\n";
    for (const name of metricNames.slice(0, 10)) {
      const stats = await this.getMetricStatistics(runId, name);
      report += `### ${name}\n`;
      report += `- Count: ${stats.count}\n`;
      report += `- Min: ${stats.min?.toFixed(4)}\n`;
      report += `- Max: ${stats.max?.toFixed(4)}\n`;
      report += `- Mean: ${stats.mean?.toFixed(4)}\n`;
      report += `- Std: ${stats.std?.toFixed(4)}\n\n`;
    }

    if (insights.length > 0) {
      report += "## AI Insights\n\n";
      for (const insight of insights) {
        const emoji =
          insight.severity === "critical"
            ? "🔴"
            : insight.severity === "warning"
              ? "🟡"
              : "🔵";
        report += `### ${emoji} ${insight.title}\n`;
        report += `${insight.description}\n\n`;
      }
    }

    return report;
  }

  // ==================== Comparison Operations ====================

  async compareRuns(
    runIds: string[],
    metricName: string
  ): Promise<Record<string, Metric[]>> {
    const result: Record<string, Metric[]> = {};

    for (const runId of runIds) {
      result[runId] = await this.getMetrics(runId, metricName);
    }

    return result;
  }

  async diffConfigs(
    runIdA: string,
    runIdB: string
  ): Promise<{
    differences: Record<string, { a: unknown; b: unknown }>;
    onlyInA: string[];
    onlyInB: string[];
  }> {
    const runA = await this.getRun(runIdA);
    const runB = await this.getRun(runIdB);

    if (!runA || !runB) {
      throw new Error("One or both runs not found");
    }

    const configA = runA.config;
    const configB = runB.config;

    const allKeys = new Set([
      ...Object.keys(configA),
      ...Object.keys(configB),
    ]);

    const differences: Record<string, { a: unknown; b: unknown }> = {};
    const onlyInA: string[] = [];
    const onlyInB: string[] = [];

    for (const key of allKeys) {
      const inA = key in configA;
      const inB = key in configB;

      if (inA && !inB) {
        onlyInA.push(key);
      } else if (!inA && inB) {
        onlyInB.push(key);
      } else if (JSON.stringify(configA[key]) !== JSON.stringify(configB[key])) {
        differences[key] = { a: configA[key], b: configB[key] };
      }
    }

    return { differences, onlyInA, onlyInB };
  }

  // ==================== Utility ====================

  close(): void {
    if (this.sqliteDb) {
      this.sqliteDb.close();
    }
    if (this.duckDb) {
      this.duckDb.close();
    }
  }
}
