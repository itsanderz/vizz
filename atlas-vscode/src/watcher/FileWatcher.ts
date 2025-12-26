/**
 * File watcher for detecting changes to .atlas data files.
 *
 * Uses chokidar for efficient file system watching.
 */

import * as path from "path";
import * as fs from "fs";

type Callback = () => void;

export class FileWatcher {
  private watcher: any; // chokidar instance
  private atlasPath: string;
  private metricsCallbacks: Callback[] = [];
  private runsCallbacks: Callback[] = [];
  private insightsCallbacks: Callback[] = [];

  private debounceTimers: Map<string, NodeJS.Timeout> = new Map();
  private debounceMs: number = 100;

  constructor(atlasPath: string, debounceMs: number = 100) {
    this.atlasPath = atlasPath;
    this.debounceMs = debounceMs;
  }

  start(): void {
    if (!fs.existsSync(this.atlasPath)) {
      console.log(`Atlas directory not found: ${this.atlasPath}`);
      return;
    }

    try {
      const chokidar = require("chokidar");

      this.watcher = chokidar.watch(this.atlasPath, {
        persistent: true,
        ignoreInitial: true,
        depth: 2,
        awaitWriteFinish: {
          stabilityThreshold: 100,
          pollInterval: 50,
        },
      });

      this.watcher.on("add", (filePath: string) =>
        this.handleFileChange(filePath)
      );
      this.watcher.on("change", (filePath: string) =>
        this.handleFileChange(filePath)
      );

      console.log(`Watching for changes in: ${this.atlasPath}`);
    } catch (error) {
      console.error("Failed to start file watcher:", error);
    }
  }

  stop(): void {
    if (this.watcher) {
      this.watcher.close();
    }

    // Clear all debounce timers
    for (const timer of this.debounceTimers.values()) {
      clearTimeout(timer);
    }
    this.debounceTimers.clear();
  }

  private handleFileChange(filePath: string): void {
    const relativePath = path.relative(this.atlasPath, filePath);
    const ext = path.extname(filePath);

    // Determine the type of change
    let changeType: "metrics" | "runs" | "insights" | null = null;

    if (
      ext === ".parquet" ||
      ext === ".duckdb" ||
      relativePath.startsWith("metrics")
    ) {
      changeType = "metrics";
    } else if (
      relativePath === "atlas.db" ||
      relativePath === "atlas.db-wal"
    ) {
      changeType = "runs";
    } else if (relativePath.includes("insights")) {
      changeType = "insights";
    }

    if (changeType) {
      this.debounceNotify(changeType);
    }
  }

  private debounceNotify(changeType: string): void {
    // Clear existing timer for this type
    const existingTimer = this.debounceTimers.get(changeType);
    if (existingTimer) {
      clearTimeout(existingTimer);
    }

    // Set new timer
    const timer = setTimeout(() => {
      this.notify(changeType);
      this.debounceTimers.delete(changeType);
    }, this.debounceMs);

    this.debounceTimers.set(changeType, timer);
  }

  private notify(changeType: string): void {
    switch (changeType) {
      case "metrics":
        this.metricsCallbacks.forEach((cb) => {
          try {
            cb();
          } catch (e) {
            console.error("Error in metrics callback:", e);
          }
        });
        break;
      case "runs":
        this.runsCallbacks.forEach((cb) => {
          try {
            cb();
          } catch (e) {
            console.error("Error in runs callback:", e);
          }
        });
        break;
      case "insights":
        this.insightsCallbacks.forEach((cb) => {
          try {
            cb();
          } catch (e) {
            console.error("Error in insights callback:", e);
          }
        });
        break;
    }
  }

  onMetricsChanged(callback: Callback): void {
    this.metricsCallbacks.push(callback);
  }

  onRunsChanged(callback: Callback): void {
    this.runsCallbacks.push(callback);
  }

  onInsightsChanged(callback: Callback): void {
    this.insightsCallbacks.push(callback);
  }
}
