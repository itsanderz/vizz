/**
 * Tree view provider for displaying live metrics from active runs.
 */

import * as vscode from "vscode";
import { AtlasStorage, Metric } from "../storage/AtlasStorage";

export class LiveMetricsProvider
  implements vscode.TreeDataProvider<LiveMetricItem>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<
    LiveMetricItem | undefined
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private cachedMetrics: Map<string, { value: number; step: number }> =
    new Map();

  constructor(private storage: AtlasStorage) {}

  refresh(): void {
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: LiveMetricItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: LiveMetricItem): Promise<LiveMetricItem[]> {
    if (element) {
      return [];
    }

    // Get active (running) runs
    const runs = await this.storage.listRuns({ status: "running", limit: 10 });

    const items: LiveMetricItem[] = [];

    for (const run of runs) {
      const metricNames = await this.storage.listMetricNames(run.id);

      // Get latest value for key metrics
      for (const name of metricNames.slice(0, 5)) {
        try {
          const metrics = await this.storage.getMetrics(run.id, name, {
            limit: 1,
          });

          if (metrics.length > 0) {
            const latest = metrics[0];
            const previous = this.cachedMetrics.get(`${run.id}:${name}`);

            let trend: "up" | "down" | "flat" = "flat";
            if (previous) {
              if (latest.value > previous.value) {
                trend = "up";
              } else if (latest.value < previous.value) {
                trend = "down";
              }
            }

            this.cachedMetrics.set(`${run.id}:${name}`, {
              value: latest.value,
              step: latest.step,
            });

            items.push(
              new LiveMetricItem(run.name, name, latest.value, latest.step, trend)
            );
          }
        } catch (error) {
          // Skip metrics that can't be read
        }
      }
    }

    if (items.length === 0) {
      return [new LiveMetricItem("No active runs", "", 0, 0, "flat", true)];
    }

    return items;
  }
}

class LiveMetricItem extends vscode.TreeItem {
  constructor(
    public readonly runName: string,
    public readonly metricName: string,
    public readonly value: number,
    public readonly step: number,
    public readonly trend: "up" | "down" | "flat",
    private readonly isPlaceholder: boolean = false
  ) {
    super(
      isPlaceholder ? runName : `${metricName}`,
      vscode.TreeItemCollapsibleState.None
    );

    if (!isPlaceholder) {
      const formattedValue =
        Math.abs(value) < 0.001 || Math.abs(value) > 10000
          ? value.toExponential(3)
          : value.toFixed(4);

      this.description = `${formattedValue} @ step ${step}`;

      // Trend indicator
      let trendIcon = "";
      let trendColor = "";
      switch (trend) {
        case "up":
          trendIcon = "arrow-up";
          trendColor = "charts.red"; // Usually loss going up is bad
          break;
        case "down":
          trendIcon = "arrow-down";
          trendColor = "charts.green";
          break;
        default:
          trendIcon = "dash";
          trendColor = "charts.gray";
      }

      this.iconPath = new vscode.ThemeIcon(
        trendIcon,
        new vscode.ThemeColor(trendColor)
      );

      this.tooltip = `${runName}\n${metricName}: ${formattedValue}\nStep: ${step}\nTrend: ${trend}`;
    } else {
      this.iconPath = new vscode.ThemeIcon("info");
    }
  }
}
