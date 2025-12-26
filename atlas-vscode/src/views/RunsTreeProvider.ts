/**
 * Tree view provider for displaying runs in the sidebar.
 */

import * as vscode from "vscode";
import { AtlasStorage, Run } from "../storage/AtlasStorage";

export class RunsTreeProvider
  implements vscode.TreeDataProvider<RunTreeItem | MetricTreeItem>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<
    RunTreeItem | MetricTreeItem | undefined
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  constructor(private storage: AtlasStorage) {}

  refresh(): void {
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: RunTreeItem | MetricTreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(
    element?: RunTreeItem | MetricTreeItem
  ): Promise<(RunTreeItem | MetricTreeItem)[]> {
    if (!element) {
      // Root level - show runs
      const runs = await this.storage.listRuns({ limit: 50 });
      return runs.map((run) => new RunTreeItem(run));
    }

    if (element instanceof RunTreeItem) {
      // Run level - show metrics
      const metricNames = await this.storage.listMetricNames(element.run.id);
      return metricNames.map(
        (name) => new MetricTreeItem(name, element.run.id)
      );
    }

    return [];
  }
}

class RunTreeItem extends vscode.TreeItem {
  constructor(public readonly run: Run) {
    super(run.name, vscode.TreeItemCollapsibleState.Collapsed);

    this.description = run.status;
    this.tooltip = `${run.name}\nStatus: ${run.status}\nCreated: ${run.created_at}`;
    this.contextValue = "run";

    // Set icon based on status
    switch (run.status) {
      case "running":
        this.iconPath = new vscode.ThemeIcon(
          "sync~spin",
          new vscode.ThemeColor("charts.green")
        );
        break;
      case "completed":
        this.iconPath = new vscode.ThemeIcon(
          "check",
          new vscode.ThemeColor("charts.green")
        );
        break;
      case "failed":
        this.iconPath = new vscode.ThemeIcon(
          "error",
          new vscode.ThemeColor("charts.red")
        );
        break;
      case "interrupted":
        this.iconPath = new vscode.ThemeIcon(
          "warning",
          new vscode.ThemeColor("charts.yellow")
        );
        break;
    }
  }
}

class MetricTreeItem extends vscode.TreeItem {
  constructor(
    public readonly metricName: string,
    public readonly runId: string
  ) {
    super(metricName, vscode.TreeItemCollapsibleState.None);

    this.description = "";
    this.tooltip = `Metric: ${metricName}`;
    this.contextValue = "metric";
    this.iconPath = new vscode.ThemeIcon("graph-line");

    // Command to view metric
    this.command = {
      command: "atlas.openDashboard",
      title: "View Metric",
      arguments: [{ runId, metricName }],
    };
  }
}
