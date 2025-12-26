/**
 * Tree view provider for displaying AI-generated insights.
 */

import * as vscode from "vscode";
import { AtlasStorage, Insight } from "../storage/AtlasStorage";

export class InsightsProvider
  implements vscode.TreeDataProvider<InsightTreeItem>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<
    InsightTreeItem | undefined
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  constructor(private storage: AtlasStorage) {}

  refresh(): void {
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: InsightTreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: InsightTreeItem): Promise<InsightTreeItem[]> {
    if (element) {
      return [];
    }

    // Get recent runs
    const runs = await this.storage.listRuns({ limit: 10 });
    const items: InsightTreeItem[] = [];

    for (const run of runs) {
      try {
        const insights = await this.storage.getInsights(run.id);

        for (const insight of insights.slice(0, 5)) {
          items.push(new InsightTreeItem(insight, run.name));
        }
      } catch (error) {
        // Skip runs without insights
      }
    }

    if (items.length === 0) {
      return [
        new InsightTreeItem(
          {
            id: "",
            run_id: "",
            insight_type: "placeholder",
            title: "No insights yet",
            description: "Insights will appear here as runs progress",
            severity: "info",
            related_metrics: [],
            timestamp: "",
            votes_up: 0,
            votes_down: 0,
          },
          "",
          true
        ),
      ];
    }

    // Sort by timestamp descending
    items.sort((a, b) => {
      const dateA = new Date(a.insight.timestamp).getTime();
      const dateB = new Date(b.insight.timestamp).getTime();
      return dateB - dateA;
    });

    return items.slice(0, 20);
  }
}

class InsightTreeItem extends vscode.TreeItem {
  constructor(
    public readonly insight: Insight,
    public readonly runName: string,
    private readonly isPlaceholder: boolean = false
  ) {
    super(insight.title, vscode.TreeItemCollapsibleState.None);

    if (!isPlaceholder) {
      this.description = runName;
      this.tooltip = new vscode.MarkdownString(
        `**${insight.title}**\n\n${insight.description}\n\n` +
          `*Run: ${runName}*\n\n` +
          `👍 ${insight.votes_up} | 👎 ${insight.votes_down}`
      );

      // Set icon based on severity
      switch (insight.severity) {
        case "critical":
          this.iconPath = new vscode.ThemeIcon(
            "error",
            new vscode.ThemeColor("errorForeground")
          );
          break;
        case "warning":
          this.iconPath = new vscode.ThemeIcon(
            "warning",
            new vscode.ThemeColor("editorWarning.foreground")
          );
          break;
        default:
          this.iconPath = new vscode.ThemeIcon(
            "lightbulb",
            new vscode.ThemeColor("editorInfo.foreground")
          );
      }

      // Click to view insight details
      this.command = {
        command: "atlas.openDashboard",
        title: "View Insight",
        arguments: [{ insightId: insight.id, runId: insight.run_id }],
      };
    } else {
      this.iconPath = new vscode.ThemeIcon("info");
    }
  }
}
