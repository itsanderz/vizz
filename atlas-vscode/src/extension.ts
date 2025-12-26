/**
 * Atlas VS Code Extension
 *
 * Main entry point for the Sovereign Experiment Engine visualization.
 */

import * as vscode from "vscode";
import * as path from "path";
import { AtlasStorage } from "./storage/AtlasStorage";
import { RunsTreeProvider } from "./views/RunsTreeProvider";
import { LiveMetricsProvider } from "./views/LiveMetricsProvider";
import { InsightsProvider } from "./views/InsightsProvider";
import { DashboardPanel } from "./webview/DashboardPanel";
import { TensorSurgeonPanel } from "./webview/TensorSurgeonPanel";
import { FileWatcher } from "./watcher/FileWatcher";

let storage: AtlasStorage | undefined;
let fileWatcher: FileWatcher | undefined;

export async function activate(context: vscode.ExtensionContext) {
  console.log("Atlas extension activating...");

  // Find .atlas directory in workspace
  const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
  if (!workspaceFolder) {
    console.log("No workspace folder found");
    return;
  }

  const atlasPath = path.join(workspaceFolder.uri.fsPath, ".atlas");

  try {
    // Initialize storage connection
    storage = new AtlasStorage(atlasPath);
    await storage.initialize();

    // Initialize file watcher for live updates
    fileWatcher = new FileWatcher(atlasPath);

    // Register tree view providers
    const runsProvider = new RunsTreeProvider(storage);
    const liveMetricsProvider = new LiveMetricsProvider(storage);
    const insightsProvider = new InsightsProvider(storage);

    context.subscriptions.push(
      vscode.window.registerTreeDataProvider("atlas.runs", runsProvider),
      vscode.window.registerTreeDataProvider(
        "atlas.liveMetrics",
        liveMetricsProvider
      ),
      vscode.window.registerTreeDataProvider("atlas.insights", insightsProvider)
    );

    // Wire up file watcher to refresh views
    fileWatcher.onMetricsChanged(() => {
      liveMetricsProvider.refresh();
    });

    fileWatcher.onRunsChanged(() => {
      runsProvider.refresh();
    });

    fileWatcher.start();

    // Register commands
    context.subscriptions.push(
      vscode.commands.registerCommand("atlas.openDashboard", () => {
        DashboardPanel.createOrShow(context.extensionUri, storage!);
      }),

      vscode.commands.registerCommand("atlas.openTensorSurgeon", () => {
        TensorSurgeonPanel.createOrShow(context.extensionUri, storage!);
      }),

      vscode.commands.registerCommand("atlas.compareRuns", async () => {
        const runs = await storage!.listRuns();
        const items = runs.map((r) => ({
          label: r.name,
          description: r.id,
          run: r,
        }));

        const selected = await vscode.window.showQuickPick(items, {
          canPickMany: true,
          placeHolder: "Select runs to compare",
        });

        if (selected && selected.length >= 2) {
          DashboardPanel.createOrShow(context.extensionUri, storage!);
          DashboardPanel.currentPanel?.compareRuns(
            selected.map((s) => s.run.id)
          );
        }
      }),

      vscode.commands.registerCommand("atlas.generateReport", async () => {
        const runs = await storage!.listRuns();
        const items = runs.map((r) => ({
          label: r.name,
          description: r.status,
          run: r,
        }));

        const selected = await vscode.window.showQuickPick(items, {
          placeHolder: "Select run to generate report for",
        });

        if (selected) {
          vscode.window.withProgress(
            {
              location: vscode.ProgressLocation.Notification,
              title: "Generating report...",
            },
            async () => {
              const report = await storage!.generateReport(selected.run.id);
              const doc = await vscode.workspace.openTextDocument({
                content: report,
                language: "markdown",
              });
              await vscode.window.showTextDocument(doc);
            }
          );
        }
      }),

      vscode.commands.registerCommand("atlas.promptToPlot", async () => {
        const prompt = await vscode.window.showInputBox({
          prompt: "Describe the visualization you want",
          placeHolder:
            'e.g., "Compare validation loss vs learning rate for the last 5 runs"',
        });

        if (prompt) {
          DashboardPanel.createOrShow(context.extensionUri, storage!);
          DashboardPanel.currentPanel?.executePromptToPlot(prompt);
        }
      }),

      // Refresh command
      vscode.commands.registerCommand("atlas.refresh", () => {
        runsProvider.refresh();
        liveMetricsProvider.refresh();
        insightsProvider.refresh();
      })
    );

    // Status bar item
    const statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Left,
      100
    );
    statusBarItem.text = "$(beaker) Atlas";
    statusBarItem.command = "atlas.openDashboard";
    statusBarItem.tooltip = "Open Atlas Dashboard";
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    console.log("Atlas extension activated successfully");
  } catch (error) {
    console.error("Failed to initialize Atlas:", error);
    vscode.window.showErrorMessage(
      `Atlas initialization failed: ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }
}

export function deactivate() {
  if (fileWatcher) {
    fileWatcher.stop();
  }
  if (storage) {
    storage.close();
  }
}
