/**
 * Dashboard Panel - Main visualization webview.
 *
 * Renders metrics using Vega-Lite for standard charts.
 */

import * as vscode from "vscode";
import { AtlasStorage, Run, Metric } from "../storage/AtlasStorage";

export class DashboardPanel {
  public static currentPanel: DashboardPanel | undefined;
  private static readonly viewType = "atlas.dashboard";

  private readonly panel: vscode.WebviewPanel;
  private readonly extensionUri: vscode.Uri;
  private storage: AtlasStorage;
  private disposables: vscode.Disposable[] = [];

  private currentRunId?: string;
  private refreshInterval?: NodeJS.Timeout;

  public static createOrShow(
    extensionUri: vscode.Uri,
    storage: AtlasStorage
  ): void {
    const column = vscode.window.activeTextEditor
      ? vscode.window.activeTextEditor.viewColumn
      : undefined;

    if (DashboardPanel.currentPanel) {
      DashboardPanel.currentPanel.panel.reveal(column);
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      DashboardPanel.viewType,
      "Atlas Dashboard",
      column || vscode.ViewColumn.One,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [extensionUri],
      }
    );

    DashboardPanel.currentPanel = new DashboardPanel(
      panel,
      extensionUri,
      storage
    );
  }

  private constructor(
    panel: vscode.WebviewPanel,
    extensionUri: vscode.Uri,
    storage: AtlasStorage
  ) {
    this.panel = panel;
    this.extensionUri = extensionUri;
    this.storage = storage;

    this.updateContent();
    this.startAutoRefresh();

    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);

    this.panel.webview.onDidReceiveMessage(
      (message) => this.handleMessage(message),
      null,
      this.disposables
    );
  }

  public async compareRuns(runIds: string[]): Promise<void> {
    this.panel.webview.postMessage({
      type: "compareRuns",
      runIds,
    });
  }

  public async executePromptToPlot(prompt: string): Promise<void> {
    // Generate Vega-Lite spec from prompt using VS Code's language model API
    const spec = await this.generateVegaSpecFromPrompt(prompt);

    this.panel.webview.postMessage({
      type: "renderChart",
      spec,
      prompt,
    });
  }

  private async generateVegaSpecFromPrompt(
    prompt: string
  ): Promise<Record<string, unknown>> {
    // Try to use VS Code's language model API
    try {
      const models = await vscode.lm.selectChatModels({
        vendor: "copilot",
        family: "gpt-4",
      });

      if (models.length > 0) {
        const model = models[0];

        const systemPrompt = `You are an expert at creating Vega-Lite visualizations for machine learning metrics.
Given a user's description, generate a valid Vega-Lite JSON specification.
The data will be provided as an array of objects with fields: name, value, step, timestamp.
Return ONLY valid JSON, no markdown or explanation.`;

        const messages = [
          vscode.LanguageModelChatMessage.User(
            `${systemPrompt}\n\nUser request: ${prompt}`
          ),
        ];

        const response = await model.sendRequest(messages, {});

        let result = "";
        for await (const chunk of response.text) {
          result += chunk;
        }

        // Parse the response as JSON
        return JSON.parse(result.trim());
      }
    } catch (error) {
      console.error("Language model error:", error);
    }

    // Fallback: generate a basic line chart spec
    return this.generateDefaultSpec(prompt);
  }

  private generateDefaultSpec(prompt: string): Record<string, unknown> {
    // Parse prompt for metric names
    const metricMatch = prompt.match(/(?:loss|accuracy|lr|learning.?rate)/i);
    const metricName = metricMatch ? metricMatch[0].toLowerCase() : "loss";

    return {
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      title: prompt,
      width: "container",
      height: 300,
      data: { name: "metrics" },
      mark: {
        type: "line",
        point: true,
      },
      encoding: {
        x: {
          field: "step",
          type: "quantitative",
          title: "Step",
        },
        y: {
          field: "value",
          type: "quantitative",
          title: metricName,
        },
        color: {
          field: "run_id",
          type: "nominal",
          legend: { title: "Run" },
        },
        tooltip: [
          { field: "run_id", type: "nominal", title: "Run" },
          { field: "step", type: "quantitative" },
          { field: "value", type: "quantitative", format: ".4f" },
        ],
      },
    };
  }

  private async handleMessage(message: any): Promise<void> {
    switch (message.type) {
      case "ready":
        await this.sendInitialData();
        break;

      case "selectRun":
        this.currentRunId = message.runId;
        await this.sendRunData(message.runId);
        break;

      case "getMetrics":
        await this.sendMetrics(
          message.runId,
          message.metricName,
          message.options
        );
        break;

      case "vote":
        await this.storage.voteInsight(
          message.insightId,
          message.runId,
          message.vote,
          message.feedback
        );
        break;

      case "refresh":
        await this.sendInitialData();
        break;
    }
  }

  private async sendInitialData(): Promise<void> {
    const runs = await this.storage.listRuns({ limit: 50 });

    this.panel.webview.postMessage({
      type: "init",
      runs,
    });

    // If there's a running run, select it
    const activeRun = runs.find((r) => r.status === "running");
    if (activeRun) {
      this.currentRunId = activeRun.id;
      await this.sendRunData(activeRun.id);
    } else if (runs.length > 0) {
      this.currentRunId = runs[0].id;
      await this.sendRunData(runs[0].id);
    }
  }

  private async sendRunData(runId: string): Promise<void> {
    const run = await this.storage.getRun(runId);
    const metricNames = await this.storage.listMetricNames(runId);
    const insights = await this.storage.getInsights(runId);

    // Get data for key metrics
    const metricsData: Record<string, Metric[]> = {};
    for (const name of metricNames.slice(0, 10)) {
      metricsData[name] = await this.storage.getMetrics(runId, name);
    }

    this.panel.webview.postMessage({
      type: "runData",
      run,
      metricNames,
      metricsData,
      insights,
    });
  }

  private async sendMetrics(
    runId: string,
    metricName: string,
    options?: any
  ): Promise<void> {
    const metrics = await this.storage.getMetrics(runId, metricName, options);

    this.panel.webview.postMessage({
      type: "metrics",
      runId,
      metricName,
      data: metrics,
    });
  }

  private startAutoRefresh(): void {
    const config = vscode.workspace.getConfiguration("atlas");
    const interval = config.get<number>("refreshInterval", 1000);

    this.refreshInterval = setInterval(async () => {
      if (this.currentRunId) {
        const run = await this.storage.getRun(this.currentRunId);
        if (run?.status === "running") {
          await this.sendRunData(this.currentRunId);
        }
      }
    }, interval);
  }

  private async updateContent(): Promise<void> {
    this.panel.webview.html = this.getHtmlContent();
  }

  private getHtmlContent(): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'unsafe-inline'; img-src data:;">
  <title>Atlas Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    :root {
      --bg-primary: var(--vscode-editor-background);
      --bg-secondary: var(--vscode-sideBar-background);
      --text-primary: var(--vscode-editor-foreground);
      --text-secondary: var(--vscode-descriptionForeground);
      --border-color: var(--vscode-panel-border);
      --accent-color: var(--vscode-button-background);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: var(--vscode-font-family);
      background: var(--bg-primary);
      color: var(--text-primary);
      padding: 16px;
    }

    .dashboard {
      display: grid;
      grid-template-columns: 200px 1fr 280px;
      grid-template-rows: auto 1fr;
      gap: 16px;
      height: calc(100vh - 32px);
    }

    .header {
      grid-column: 1 / -1;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border-color);
    }

    .header h1 {
      font-size: 1.5rem;
      font-weight: 600;
    }

    .header-actions {
      display: flex;
      gap: 8px;
    }

    .btn {
      background: var(--accent-color);
      color: var(--vscode-button-foreground);
      border: none;
      padding: 6px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
    }

    .btn:hover {
      opacity: 0.9;
    }

    .sidebar {
      background: var(--bg-secondary);
      border-radius: 8px;
      padding: 12px;
      overflow-y: auto;
    }

    .sidebar h3 {
      font-size: 12px;
      text-transform: uppercase;
      color: var(--text-secondary);
      margin-bottom: 8px;
    }

    .run-item {
      padding: 8px;
      border-radius: 4px;
      cursor: pointer;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .run-item:hover {
      background: var(--vscode-list-hoverBackground);
    }

    .run-item.active {
      background: var(--vscode-list-activeSelectionBackground);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    .status-running { background: #4caf50; }
    .status-completed { background: #2196f3; }
    .status-failed { background: #f44336; }
    .status-interrupted { background: #ff9800; }

    .main-content {
      display: flex;
      flex-direction: column;
      gap: 16px;
      overflow-y: auto;
    }

    .ticker {
      display: flex;
      gap: 16px;
      padding: 12px;
      background: var(--bg-secondary);
      border-radius: 8px;
    }

    .ticker-item {
      text-align: center;
    }

    .ticker-item .label {
      font-size: 10px;
      color: var(--text-secondary);
      text-transform: uppercase;
    }

    .ticker-item .value {
      font-size: 1.5rem;
      font-weight: 600;
      font-family: monospace;
    }

    .charts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      gap: 16px;
      flex: 1;
    }

    .chart-container {
      background: var(--bg-secondary);
      border-radius: 8px;
      padding: 12px;
    }

    .chart-container h4 {
      margin-bottom: 8px;
      font-size: 14px;
    }

    .insights-panel {
      background: var(--bg-secondary);
      border-radius: 8px;
      padding: 12px;
      overflow-y: auto;
    }

    .insights-panel h3 {
      font-size: 12px;
      text-transform: uppercase;
      color: var(--text-secondary);
      margin-bottom: 12px;
    }

    .insight {
      background: var(--bg-primary);
      border-radius: 6px;
      padding: 10px;
      margin-bottom: 8px;
    }

    .insight-header {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 4px;
    }

    .insight-severity {
      width: 6px;
      height: 6px;
      border-radius: 50%;
    }

    .severity-critical { background: #f44336; }
    .severity-warning { background: #ff9800; }
    .severity-info { background: #2196f3; }

    .insight-title {
      font-weight: 600;
      font-size: 13px;
    }

    .insight-desc {
      font-size: 12px;
      color: var(--text-secondary);
      margin-bottom: 8px;
    }

    .insight-actions {
      display: flex;
      gap: 8px;
    }

    .vote-btn {
      background: none;
      border: 1px solid var(--border-color);
      padding: 2px 8px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      color: var(--text-secondary);
    }

    .vote-btn:hover {
      background: var(--vscode-list-hoverBackground);
    }

    .loading {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100%;
      color: var(--text-secondary);
    }
  </style>
</head>
<body>
  <div class="dashboard">
    <div class="header">
      <h1>Atlas Dashboard</h1>
      <div class="header-actions">
        <button class="btn" onclick="refresh()">Refresh</button>
        <button class="btn" onclick="promptToPlot()">Prompt to Plot</button>
      </div>
    </div>

    <div class="sidebar">
      <h3>Runs</h3>
      <div id="runs-list">
        <div class="loading">Loading...</div>
      </div>
    </div>

    <div class="main-content">
      <div class="ticker" id="ticker">
        <div class="ticker-item">
          <div class="label">Loss</div>
          <div class="value" id="ticker-loss">--</div>
        </div>
        <div class="ticker-item">
          <div class="label">Step</div>
          <div class="value" id="ticker-step">--</div>
        </div>
        <div class="ticker-item">
          <div class="label">Duration</div>
          <div class="value" id="ticker-duration">--</div>
        </div>
        <div class="ticker-item">
          <div class="label">Status</div>
          <div class="value" id="ticker-status">--</div>
        </div>
      </div>

      <div class="charts-grid" id="charts-grid">
        <div class="loading">Select a run to view metrics</div>
      </div>
    </div>

    <div class="insights-panel">
      <h3>AI Insights</h3>
      <div id="insights-list">
        <div class="loading">No insights yet</div>
      </div>
    </div>
  </div>

  <script>
    const vscode = acquireVsCodeApi();
    let currentRuns = [];
    let currentRunId = null;
    let charts = {};

    // Send ready message
    vscode.postMessage({ type: 'ready' });

    // Handle messages from extension
    window.addEventListener('message', async (event) => {
      const message = event.data;

      switch (message.type) {
        case 'init':
          currentRuns = message.runs;
          renderRunsList(message.runs);
          break;

        case 'runData':
          currentRunId = message.run.id;
          updateTicker(message.run, message.metricsData);
          renderCharts(message.metricsData);
          renderInsights(message.insights);
          break;

        case 'metrics':
          if (charts[message.metricName]) {
            updateChart(message.metricName, message.data);
          }
          break;

        case 'renderChart':
          renderCustomChart(message.spec, message.prompt);
          break;
      }
    });

    function renderRunsList(runs) {
      const container = document.getElementById('runs-list');
      container.innerHTML = runs.map(run => \`
        <div class="run-item \${run.id === currentRunId ? 'active' : ''}" onclick="selectRun('\${run.id}')">
          <span class="status-dot status-\${run.status}"></span>
          <span>\${run.name}</span>
        </div>
      \`).join('');
    }

    function selectRun(runId) {
      currentRunId = runId;
      vscode.postMessage({ type: 'selectRun', runId });

      // Update active state
      document.querySelectorAll('.run-item').forEach(el => {
        el.classList.toggle('active', el.onclick.toString().includes(runId));
      });
    }

    function updateTicker(run, metricsData) {
      // Find loss metric
      const lossMetric = Object.keys(metricsData).find(k =>
        k.toLowerCase().includes('loss')
      );
      if (lossMetric && metricsData[lossMetric].length > 0) {
        const latest = metricsData[lossMetric][metricsData[lossMetric].length - 1];
        document.getElementById('ticker-loss').textContent = latest.value.toFixed(4);
        document.getElementById('ticker-step').textContent = latest.step;
      }

      document.getElementById('ticker-status').textContent = run.status;

      if (run.duration_seconds) {
        document.getElementById('ticker-duration').textContent =
          Math.round(run.duration_seconds) + 's';
      }
    }

    async function renderCharts(metricsData) {
      const container = document.getElementById('charts-grid');
      container.innerHTML = '';
      charts = {};

      for (const [name, data] of Object.entries(metricsData)) {
        if (data.length === 0) continue;

        const chartDiv = document.createElement('div');
        chartDiv.className = 'chart-container';
        chartDiv.innerHTML = \`<h4>\${name}</h4><div id="chart-\${name.replace(/[^a-zA-Z0-9]/g, '_')}"></div>\`;
        container.appendChild(chartDiv);

        const spec = {
          $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
          width: 'container',
          height: 200,
          data: { values: data },
          mark: { type: 'line', point: false, strokeWidth: 1.5 },
          encoding: {
            x: { field: 'step', type: 'quantitative', title: 'Step' },
            y: {
              field: 'value',
              type: 'quantitative',
              title: name,
              scale: name.toLowerCase().includes('loss') ? { type: 'log' } : {}
            },
            tooltip: [
              { field: 'step', type: 'quantitative' },
              { field: 'value', type: 'quantitative', format: '.4f' }
            ]
          },
          config: {
            background: 'transparent',
            axis: { labelColor: '#888', titleColor: '#888', gridColor: '#333' },
            view: { stroke: 'transparent' }
          }
        };

        const chartId = 'chart-' + name.replace(/[^a-zA-Z0-9]/g, '_');
        try {
          const result = await vegaEmbed('#' + chartId, spec, {
            actions: false,
            renderer: 'canvas'
          });
          charts[name] = result;
        } catch (e) {
          console.error('Chart error:', e);
        }
      }
    }

    function updateChart(metricName, data) {
      if (charts[metricName]) {
        charts[metricName].view.change('data', vega.changeset().remove(() => true).insert(data)).run();
      }
    }

    function renderInsights(insights) {
      const container = document.getElementById('insights-list');

      if (!insights || insights.length === 0) {
        container.innerHTML = '<div class="loading">No insights yet</div>';
        return;
      }

      container.innerHTML = insights.map(insight => \`
        <div class="insight">
          <div class="insight-header">
            <span class="insight-severity severity-\${insight.severity}"></span>
            <span class="insight-title">\${insight.title}</span>
          </div>
          <div class="insight-desc">\${insight.description}</div>
          <div class="insight-actions">
            <button class="vote-btn" onclick="vote('\${insight.id}', 1)">👍 \${insight.votes_up}</button>
            <button class="vote-btn" onclick="vote('\${insight.id}', -1)">👎 \${insight.votes_down}</button>
          </div>
        </div>
      \`).join('');
    }

    async function renderCustomChart(spec, prompt) {
      const container = document.getElementById('charts-grid');

      const chartDiv = document.createElement('div');
      chartDiv.className = 'chart-container';
      chartDiv.innerHTML = \`<h4>\${prompt}</h4><div id="custom-chart"></div>\`;
      container.prepend(chartDiv);

      try {
        await vegaEmbed('#custom-chart', spec, {
          actions: false,
          renderer: 'canvas'
        });
      } catch (e) {
        chartDiv.innerHTML = \`<h4>\${prompt}</h4><div class="loading">Failed to render chart</div>\`;
      }
    }

    function refresh() {
      vscode.postMessage({ type: 'refresh' });
    }

    function promptToPlot() {
      // This is handled by the extension command
      vscode.postMessage({ type: 'promptToPlot' });
    }

    function vote(insightId, value) {
      vscode.postMessage({
        type: 'vote',
        insightId,
        runId: currentRunId,
        vote: value
      });
    }
  </script>
</body>
</html>`;
  }

  private dispose(): void {
    DashboardPanel.currentPanel = undefined;

    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }

    this.panel.dispose();

    while (this.disposables.length) {
      const disposable = this.disposables.pop();
      if (disposable) {
        disposable.dispose();
      }
    }
  }
}
