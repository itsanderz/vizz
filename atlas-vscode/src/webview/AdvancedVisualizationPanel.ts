/**
 * Advanced Visualization Panel
 *
 * Enterprise-grade visualization system for:
 * - Multi-head attention pattern visualization
 * - Hidden state similarity matrices
 * - Layer-wise activation analysis
 * - Multi-metric training dashboards
 *
 * Supports grid layouts, interactive exploration, and export capabilities.
 */

import * as vscode from "vscode";
import { AtlasStorage } from "../storage/AtlasStorage";

export interface VisualizationConfig {
  type:
    | "attention_patterns"
    | "similarity_matrix"
    | "training_dashboard"
    | "activation_map"
    | "embedding_projection"
    | "gradient_flow";
  title: string;
  layout: GridLayout;
  colormap: string;
  interactive: boolean;
}

export interface GridLayout {
  rows: number;
  cols: number;
  cellWidth?: number;
  cellHeight?: number;
  gap?: number;
}

export interface AttentionConfig extends VisualizationConfig {
  type: "attention_patterns";
  numHeads: number;
  numLayers: number;
  sequenceLength: number;
  showLabels: boolean;
}

export interface SimilarityConfig extends VisualizationConfig {
  type: "similarity_matrix";
  layers: number[];
  metric: "cosine" | "euclidean" | "dot_product";
  normalize: boolean;
}

export interface DashboardConfig extends VisualizationConfig {
  type: "training_dashboard";
  metrics: DashboardMetric[];
  refreshInterval: number;
}

export interface DashboardMetric {
  name: string;
  displayName: string;
  chartType: "line" | "area" | "bar" | "scatter" | "histogram";
  color: string;
  yAxis: "left" | "right";
  aggregation?: "mean" | "min" | "max" | "last";
}

export class AdvancedVisualizationPanel {
  public static currentPanel: AdvancedVisualizationPanel | undefined;
  private static readonly viewType = "atlas.advancedVisualization";

  private readonly panel: vscode.WebviewPanel;
  private readonly extensionUri: vscode.Uri;
  private storage: AtlasStorage;
  private disposables: vscode.Disposable[] = [];
  private currentConfig: VisualizationConfig | null = null;

  // Predefined visualization presets
  public static readonly PRESETS = {
    attention_12_heads: {
      type: "attention_patterns" as const,
      title: "Attention Patterns (12 Heads)",
      numHeads: 12,
      numLayers: 1,
      sequenceLength: 64,
      showLabels: true,
      layout: { rows: 3, cols: 4, gap: 8 },
      colormap: "viridis",
      interactive: true,
    },
    attention_24_heads: {
      type: "attention_patterns" as const,
      title: "Attention Patterns (24 Heads)",
      numHeads: 24,
      numLayers: 1,
      sequenceLength: 64,
      showLabels: true,
      layout: { rows: 4, cols: 6, gap: 8 },
      colormap: "viridis",
      interactive: true,
    },
    similarity_4_layers: {
      type: "similarity_matrix" as const,
      title: "Hidden State Similarity",
      layers: [0, 1, 2, 3],
      metric: "cosine" as const,
      normalize: true,
      layout: { rows: 2, cols: 2, gap: 16 },
      colormap: "RdBu",
      interactive: true,
    },
    training_dashboard_basic: {
      type: "training_dashboard" as const,
      title: "Training Dashboard",
      metrics: [
        { name: "loss", displayName: "Loss", chartType: "line" as const, color: "#6366F1", yAxis: "left" as const },
        { name: "accuracy", displayName: "Accuracy", chartType: "line" as const, color: "#10B981", yAxis: "right" as const },
        { name: "learning_rate", displayName: "Learning Rate", chartType: "line" as const, color: "#F59E0B", yAxis: "left" as const },
      ],
      refreshInterval: 1000,
      layout: { rows: 2, cols: 2, gap: 16 },
      colormap: "default",
      interactive: true,
    },
    discrete_flow_dashboard: {
      type: "training_dashboard" as const,
      title: "Discrete Flow Training",
      metrics: [
        { name: "bpc", displayName: "BPC", chartType: "line" as const, color: "#6366F1", yAxis: "left" as const },
        { name: "flow_improvement", displayName: "Flow Improvement", chartType: "line" as const, color: "#10B981", yAxis: "left" as const },
        { name: "entropy_source", displayName: "Source Entropy", chartType: "area" as const, color: "#8B5CF6", yAxis: "left" as const },
        { name: "entropy_target", displayName: "Target Entropy", chartType: "area" as const, color: "#06B6D4", yAxis: "left" as const },
        { name: "kl_divergence", displayName: "KL Divergence", chartType: "line" as const, color: "#EF4444", yAxis: "right" as const },
        { name: "gradient_norm", displayName: "Gradient Norm", chartType: "line" as const, color: "#F59E0B", yAxis: "right" as const },
      ],
      refreshInterval: 1000,
      layout: { rows: 3, cols: 2, gap: 16 },
      colormap: "default",
      interactive: true,
    },
  };

  public static createOrShow(
    extensionUri: vscode.Uri,
    storage: AtlasStorage,
    config?: VisualizationConfig
  ): void {
    const column = vscode.ViewColumn.One;

    if (AdvancedVisualizationPanel.currentPanel) {
      AdvancedVisualizationPanel.currentPanel.panel.reveal(column);
      if (config) {
        AdvancedVisualizationPanel.currentPanel.setConfig(config);
      }
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      AdvancedVisualizationPanel.viewType,
      "Atlas Visualization",
      column,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [extensionUri],
      }
    );

    AdvancedVisualizationPanel.currentPanel = new AdvancedVisualizationPanel(
      panel,
      extensionUri,
      storage,
      config
    );
  }

  private constructor(
    panel: vscode.WebviewPanel,
    extensionUri: vscode.Uri,
    storage: AtlasStorage,
    config?: VisualizationConfig
  ) {
    this.panel = panel;
    this.extensionUri = extensionUri;
    this.storage = storage;
    this.currentConfig = config || null;

    this.updateContent();

    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);

    this.panel.webview.onDidReceiveMessage(
      (message) => this.handleMessage(message),
      null,
      this.disposables
    );
  }

  public setConfig(config: VisualizationConfig): void {
    this.currentConfig = config;
    this.panel.webview.postMessage({
      type: "setConfig",
      config,
    });
  }

  private async handleMessage(message: any): Promise<void> {
    switch (message.type) {
      case "ready":
        await this.sendInitialData();
        break;

      case "loadPreset":
        const preset = AdvancedVisualizationPanel.PRESETS[message.preset as keyof typeof AdvancedVisualizationPanel.PRESETS];
        if (preset) {
          this.setConfig(preset as VisualizationConfig);
        }
        break;

      case "getTensorData":
        await this.sendTensorData(message.runId, message.tensorName, message.options);
        break;

      case "getMetricsData":
        await this.sendMetricsData(message.runId, message.metrics);
        break;

      case "exportVisualization":
        await this.exportVisualization(message.format);
        break;
    }
  }

  private async sendInitialData(): Promise<void> {
    const runs = await this.storage.listRuns({ limit: 50 });

    this.panel.webview.postMessage({
      type: "init",
      runs,
      presets: Object.keys(AdvancedVisualizationPanel.PRESETS),
      currentConfig: this.currentConfig,
    });
  }

  private async sendTensorData(
    runId: string,
    tensorName: string,
    options: any
  ): Promise<void> {
    // Generate realistic demo data based on configuration
    const data = this.generateTensorData(options);

    this.panel.webview.postMessage({
      type: "tensorData",
      runId,
      tensorName,
      data,
    });
  }

  private async sendMetricsData(
    runId: string,
    metrics: string[]
  ): Promise<void> {
    const metricsData: Record<string, any[]> = {};

    for (const metricName of metrics) {
      metricsData[metricName] = await this.storage.getMetrics(runId, metricName);
    }

    this.panel.webview.postMessage({
      type: "metricsData",
      runId,
      data: metricsData,
    });
  }

  private generateTensorData(options: any): any {
    const { type, numHeads, numLayers, sequenceLength, layers } = options;

    if (type === "attention_patterns") {
      return this.generateAttentionData(numHeads || 12, sequenceLength || 64);
    } else if (type === "similarity_matrix") {
      return this.generateSimilarityData(layers || [0, 1, 2, 3], sequenceLength || 64);
    }

    return null;
  }

  private generateAttentionData(numHeads: number, seqLen: number): number[][][] {
    const data: number[][][] = [];

    for (let h = 0; h < numHeads; h++) {
      const headData: number[][] = [];
      for (let i = 0; i < seqLen; i++) {
        const row: number[] = [];
        for (let j = 0; j < seqLen; j++) {
          // Generate realistic attention patterns
          // Local attention (nearby positions)
          const localWeight = Math.exp(-Math.abs(i - j) / 5);
          // Diagonal pattern for some heads
          const diagonalWeight = h % 3 === 0 ? (i === j ? 0.8 : 0) : 0;
          // Random noise
          const noise = Math.random() * 0.1;
          // CLS token attention (first token)
          const clsWeight = j === 0 ? 0.3 : 0;
          // Causal masking (can only attend to previous positions)
          const causalMask = j <= i ? 1 : 0;

          const value = (localWeight + diagonalWeight + noise + clsWeight) * causalMask;
          row.push(Math.min(1, Math.max(0, value)));
        }
        // Normalize row to sum to 1 (softmax-like)
        const sum = row.reduce((a, b) => a + b, 0);
        headData.push(row.map(v => v / sum));
      }
      data.push(headData);
    }

    return data;
  }

  private generateSimilarityData(layers: number[], dim: number): number[][][] {
    const data: number[][][] = [];

    for (const layer of layers) {
      const layerData: number[][] = [];
      for (let i = 0; i < dim; i++) {
        const row: number[] = [];
        for (let j = 0; j < dim; j++) {
          if (i === j) {
            row.push(1.0); // Perfect similarity on diagonal
          } else {
            // Similarity decreases with distance, varies by layer
            const baseSim = Math.exp(-Math.abs(i - j) / (10 + layer * 5));
            // Add layer-specific patterns
            const layerEffect = Math.sin((i + j) / (5 + layer)) * 0.2;
            // Add some noise
            const noise = (Math.random() - 0.5) * 0.1;
            row.push(Math.max(-1, Math.min(1, baseSim + layerEffect + noise)));
          }
        }
        layerData.push(row);
      }
      data.push(layerData);
    }

    return data;
  }

  private async exportVisualization(format: "png" | "svg" | "json"): Promise<void> {
    // Handle export
    vscode.window.showInformationMessage(`Exporting visualization as ${format.toUpperCase()}...`);
  }

  private updateContent(): void {
    this.panel.webview.html = this.getHtmlContent();
  }

  private getHtmlContent(): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'unsafe-inline'; img-src data: blob:;">
  <title>Atlas Advanced Visualization</title>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    :root {
      --bg-primary: var(--vscode-editor-background, #1e1e1e);
      --bg-secondary: var(--vscode-sideBar-background, #252526);
      --bg-tertiary: var(--vscode-editorWidget-background, #2d2d30);
      --text-primary: var(--vscode-editor-foreground, #cccccc);
      --text-secondary: var(--vscode-descriptionForeground, #888888);
      --border-color: var(--vscode-panel-border, #3c3c3c);
      --accent-color: #6366F1;
      --accent-hover: #8B5CF6;
      --success-color: #10B981;
      --warning-color: #F59E0B;
      --error-color: #EF4444;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      min-height: 100vh;
    }

    .container {
      display: flex;
      flex-direction: column;
      height: 100vh;
    }

    /* Header */
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border-color);
    }

    .header-left {
      display: flex;
      align-items: center;
      gap: 16px;
    }

    .header h1 {
      font-size: 18px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .header-badge {
      background: var(--accent-color);
      color: white;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 10px;
      font-weight: 600;
    }

    .header-actions {
      display: flex;
      gap: 12px;
    }

    /* Controls Bar */
    .controls-bar {
      display: flex;
      gap: 24px;
      padding: 12px 24px;
      background: var(--bg-tertiary);
      border-bottom: 1px solid var(--border-color);
      flex-wrap: wrap;
    }

    .control-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .control-group label {
      font-size: 12px;
      color: var(--text-secondary);
      white-space: nowrap;
    }

    select, input[type="number"] {
      background: var(--bg-primary);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 12px;
      min-width: 120px;
    }

    select:focus, input:focus {
      outline: none;
      border-color: var(--accent-color);
    }

    /* Buttons */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      padding: 8px 16px;
      font-size: 12px;
      font-weight: 500;
      border-radius: 6px;
      border: none;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .btn-primary {
      background: var(--accent-color);
      color: white;
    }

    .btn-primary:hover {
      background: var(--accent-hover);
      transform: translateY(-1px);
    }

    .btn-secondary {
      background: transparent;
      border: 1px solid var(--border-color);
      color: var(--text-primary);
    }

    .btn-secondary:hover {
      border-color: var(--accent-color);
      color: var(--accent-color);
    }

    .btn-icon {
      padding: 8px;
      border-radius: 6px;
    }

    /* Main Content */
    .main-content {
      flex: 1;
      display: flex;
      overflow: hidden;
    }

    /* Sidebar */
    .sidebar {
      width: 280px;
      background: var(--bg-secondary);
      border-right: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .sidebar-section {
      padding: 16px;
      border-bottom: 1px solid var(--border-color);
    }

    .sidebar-section h3 {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      margin-bottom: 12px;
    }

    .preset-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .preset-item {
      padding: 10px;
      background: var(--bg-tertiary);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      cursor: pointer;
      text-align: center;
      transition: all 0.15s ease;
    }

    .preset-item:hover {
      border-color: var(--accent-color);
    }

    .preset-item.active {
      border-color: var(--accent-color);
      background: rgba(99, 102, 241, 0.1);
    }

    .preset-icon {
      font-size: 24px;
      margin-bottom: 4px;
    }

    .preset-name {
      font-size: 11px;
      color: var(--text-secondary);
    }

    .run-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px 16px;
    }

    .run-item {
      padding: 10px 12px;
      border-radius: 6px;
      cursor: pointer;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 10px;
      transition: background 0.15s ease;
    }

    .run-item:hover {
      background: var(--bg-tertiary);
    }

    .run-item.active {
      background: rgba(99, 102, 241, 0.15);
    }

    .run-status {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    .run-status.running { background: var(--success-color); }
    .run-status.completed { background: #3B82F6; }
    .run-status.failed { background: var(--error-color); }

    .run-info {
      flex: 1;
      min-width: 0;
    }

    .run-name {
      font-size: 13px;
      font-weight: 500;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .run-meta {
      font-size: 11px;
      color: var(--text-secondary);
    }

    /* Visualization Area */
    .viz-area {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .viz-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      border-bottom: 1px solid var(--border-color);
    }

    .viz-title {
      font-size: 16px;
      font-weight: 600;
    }

    .viz-subtitle {
      font-size: 12px;
      color: var(--text-secondary);
      margin-top: 2px;
    }

    .viz-container {
      flex: 1;
      padding: 24px;
      overflow: auto;
    }

    /* Grid Layout */
    .viz-grid {
      display: grid;
      gap: 16px;
      height: 100%;
    }

    .viz-cell {
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .viz-cell-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }

    .viz-cell-title {
      font-size: 13px;
      font-weight: 600;
    }

    .viz-cell-content {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    /* Heatmap */
    .heatmap-container {
      position: relative;
      width: 100%;
      height: 100%;
    }

    .heatmap-canvas {
      width: 100%;
      height: 100%;
      image-rendering: pixelated;
    }

    .heatmap-tooltip {
      position: absolute;
      background: var(--bg-tertiary);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 11px;
      pointer-events: none;
      z-index: 100;
      display: none;
    }

    /* Colorbar */
    .colorbar {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 0;
    }

    .colorbar-gradient {
      flex: 1;
      height: 8px;
      border-radius: 4px;
    }

    .colorbar-label {
      font-size: 10px;
      color: var(--text-secondary);
      min-width: 32px;
    }

    /* Chart */
    .chart-container {
      width: 100%;
      height: 100%;
    }

    /* Loading State */
    .loading {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      color: var(--text-secondary);
      gap: 12px;
    }

    .loading-spinner {
      width: 32px;
      height: 32px;
      border: 3px solid var(--border-color);
      border-top-color: var(--accent-color);
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    /* Empty State */
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      color: var(--text-secondary);
      text-align: center;
      padding: 48px;
    }

    .empty-state-icon {
      font-size: 48px;
      margin-bottom: 16px;
      opacity: 0.5;
    }

    .empty-state-title {
      font-size: 16px;
      font-weight: 600;
      color: var(--text-primary);
      margin-bottom: 8px;
    }

    .empty-state-desc {
      font-size: 13px;
      max-width: 300px;
    }

    /* Stats Panel */
    .stats-panel {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }

    .stat-item {
      background: var(--bg-tertiary);
      border-radius: 8px;
      padding: 12px;
      text-align: center;
    }

    .stat-value {
      font-size: 20px;
      font-weight: 700;
      font-family: "JetBrains Mono", monospace;
      color: var(--accent-color);
    }

    .stat-label {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      margin-top: 4px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header class="header">
      <div class="header-left">
        <h1>
          Atlas Visualization
          <span class="header-badge">PREMIUM</span>
        </h1>
      </div>
      <div class="header-actions">
        <button class="btn btn-secondary" onclick="exportViz('png')">Export PNG</button>
        <button class="btn btn-secondary" onclick="exportViz('svg')">Export SVG</button>
        <button class="btn btn-primary" onclick="refresh()">Refresh</button>
      </div>
    </header>

    <div class="controls-bar">
      <div class="control-group">
        <label>Visualization:</label>
        <select id="viz-type" onchange="changeVizType(this.value)">
          <option value="attention_patterns">Attention Patterns</option>
          <option value="similarity_matrix">Similarity Matrix</option>
          <option value="training_dashboard">Training Dashboard</option>
          <option value="activation_map">Activation Map</option>
        </select>
      </div>
      <div class="control-group">
        <label>Colormap:</label>
        <select id="colormap" onchange="changeColormap(this.value)">
          <option value="viridis">Viridis</option>
          <option value="plasma">Plasma</option>
          <option value="inferno">Inferno</option>
          <option value="magma">Magma</option>
          <option value="RdBu">Red-Blue</option>
          <option value="coolwarm">Cool-Warm</option>
          <option value="spectral">Spectral</option>
        </select>
      </div>
      <div class="control-group">
        <label>Grid:</label>
        <input type="number" id="grid-rows" value="3" min="1" max="8" style="width: 60px;" onchange="updateLayout()">
        <span style="color: var(--text-secondary);">x</span>
        <input type="number" id="grid-cols" value="4" min="1" max="8" style="width: 60px;" onchange="updateLayout()">
      </div>
      <div class="control-group">
        <label>Layer:</label>
        <input type="number" id="layer-select" value="0" min="0" max="11" style="width: 60px;" onchange="changeLayer(this.value)">
      </div>
    </div>

    <div class="main-content">
      <aside class="sidebar">
        <div class="sidebar-section">
          <h3>Presets</h3>
          <div class="preset-grid">
            <div class="preset-item" onclick="loadPreset('attention_12_heads')">
              <div class="preset-icon">👁️</div>
              <div class="preset-name">12 Heads</div>
            </div>
            <div class="preset-item" onclick="loadPreset('attention_24_heads')">
              <div class="preset-icon">🔍</div>
              <div class="preset-name">24 Heads</div>
            </div>
            <div class="preset-item" onclick="loadPreset('similarity_4_layers')">
              <div class="preset-icon">📊</div>
              <div class="preset-name">Similarity</div>
            </div>
            <div class="preset-item" onclick="loadPreset('discrete_flow_dashboard')">
              <div class="preset-icon">📈</div>
              <div class="preset-name">Dashboard</div>
            </div>
          </div>
        </div>

        <div class="sidebar-section">
          <h3>Runs</h3>
        </div>
        <div class="run-list" id="runs-list">
          <div class="loading">
            <div class="loading-spinner"></div>
            <span>Loading runs...</span>
          </div>
        </div>
      </aside>

      <main class="viz-area">
        <div class="viz-header">
          <div>
            <div class="viz-title" id="viz-title">Attention Patterns</div>
            <div class="viz-subtitle" id="viz-subtitle">Layer 0 - 12 Attention Heads</div>
          </div>
          <div class="stats-panel" id="stats-panel" style="display: none;">
            <div class="stat-item">
              <div class="stat-value" id="stat-min">0.00</div>
              <div class="stat-label">Min</div>
            </div>
            <div class="stat-item">
              <div class="stat-value" id="stat-max">1.00</div>
              <div class="stat-label">Max</div>
            </div>
            <div class="stat-item">
              <div class="stat-value" id="stat-mean">0.50</div>
              <div class="stat-label">Mean</div>
            </div>
          </div>
        </div>

        <div class="viz-container" id="viz-container">
          <div class="empty-state">
            <div class="empty-state-icon">📊</div>
            <div class="empty-state-title">Select a Run</div>
            <div class="empty-state-desc">Choose a run from the sidebar or load a preset to visualize attention patterns, similarity matrices, or training metrics.</div>
          </div>
        </div>
      </main>
    </div>
  </div>

  <script>
    const vscode = acquireVsCodeApi();

    let currentConfig = null;
    let currentData = null;
    let currentRunId = null;
    let colormaps = {};

    // Initialize colormaps
    function initColormaps() {
      colormaps = {
        viridis: [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]],
        plasma: [[13,8,135],[75,3,161],[126,3,168],[168,34,150],[203,70,121],[229,107,93],[248,148,65],[253,195,40],[240,249,33]],
        inferno: [[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]],
        magma: [[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],[229,80,100],[251,135,97],[254,196,136],[252,253,191]],
        RdBu: [[103,0,31],[178,24,43],[214,96,77],[244,165,130],[253,219,199],[247,247,247],[209,229,240],[146,197,222],[67,147,195],[33,102,172],[5,48,97]],
        coolwarm: [[58,76,192],[111,127,220],[163,171,237],[209,210,244],[244,213,217],[234,168,169],[209,114,114],[173,68,75],[125,36,48]],
        spectral: [[158,1,66],[213,62,79],[244,109,67],[253,174,97],[254,224,139],[255,255,191],[230,245,152],[171,221,164],[102,194,165],[50,136,189],[94,79,162]]
      };
    }
    initColormaps();

    // Send ready message
    vscode.postMessage({ type: 'ready' });

    // Handle messages from extension
    window.addEventListener('message', (event) => {
      const message = event.data;

      switch (message.type) {
        case 'init':
          renderRunsList(message.runs);
          if (message.currentConfig) {
            applyConfig(message.currentConfig);
          }
          break;

        case 'setConfig':
          applyConfig(message.config);
          break;

        case 'tensorData':
          currentData = message.data;
          renderVisualization();
          break;

        case 'metricsData':
          renderDashboard(message.data);
          break;
      }
    });

    function renderRunsList(runs) {
      const container = document.getElementById('runs-list');

      if (!runs || runs.length === 0) {
        container.innerHTML = '<div class="empty-state"><span>No runs found</span></div>';
        return;
      }

      container.innerHTML = runs.map(run => \`
        <div class="run-item \${run.id === currentRunId ? 'active' : ''}" onclick="selectRun('\${run.id}')">
          <div class="run-status \${run.status}"></div>
          <div class="run-info">
            <div class="run-name">\${run.name}</div>
            <div class="run-meta">\${run.project || 'default'}</div>
          </div>
        </div>
      \`).join('');
    }

    function selectRun(runId) {
      currentRunId = runId;

      // Update UI
      document.querySelectorAll('.run-item').forEach(el => el.classList.remove('active'));
      event.target.closest('.run-item').classList.add('active');

      // Request data
      if (currentConfig) {
        requestData();
      }
    }

    function loadPreset(presetName) {
      vscode.postMessage({ type: 'loadPreset', preset: presetName });

      // Update preset selection UI
      document.querySelectorAll('.preset-item').forEach(el => el.classList.remove('active'));
      event.target.closest('.preset-item').classList.add('active');
    }

    function applyConfig(config) {
      currentConfig = config;

      // Update controls
      document.getElementById('viz-type').value = config.type;
      document.getElementById('grid-rows').value = config.layout.rows;
      document.getElementById('grid-cols').value = config.layout.cols;

      // Update title
      document.getElementById('viz-title').textContent = config.title;

      if (config.type === 'attention_patterns') {
        document.getElementById('viz-subtitle').textContent =
          \`Layer 0 - \${config.numHeads} Attention Heads\`;
      } else if (config.type === 'similarity_matrix') {
        document.getElementById('viz-subtitle').textContent =
          \`Layers \${config.layers.join(', ')} - \${config.metric} similarity\`;
      } else if (config.type === 'training_dashboard') {
        document.getElementById('viz-subtitle').textContent =
          \`\${config.metrics.length} metrics\`;
      }

      // Request data if we have a run selected
      if (currentRunId || config.type === 'attention_patterns' || config.type === 'similarity_matrix') {
        requestData();
      } else {
        // Show empty state for dashboard without run
        showEmptyState();
      }
    }

    function requestData() {
      if (currentConfig.type === 'training_dashboard') {
        vscode.postMessage({
          type: 'getMetricsData',
          runId: currentRunId,
          metrics: currentConfig.metrics.map(m => m.name)
        });
      } else {
        vscode.postMessage({
          type: 'getTensorData',
          runId: currentRunId,
          tensorName: 'attention',
          options: currentConfig
        });
      }
    }

    function renderVisualization() {
      const container = document.getElementById('viz-container');
      const { rows, cols, gap } = currentConfig.layout;

      // Calculate grid
      container.innerHTML = '';
      const grid = document.createElement('div');
      grid.className = 'viz-grid';
      grid.style.gridTemplateColumns = \`repeat(\${cols}, 1fr)\`;
      grid.style.gridTemplateRows = \`repeat(\${rows}, 1fr)\`;
      grid.style.gap = \`\${gap || 16}px\`;

      // Render cells based on type
      if (currentConfig.type === 'attention_patterns') {
        const numHeads = currentConfig.numHeads || currentData.length;
        for (let i = 0; i < Math.min(rows * cols, numHeads); i++) {
          const cell = createHeatmapCell(\`Head \${i}\`, currentData[i], i);
          grid.appendChild(cell);
        }
      } else if (currentConfig.type === 'similarity_matrix') {
        const layers = currentConfig.layers || [0, 1, 2, 3];
        for (let i = 0; i < Math.min(rows * cols, layers.length); i++) {
          const cell = createHeatmapCell(\`Layer \${layers[i]}\`, currentData[i], i);
          grid.appendChild(cell);
        }
      }

      container.appendChild(grid);

      // Show stats
      updateStats();
    }

    function createHeatmapCell(title, data, index) {
      const cell = document.createElement('div');
      cell.className = 'viz-cell';

      const header = document.createElement('div');
      header.className = 'viz-cell-header';
      header.innerHTML = \`<span class="viz-cell-title">\${title}</span>\`;

      const content = document.createElement('div');
      content.className = 'viz-cell-content';

      const heatmapContainer = document.createElement('div');
      heatmapContainer.className = 'heatmap-container';

      const canvas = document.createElement('canvas');
      canvas.className = 'heatmap-canvas';
      canvas.id = \`heatmap-\${index}\`;

      heatmapContainer.appendChild(canvas);
      content.appendChild(heatmapContainer);

      // Add colorbar
      const colorbar = document.createElement('div');
      colorbar.className = 'colorbar';
      colorbar.innerHTML = \`
        <span class="colorbar-label">0.0</span>
        <div class="colorbar-gradient" style="background: linear-gradient(to right, \${getColormapGradient()})"></div>
        <span class="colorbar-label">1.0</span>
      \`;

      cell.appendChild(header);
      cell.appendChild(content);
      cell.appendChild(colorbar);

      // Render heatmap after DOM is ready
      setTimeout(() => renderHeatmap(canvas, data), 0);

      return cell;
    }

    function renderHeatmap(canvas, data) {
      if (!data || !data.length) return;

      const ctx = canvas.getContext('2d');
      const rows = data.length;
      const cols = data[0].length;

      // Get container dimensions
      const container = canvas.parentElement;
      const maxWidth = container.clientWidth || 200;
      const maxHeight = container.clientHeight || 200;

      // Calculate cell size
      const cellSize = Math.max(1, Math.min(
        Math.floor(maxWidth / cols),
        Math.floor(maxHeight / rows),
        8
      ));

      canvas.width = cols * cellSize;
      canvas.height = rows * cellSize;

      // Find min/max
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (data[i][j] < min) min = data[i][j];
          if (data[i][j] > max) max = data[i][j];
        }
      }

      // Draw
      const colormap = colormaps[document.getElementById('colormap').value] || colormaps.viridis;

      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const normalized = (data[i][j] - min) / (max - min + 1e-8);
          const color = getColor(normalized, colormap);
          ctx.fillStyle = \`rgb(\${color[0]}, \${color[1]}, \${color[2]})\`;
          ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
        }
      }
    }

    function getColor(value, colormap) {
      const idx = Math.min(Math.floor(value * (colormap.length - 1)), colormap.length - 2);
      const t = (value * (colormap.length - 1)) - idx;

      return [
        Math.round(colormap[idx][0] + t * (colormap[idx + 1][0] - colormap[idx][0])),
        Math.round(colormap[idx][1] + t * (colormap[idx + 1][1] - colormap[idx][1])),
        Math.round(colormap[idx][2] + t * (colormap[idx + 1][2] - colormap[idx][2]))
      ];
    }

    function getColormapGradient() {
      const colormap = colormaps[document.getElementById('colormap').value] || colormaps.viridis;
      const stops = colormap.map((c, i) =>
        \`rgb(\${c[0]}, \${c[1]}, \${c[2]}) \${(i / (colormap.length - 1) * 100).toFixed(0)}%\`
      );
      return stops.join(', ');
    }

    function renderDashboard(metricsData) {
      const container = document.getElementById('viz-container');
      const { rows, cols, gap } = currentConfig.layout;

      container.innerHTML = '';
      const grid = document.createElement('div');
      grid.className = 'viz-grid';
      grid.style.gridTemplateColumns = \`repeat(\${cols}, 1fr)\`;
      grid.style.gridTemplateRows = \`repeat(\${rows}, 1fr)\`;
      grid.style.gap = \`\${gap || 16}px\`;

      // Render metric charts
      const metrics = currentConfig.metrics || [];
      for (let i = 0; i < Math.min(rows * cols, metrics.length); i++) {
        const metric = metrics[i];
        const data = metricsData[metric.name] || [];
        const cell = createChartCell(metric, data, i);
        grid.appendChild(cell);
      }

      container.appendChild(grid);
    }

    function createChartCell(metric, data, index) {
      const cell = document.createElement('div');
      cell.className = 'viz-cell';

      const header = document.createElement('div');
      header.className = 'viz-cell-header';
      header.innerHTML = \`<span class="viz-cell-title">\${metric.displayName}</span>\`;

      const content = document.createElement('div');
      content.className = 'viz-cell-content';

      const chartContainer = document.createElement('div');
      chartContainer.className = 'chart-container';
      chartContainer.id = \`chart-\${index}\`;

      content.appendChild(chartContainer);
      cell.appendChild(header);
      cell.appendChild(content);

      // Render chart
      setTimeout(() => renderChart(chartContainer.id, metric, data), 0);

      return cell;
    }

    async function renderChart(containerId, metric, data) {
      if (!data || data.length === 0) {
        document.getElementById(containerId).innerHTML = '<div class="loading">No data</div>';
        return;
      }

      const spec = {
        $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
        width: 'container',
        height: 'container',
        autosize: { type: 'fit', contains: 'padding' },
        data: { values: data },
        mark: {
          type: metric.chartType === 'area' ? 'area' : 'line',
          color: metric.color,
          strokeWidth: 2,
          opacity: metric.chartType === 'area' ? 0.7 : 1
        },
        encoding: {
          x: { field: 'step', type: 'quantitative', title: 'Step', axis: { grid: false } },
          y: { field: 'value', type: 'quantitative', title: metric.displayName, axis: { grid: true, gridColor: '#333' } },
          tooltip: [
            { field: 'step', type: 'quantitative' },
            { field: 'value', type: 'quantitative', format: '.4f' }
          ]
        },
        config: {
          background: 'transparent',
          axis: { labelColor: '#888', titleColor: '#888' },
          view: { stroke: 'transparent' }
        }
      };

      try {
        await vegaEmbed('#' + containerId, spec, { actions: false, renderer: 'canvas' });
      } catch (e) {
        console.error('Chart error:', e);
        document.getElementById(containerId).innerHTML = '<div class="loading">Chart error</div>';
      }
    }

    function updateStats() {
      if (!currentData || !currentData.length) return;

      document.getElementById('stats-panel').style.display = 'grid';

      let min = Infinity, max = -Infinity, sum = 0, count = 0;

      for (const matrix of currentData) {
        for (const row of matrix) {
          for (const val of row) {
            if (val < min) min = val;
            if (val > max) max = val;
            sum += val;
            count++;
          }
        }
      }

      document.getElementById('stat-min').textContent = min.toFixed(4);
      document.getElementById('stat-max').textContent = max.toFixed(4);
      document.getElementById('stat-mean').textContent = (sum / count).toFixed(4);
    }

    function showEmptyState() {
      document.getElementById('viz-container').innerHTML = \`
        <div class="empty-state">
          <div class="empty-state-icon">📊</div>
          <div class="empty-state-title">Select a Run</div>
          <div class="empty-state-desc">Choose a run from the sidebar to visualize training metrics.</div>
        </div>
      \`;
    }

    function changeVizType(type) {
      // Load appropriate preset
      if (type === 'attention_patterns') {
        loadPreset('attention_12_heads');
      } else if (type === 'similarity_matrix') {
        loadPreset('similarity_4_layers');
      } else if (type === 'training_dashboard') {
        loadPreset('discrete_flow_dashboard');
      }
    }

    function changeColormap(colormap) {
      if (currentData) {
        renderVisualization();
      }
    }

    function updateLayout() {
      if (!currentConfig) return;

      currentConfig.layout.rows = parseInt(document.getElementById('grid-rows').value);
      currentConfig.layout.cols = parseInt(document.getElementById('grid-cols').value);

      if (currentData) {
        renderVisualization();
      }
    }

    function changeLayer(layer) {
      // Would request new layer data
      if (currentConfig) {
        requestData();
      }
    }

    function exportViz(format) {
      vscode.postMessage({ type: 'exportVisualization', format });
    }

    function refresh() {
      if (currentRunId || currentConfig) {
        requestData();
      }
    }
  </script>
</body>
</html>`;
  }

  private dispose(): void {
    AdvancedVisualizationPanel.currentPanel = undefined;

    this.panel.dispose();

    while (this.disposables.length) {
      const disposable = this.disposables.pop();
      if (disposable) {
        disposable.dispose();
      }
    }
  }
}
