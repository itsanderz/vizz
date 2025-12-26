/**
 * Tensor Surgeon Panel - Tensor exploration and visualization.
 *
 * Uses Deck.gl for WebGL-accelerated rendering of large tensors.
 */

import * as vscode from "vscode";
import { AtlasStorage, Artifact } from "../storage/AtlasStorage";

export class TensorSurgeonPanel {
  public static currentPanel: TensorSurgeonPanel | undefined;
  private static readonly viewType = "atlas.tensorSurgeon";

  private readonly panel: vscode.WebviewPanel;
  private readonly extensionUri: vscode.Uri;
  private storage: AtlasStorage;
  private disposables: vscode.Disposable[] = [];

  public static createOrShow(
    extensionUri: vscode.Uri,
    storage: AtlasStorage
  ): void {
    const column = vscode.ViewColumn.Beside;

    if (TensorSurgeonPanel.currentPanel) {
      TensorSurgeonPanel.currentPanel.panel.reveal(column);
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      TensorSurgeonPanel.viewType,
      "Tensor Surgeon",
      column,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [extensionUri],
      }
    );

    TensorSurgeonPanel.currentPanel = new TensorSurgeonPanel(
      panel,
      extensionUri,
      storage
    );
  }

  public static openTensor(path: string): void {
    if (TensorSurgeonPanel.currentPanel) {
      TensorSurgeonPanel.currentPanel.loadTensor(path);
    }
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

    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);

    this.panel.webview.onDidReceiveMessage(
      (message) => this.handleMessage(message),
      null,
      this.disposables
    );
  }

  private async loadTensor(path: string): Promise<void> {
    this.panel.webview.postMessage({
      type: "loadTensor",
      path,
    });
  }

  private async handleMessage(message: any): Promise<void> {
    switch (message.type) {
      case "ready":
        await this.sendArtifactsList();
        break;

      case "loadTensor":
        // In a real implementation, this would use a native module to read safetensors
        // For now, we just acknowledge the request
        this.panel.webview.postMessage({
          type: "tensorInfo",
          path: message.path,
          info: {
            name: "attention_weights",
            shape: [12, 64, 64],
            dtype: "float32",
          },
        });
        break;

      case "getSlice":
        // Would use native safetensors reader
        this.panel.webview.postMessage({
          type: "tensorSlice",
          slice: message.slice,
          data: this.generateDemoData(message.shape || [64, 64]),
        });
        break;
    }
  }

  private async sendArtifactsList(): Promise<void> {
    const runs = await this.storage.listRuns({ limit: 20 });
    const allArtifacts: (Artifact & { runName: string })[] = [];

    for (const run of runs) {
      const artifacts = await this.storage.getArtifacts(run.id);
      for (const artifact of artifacts) {
        if (
          artifact.artifact_type === "tensor" ||
          artifact.artifact_type === "checkpoint"
        ) {
          allArtifacts.push({ ...artifact, runName: run.name });
        }
      }
    }

    this.panel.webview.postMessage({
      type: "artifacts",
      artifacts: allArtifacts,
    });
  }

  private generateDemoData(shape: number[]): number[][] {
    // Generate demo heatmap data for visualization
    const [rows, cols] = shape;
    const data: number[][] = [];

    for (let i = 0; i < rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        // Create an interesting pattern
        const value =
          Math.sin((i / rows) * Math.PI * 2) *
            Math.cos((j / cols) * Math.PI * 2) +
          Math.random() * 0.2;
        row.push(value);
      }
      data.push(row);
    }

    return data;
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
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' https://unpkg.com; style-src 'unsafe-inline';">
  <title>Tensor Surgeon</title>
  <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
  <style>
    :root {
      --bg-primary: var(--vscode-editor-background);
      --bg-secondary: var(--vscode-sideBar-background);
      --text-primary: var(--vscode-editor-foreground);
      --text-secondary: var(--vscode-descriptionForeground);
      --border-color: var(--vscode-panel-border);
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
      height: 100vh;
      display: flex;
    }

    .sidebar {
      width: 250px;
      background: var(--bg-secondary);
      border-right: 1px solid var(--border-color);
      padding: 16px;
      overflow-y: auto;
    }

    .sidebar h2 {
      font-size: 14px;
      margin-bottom: 16px;
    }

    .artifact-item {
      padding: 8px;
      border-radius: 4px;
      cursor: pointer;
      margin-bottom: 4px;
      font-size: 12px;
    }

    .artifact-item:hover {
      background: var(--vscode-list-hoverBackground);
    }

    .artifact-item.active {
      background: var(--vscode-list-activeSelectionBackground);
    }

    .artifact-name {
      font-weight: 600;
    }

    .artifact-meta {
      color: var(--text-secondary);
      font-size: 10px;
    }

    .main {
      flex: 1;
      display: flex;
      flex-direction: column;
    }

    .controls {
      padding: 12px 16px;
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      gap: 16px;
      align-items: center;
    }

    .control-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .control-group label {
      font-size: 12px;
      color: var(--text-secondary);
    }

    .control-group input, .control-group select {
      background: var(--bg-primary);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 12px;
    }

    .viewer {
      flex: 1;
      position: relative;
    }

    #canvas-container {
      width: 100%;
      height: 100%;
    }

    .info-panel {
      position: absolute;
      top: 16px;
      right: 16px;
      background: var(--bg-secondary);
      border-radius: 8px;
      padding: 12px;
      font-size: 12px;
      min-width: 200px;
    }

    .info-panel h3 {
      font-size: 13px;
      margin-bottom: 8px;
    }

    .info-row {
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
    }

    .info-label {
      color: var(--text-secondary);
    }

    .colorbar {
      position: absolute;
      bottom: 16px;
      right: 16px;
      background: var(--bg-secondary);
      border-radius: 4px;
      padding: 8px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .colorbar-gradient {
      width: 150px;
      height: 12px;
      background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000);
      border-radius: 2px;
    }

    .colorbar-labels {
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: var(--text-secondary);
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
  <div class="sidebar">
    <h2>Tensor Artifacts</h2>
    <div id="artifacts-list">
      <div class="loading">Loading...</div>
    </div>
  </div>

  <div class="main">
    <div class="controls">
      <div class="control-group">
        <label>Dimension:</label>
        <select id="dim-select">
          <option value="0">Dim 0 (Heads)</option>
          <option value="1">Dim 1 (Seq)</option>
          <option value="2">Dim 2 (Seq)</option>
        </select>
      </div>
      <div class="control-group">
        <label>Slice:</label>
        <input type="range" id="slice-slider" min="0" max="11" value="0">
        <span id="slice-value">0</span>
      </div>
      <div class="control-group">
        <label>Scale:</label>
        <select id="scale-select">
          <option value="linear">Linear</option>
          <option value="log">Logarithmic</option>
          <option value="sqrt">Square Root</option>
        </select>
      </div>
    </div>

    <div class="viewer">
      <div id="canvas-container">
        <div class="loading">Select a tensor to visualize</div>
      </div>

      <div class="info-panel" id="info-panel" style="display: none;">
        <h3>Tensor Info</h3>
        <div class="info-row">
          <span class="info-label">Name:</span>
          <span id="info-name">-</span>
        </div>
        <div class="info-row">
          <span class="info-label">Shape:</span>
          <span id="info-shape">-</span>
        </div>
        <div class="info-row">
          <span class="info-label">Dtype:</span>
          <span id="info-dtype">-</span>
        </div>
        <div class="info-row">
          <span class="info-label">Min:</span>
          <span id="info-min">-</span>
        </div>
        <div class="info-row">
          <span class="info-label">Max:</span>
          <span id="info-max">-</span>
        </div>
        <div class="info-row">
          <span class="info-label">Mean:</span>
          <span id="info-mean">-</span>
        </div>
      </div>

      <div class="colorbar">
        <span style="font-size: 10px;">Min</span>
        <div class="colorbar-gradient"></div>
        <span style="font-size: 10px;">Max</span>
      </div>
    </div>
  </div>

  <script>
    const vscode = acquireVsCodeApi();
    let currentTensor = null;
    let currentData = null;
    let deckInstance = null;

    vscode.postMessage({ type: 'ready' });

    window.addEventListener('message', (event) => {
      const message = event.data;

      switch (message.type) {
        case 'artifacts':
          renderArtifactsList(message.artifacts);
          break;

        case 'tensorInfo':
          currentTensor = message.info;
          updateInfoPanel(message.info);
          requestSlice(0);
          break;

        case 'tensorSlice':
          currentData = message.data;
          renderHeatmap(message.data);
          break;
      }
    });

    function renderArtifactsList(artifacts) {
      const container = document.getElementById('artifacts-list');

      if (!artifacts || artifacts.length === 0) {
        container.innerHTML = '<div class="loading">No tensor artifacts</div>';
        return;
      }

      container.innerHTML = artifacts.map(artifact => \`
        <div class="artifact-item" onclick="loadTensor('\${artifact.path}')">
          <div class="artifact-name">\${artifact.name}</div>
          <div class="artifact-meta">
            \${artifact.runName} | \${formatBytes(artifact.size_bytes)}
          </div>
        </div>
      \`).join('');
    }

    function loadTensor(path) {
      vscode.postMessage({ type: 'loadTensor', path });

      // Update active state
      document.querySelectorAll('.artifact-item').forEach(el => {
        el.classList.remove('active');
      });
      event.target.closest('.artifact-item').classList.add('active');
    }

    function updateInfoPanel(info) {
      document.getElementById('info-panel').style.display = 'block';
      document.getElementById('info-name').textContent = info.name;
      document.getElementById('info-shape').textContent = JSON.stringify(info.shape);
      document.getElementById('info-dtype').textContent = info.dtype;

      // Update slider
      const slider = document.getElementById('slice-slider');
      slider.max = info.shape[0] - 1;
      slider.value = 0;
      document.getElementById('slice-value').textContent = '0';
    }

    function requestSlice(index) {
      if (!currentTensor) return;

      vscode.postMessage({
        type: 'getSlice',
        slice: index,
        shape: [currentTensor.shape[1], currentTensor.shape[2]]
      });
    }

    function renderHeatmap(data) {
      const container = document.getElementById('canvas-container');
      container.innerHTML = '<canvas id="heatmap-canvas"></canvas>';

      const canvas = document.getElementById('heatmap-canvas');
      const ctx = canvas.getContext('2d');

      const rows = data.length;
      const cols = data[0].length;

      // Set canvas size
      const cellSize = Math.min(
        Math.floor(container.clientWidth / cols),
        Math.floor(container.clientHeight / rows),
        10
      );

      canvas.width = cols * cellSize;
      canvas.height = rows * cellSize;

      // Find min/max for normalization
      let min = Infinity, max = -Infinity, sum = 0;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const v = data[i][j];
          if (v < min) min = v;
          if (v > max) max = v;
          sum += v;
        }
      }

      // Update stats
      document.getElementById('info-min').textContent = min.toFixed(4);
      document.getElementById('info-max').textContent = max.toFixed(4);
      document.getElementById('info-mean').textContent = (sum / (rows * cols)).toFixed(4);

      // Draw heatmap
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const normalized = (data[i][j] - min) / (max - min);
          ctx.fillStyle = valueToColor(normalized);
          ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
        }
      }
    }

    function valueToColor(value) {
      // Viridis-like colormap
      const colors = [
        [68, 1, 84],
        [72, 40, 120],
        [62, 74, 137],
        [49, 104, 142],
        [38, 130, 142],
        [31, 158, 137],
        [53, 183, 121],
        [109, 205, 89],
        [180, 222, 44],
        [253, 231, 37]
      ];

      const idx = Math.min(Math.floor(value * (colors.length - 1)), colors.length - 2);
      const t = (value * (colors.length - 1)) - idx;

      const r = Math.round(colors[idx][0] + t * (colors[idx + 1][0] - colors[idx][0]));
      const g = Math.round(colors[idx][1] + t * (colors[idx + 1][1] - colors[idx][1]));
      const b = Math.round(colors[idx][2] + t * (colors[idx + 1][2] - colors[idx][2]));

      return \`rgb(\${r}, \${g}, \${b})\`;
    }

    function formatBytes(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // Event listeners
    document.getElementById('slice-slider').addEventListener('input', (e) => {
      document.getElementById('slice-value').textContent = e.target.value;
      requestSlice(parseInt(e.target.value));
    });

    document.getElementById('scale-select').addEventListener('change', () => {
      if (currentData) {
        renderHeatmap(currentData);
      }
    });
  </script>
</body>
</html>`;
  }

  private dispose(): void {
    TensorSurgeonPanel.currentPanel = undefined;

    this.panel.dispose();

    while (this.disposables.length) {
      const disposable = this.disposables.pop();
      if (disposable) {
        disposable.dispose();
      }
    }
  }
}
