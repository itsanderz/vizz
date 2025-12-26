# Atlas: The Sovereign Experiment Engine

A local-first, high-performance instrumentation and analysis platform for deep learning research.

## Core Principles

- **Sovereignty:** 100% local data residency. No cloud dependencies. No rate limits.
- **Crash-Safety:** Instrumentation is non-blocking and decoupled. If the logger fails, training continues.
- **Zero-Copy Latency:** Metrics move from Python to pixels via shared memory (Apache Arrow).
- **Active Intelligence:** The system proactively analyzes data rather than passively displaying it.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VS Code Extension                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   Runs View  │  │ Live Metrics │  │      AI Insights         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Dashboard Webview                            │ │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │ │
│  │   │  Vega-Lite  │  │   Deck.gl   │  │   Tensor Surgeon    │   │ │
│  │   │   Charts    │  │  Heatmaps   │  │   (safetensors)     │   │ │
│  │   └─────────────┘  └─────────────┘  └─────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ File Watching
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         .atlas Directory                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   SQLite     │  │   DuckDB +   │  │     Safetensors          │  │
│  │  (metadata)  │  │   Parquet    │  │    (artifacts)           │  │
│  │   WAL mode   │  │ (timeseries) │  │                          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ Shared Memory (Arrow IPC)
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Python SDK                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   atlas.log  │──│ Ring Buffer  │──│      Harvester           │  │
│  │   (< 5µs)    │  │   (shared    │  │   (background process)   │  │
│  │              │  │    memory)   │  │                          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Hardware Telemetry                         │  │
│  │         (CPU, Memory, GPU via pynvml @ 1Hz)                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Python SDK

```python
import atlas_sdk as atlas

# Initialize a run
with atlas.run("my_experiment", config={"lr": 0.001, "batch_size": 32}) as run:
    for step in range(10000):
        loss = train_step()

        # Log metrics (< 5µs, non-blocking)
        run.log({"loss": loss, "accuracy": accuracy}, step=step)

        # Log tensors (saved as safetensors)
        if step % 100 == 0:
            run.log_tensor("attention_weights", attention_map, step=step)
```

### VS Code Extension

1. Install the Atlas extension
2. Open a workspace with an `.atlas` directory
3. Click the Atlas icon in the activity bar
4. View runs, metrics, and AI-generated insights

## Installation

### Python SDK

```bash
pip install atlas-sdk
```

### VS Code Extension

```bash
cd atlas-vscode
npm install
npm run compile
# Then install the .vsix file
```

## Features

### Instrumentation (Python SDK)

- **Non-blocking logging:** `atlas.log()` returns in < 5µs
- **Hardware telemetry:** Automatic GPU/CPU/memory monitoring at 1Hz
- **Tensor artifacts:** Save and explore tensors with safetensors
- **Crash safety:** Background process handles I/O independently

### Visualization (VS Code Extension)

- **Live Dashboard:** Real-time metrics with Vega-Lite charts
- **Tensor Surgeon:** Explore high-dimensional tensors slice by slice
- **Prompt-to-Plot:** Generate visualizations using natural language
- **Run Comparison:** Overlay metrics from multiple experiments

### AI Features (Planned)

- **Deep Digging:** Automatic anomaly detection (loss spikes, gradient collapse)
- **Hypothesis Generation:** VLM-powered insights from visualizations
- **Recommendation Loop:** Learn from user votes to rank insights

## Storage Format

The `.atlas` directory contains:

```
.atlas/
├── atlas.db              # SQLite (WAL mode) - metadata, votes
├── metrics_<run_id>.duckdb  # DuckDB - time series data
├── metrics/
│   └── <run_id>_*.parquet   # Parquet files for persistence
└── artifacts/
    └── <run_id>/
        ├── index.json       # Artifact index
        └── *.safetensors    # Tensor data
```

## Configuration

### Python SDK

```python
atlas.init(
    name="my_run",
    project="experiments",
    config={"lr": 0.001},
    tags=["baseline", "v1"],
    log_telemetry=True,      # Enable hardware monitoring
    telemetry_interval=1.0,  # Sampling rate in seconds
)
```

### VS Code Extension

```json
{
  "atlas.refreshInterval": 1000,
  "atlas.maxDisplayedRuns": 50,
  "atlas.enableAIInsights": true
}
```

## License

MIT
