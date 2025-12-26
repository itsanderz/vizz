# Atlas: The Sovereign Experiment Engine

<p align="center">
  <strong>Enterprise-Grade ML Experiment Tracking • $10M Budget • Local-First • Zero Cloud Costs</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#enterprise">Enterprise</a> •
  <a href="#integrations">Integrations</a>
</p>

---

## Why Atlas?

Atlas is the definitive "Formula 1 class" instrumentation and analysis platform for deep learning research. Unlike cloud-based alternatives:

- **100% Local Data** - Your data never leaves your infrastructure
- **Zero Rate Limits** - Log as fast as your hardware allows
- **Zero Cloud Costs** - No per-user, per-metric, or per-artifact pricing
- **Crash-Safe** - Logger failures never interrupt training
- **Sub-5µs Latency** - Non-blocking logging via shared memory

## Quick Start

### Installation

```bash
pip install atlas-sdk

# With framework integrations
pip install atlas-sdk[torch,lightning,huggingface]

# Full enterprise installation
pip install atlas-sdk[all]
```

### Basic Usage

```python
import atlas_sdk as atlas

# Start a run
with atlas.run("gpt4-finetune", config={"lr": 1e-4, "batch_size": 32}) as run:
    for step in range(10000):
        loss = train_step()
        run.log({"loss": loss, "accuracy": accuracy}, step=step)

        # Log tensors (saved as safetensors)
        if step % 100 == 0:
            run.log_tensor("attention_weights", attn, step=step)
```

### Framework Integrations

**PyTorch Lightning:**
```python
from atlas_sdk.integrations import AtlasLightningCallback

trainer = Trainer(
    callbacks=[AtlasLightningCallback(project="llm-research")]
)
```

**Hugging Face Transformers:**
```python
from atlas_sdk.integrations import AtlasTrainerCallback

trainer = Trainer(
    model=model,
    callbacks=[AtlasTrainerCallback(project="bert-finetuning")]
)
```

**JAX/Flax:**
```python
from atlas_sdk.integrations import AtlasJAXLogger

with AtlasJAXLogger(project="jax-training") as logger:
    for step, batch in enumerate(dataloader):
        state, loss = train_step(state, batch)
        logger.log_step(step, {"loss": loss}, params=state.params)
```

## Features

### High-Performance Instrumentation

- **Lock-Free Ring Buffer** - Shared memory via Apache Arrow
- **Background Harvester** - Separate process for disk I/O
- **Zero-Copy Tensors** - Safetensors for memory-mapped access
- **Hardware Telemetry** - GPU/CPU/Memory at 1Hz

### Enterprise Visualization

- **VS Code Extension** - Native IDE integration
- **Vega-Lite Charts** - Declarative, interactive visualizations
- **Tensor Surgeon** - Explore 4GB+ attention maps
- **Prompt-to-Plot** - Natural language chart generation

### Deep Digging AI

- **Anomaly Detection** - Loss spikes, gradient collapse, NaN detection
- **Hypothesis Generation** - AI-powered training insights
- **Pattern Recognition** - Convergence, divergence, plateau detection
- **Recommendation Loop** - Learn from user feedback

### Enterprise Security

- **Audit Logging** - Tamper-evident event chain
- **Data Encryption** - AES-256-GCM at rest
- **Compliance** - GDPR, HIPAA, SOC 2 support
- **PII Detection** - Automatic sensitive data scanning

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
                                    │ File Watching + Arrow IPC
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
│  │                    Deep Digging Agent                         │  │
│  │    (Anomaly Detection • Pattern Recognition • Insights)       │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## CLI

```bash
# List runs
atlas runs list --project my-project

# Show run details
atlas runs show <run_id>

# Compare runs
atlas runs compare <run_id_1> <run_id_2> --metric loss

# Query metrics (SQL)
atlas metrics query "SELECT AVG(value) FROM metrics WHERE name = 'loss'"

# Generate report
atlas report generate <run_id> --output report.md

# Compliance check
atlas compliance check --standard gdpr --standard hipaa

# Verify audit log integrity
atlas audit verify
```

## Enterprise Configuration

```python
import atlas_sdk as atlas

atlas.configure(
    project_name="llm-research",
    environment="production",

    # Enterprise settings
    enterprise={
        "organization_id": "org-123",
        "audit_logging_enabled": True,
        "data_retention_days": 365,
        "gdpr_mode": True,
    },

    # AI settings
    ai={
        "anomaly_detection_enabled": True,
        "anomaly_sensitivity": 0.8,
        "insight_generation_enabled": True,
    },
)
```

## Deep Digging Agent

```python
from atlas_sdk.analysis import DeepDiggingAgent

# Start the autonomous agent
agent = DeepDiggingAgent(
    storage_engine=storage,
    analysis_interval=30.0,
    sensitivity=0.8,
)

agent.on_anomaly(lambda a: print(f"Anomaly: {a.description}"))
agent.on_insight(lambda i: print(f"Insight: {i.title}"))
agent.start()
```

## Comparison

| Feature | Atlas | Weights & Biases | MLflow |
|---------|-------|-----------------|--------|
| Local-First | Yes | No | Partial |
| Zero Cloud Cost | Yes | No | Partial |
| Sub-5µs Logging | Yes | No | No |
| Crash-Safe | Yes | Partial | Partial |
| IDE Integration | Yes | No | No |
| AI Insights | Yes | Partial | No |
| Enterprise Security | Yes | Yes | Partial |

## License

MIT
