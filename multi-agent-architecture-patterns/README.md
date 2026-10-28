# Multi-Agent Architecture Patterns

Six multi-agent patterns, each run as a hands-on lab against the DAS
(Data Archival System) chatbot used throughout this training. Standalone
project: it only depends on this folder's own `archive.py` and `data/`.

| Pattern | Lab |
|---|---|
| 01 Sequential Pipeline | `labs/01_sequential_pipeline.py` |
| 01b Sequential Pipeline, with real agents ([design](docs/01b_sequential_agents.md)) | `labs/01b_sequential_agents.py` |
| 02 Coordinator + Specialists | `labs/02_coordinator_specialists.py` |
| 03 Parallel Fan-Out | `labs/03_parallel_fanout.py` |
| 04 Hierarchical Escalation | `labs/04_hierarchical_escalation.py` |
| 05 Event-Driven | `labs/05_event_driven.py` |
| 06 Blackboard / P2P | `labs/06_blackboard.py` |

## Setup

Copy `.env.example` to `.env` and fill in Nexus's URL and key:

```
NEXUS_BASE_URL=https://nexus.example.internal/v1
NEXUS_API_KEY=...
NEXUS_MODEL=MODEL-NAME
```

Every lab reads these from the environment — nothing is hardcoded.

## Run

```
uv run labs/01_sequential_pipeline.py
```

Each lab prints one trace line per agent step, so the pattern's flow is
visible while it runs.
