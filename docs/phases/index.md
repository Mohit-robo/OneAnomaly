# OneAnomaly — Execution & Integration Guide

This directory contains step-by-step operational guides for each phase of the OneAnomaly pipeline. Each guide covers intent, UI interactions, verification criteria, expected outputs, and error references.

---

## Phase Index

| Phase | File | Purpose |
|---|---|---|
| 0 | [00_setup_and_launch.md](./00_setup_and_launch.md) | Environment validation, server startup, pre-flight checks |
| 1 | [01_preprocessing.md](./01_preprocessing.md) | Configure image preprocessing (thresholding or background subtraction) |
| 2 | [02_spatial_regions.md](./02_spatial_regions.md) | Define spatial inspection zones and export TensorRT engine |
| 3 | [03_memory_bank.md](./03_memory_bank.md) | Build or load the FAISS feature memory bank from good samples |
| 4 | [04_detection_analysis.md](./04_detection_analysis.md) | Run inference and inspect anomaly heatmap results |

---

## Quick Reference: Stage Dependency Chain

```
Phase 0: Server Running
        ↓
Phase 1: Preprocessing Config Saved        → outputs: preprocess_config (in-memory)
        ↓
Phase 2: Spatial Regions + Engine Selected → outputs: spatial_config, .engine file
        ↓
Phase 3: Memory Bank Built or Loaded       → outputs: .pkl FAISS bank in memory_banks/
        ↓
Phase 4: Detection + Analysis              → outputs: session artifacts in outputs/sessions/
```

**Each stage is gated**: you cannot proceed to the next stage until the current one is saved and validated.

---

## General Principles

- All configuration is held **in Flask server global state** (`preprocess_config`, `spatial_config`, `memory_bank`). Restarting the server resets all state.
- Session artifacts are stored in `outputs/sessions/<session_id>/` and persist across server restarts.
- The backend logs all major steps — the **Live Terminal** in the right inspector panel shows them in real time.
- If the UI shows a stage as locked (greyed badge), the prerequisite stage has not been saved yet.
