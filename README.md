# Artifact Description: Stable-Rank–Based Analysis Examples

This repository provides **illustrative artifacts** used to explain how the stable-rank metric behaves under fault injection and how threshold choices affect anomaly detection.  
The content is intentionally minimal and focuses on **metric interpretation**, rather than large-scale data collection or end-to-end pipelines.

## Directory Structure

### `rawdata/` — Fault-Injection Outputs

This directory contains **model outputs after fault injection**.

- Each file corresponds to an inference result affected by injected SEU faults.
- No additional preprocessing or aggregation is applied.
- These outputs serve as the **input basis** for computing stable-rank metrics shown in later examples.

---

### `stableranksp/` — Stable-Rank Visualization Examples

This directory contains **designed visualization examples** illustrating the behavior of the stable-rank metric.

- Includes line plots comparing **fault-free inference** and **fault-injected inference**.
- Demonstrates how stable rank evolves differently when faults induce abnormal or unstable behavior.

---

### `threshold/` — Threshold-Based Decision Comparison

This directory compares **decision outcomes under different stable-rank thresholds**.

- Evaluates whether abnormal inference can be **correctly identified** at different threshold settings.
- Highlights how threshold choice influences **detection reliability**.
- Supports the discussion on **threshold sensitivity** in stability-aware anomaly judgment.


