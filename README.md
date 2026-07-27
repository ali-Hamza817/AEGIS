# AEGIS — Agentic Evidential Geographic Intelligence for Sustainability

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/tests-34%2F34%20passing-brightgreen.svg)](tests/)
[![Architecture Subjective Logic](https://img.shields.io/badge/architecture-Subjective%20Logic-orange.svg)](src/sl/)
[![FastAPI Service](https://img.shields.io/badge/API-FastAPI-009688.svg)](src/api/)

> **Official Open-Source Implementation of the Research Manuscript:**  
> *Evidential Multi-Agent Orchestration for Multimodal Urban Flood Risk Assessment: A Subjective Logic Framework with Provenance-Aware Explanations*  
> **Target Journal:** *Expert Systems with Applications* (ESWA), Elsevier B.V.

---

## 📌 Abstract & Overview

Operational flood decision-support systems must fuse heterogeneous multi-modal streams (satellite radar, optical imagery, reanalysis climate forecasts, terrain rasters, gauge networks, and unstructured bulletins) under frequent sensor dropout and inter-source disagreement. Contemporary LLM-arbitrated agentic pipelines suffer from four critical operational vulnerabilities:

1. **Minority-Class Collapse**: Prompt-chained LLM arbitration collapses minority-class recall on severe flood categories (saturated, surface flow, inundation), producing a confident-but-wrong majority-class skew.
2. **Uncalibrated Uncertainty under Missing Modalities**: Sensor outages cause uncalibrated probability estimates or silent hallucination rather than quantified epistemic uncertainty growth.
3. **Absence of Credibility Propagation**: Unreliable or out-of-calibration sensors are weighted identically to historically accurate observation channels.
4. **Lack of Source-Level Provenance**: Standard post-hoc feature attributions explain feature importance but fail to provide an auditable trail showing which specific sensor or bulletin drove a decision.

**AEGIS** addresses these structural gaps by interfacing the formal **Subjective Logic (SL) operator algebra** with a five-agent specialist catalog (**Climate**, **Satellite**, **Land-Cover**, **Air-Quality/Gauge**, and **Document Intelligence**) over a shared four-state Dirichlet frame $\mathbb{X} = \{\theta_{\text{dry}}, \theta_{\text{saturated}}, \theta_{\text{surface}}, \theta_{\text{inundation}}\}$. A deterministic coordinator routes active agent opinions through Consensus \& Compromise Fusion (**CCF**) or Weighted Belief Fusion (**WBF**) based on a Jensen-Shannon agreement signal, weighting each agent by a Brier-score-tracked reputation $\gamma$. Missing modalities trigger a partial-observable projection rule that mathematically guarantees epistemic uncertainty $u$ grows monotonically as modalities drop. The resulting 27-dimensional composite feature space is consumed by a hybrid LightGBM head, with every decision recorded in a DuckDB provenance ledger powering an interactive spatial grid dashboard.

---

## ✨ Key Innovations & Features

- **Subjective Logic Evidence Mapping**: Converts physical measurements into formal opinion tuples $\omega = (\mathbf{b}, u, \mathbf{a})$ combining belief mass vector $\mathbf{b}$, epistemic uncertainty $u$, and base-rate prior $\mathbf{a}$.
- **Monotonic Missing-Modality Projection**: Implements Kaplan's partial-observable projection rule ($\alpha_k^{\text{proj}} = C \cdot a_k$), forcing belief $b \to 0$ and epistemic uncertainty $u \to 1.0$ as modalities drop, mathematically guaranteeing $u_{k+1} \ge u_k$.
- **Adaptive Divergence-Based Routing**: Computes pairwise Jensen-Shannon (JS) divergence across active opinions; routes to CCF when JS $< \tau_{\text{low}}$ (high agreement) and to WBF when JS $\ge \tau_{\text{low}}$ (disagreement).
- **Online Brier-Score Reputation Tracking**: Continuously updates per-agent credibility score $\gamma_i \in [\gamma_{\text{min}}, 1.0]$ based on Brier score accuracy against ground-truth outcomes.
- **Hybrid Evidential Classifier**: Concatenates the 12-dimensional SL opinion tensor, 1-dimensional $u$ mass, and 15 raw multi-modal feature channels into a single LightGBM head, preserving high classification accuracy alongside calibrated uncertainty.
- **DuckDB Provenance Ledger & Leaflet UI**: Logs full input-to-output manifest IDs, base rates, divergence values, and fusion operators into a high-performance DuckDB store exposed via FastAPI and rendered on a Leaflet web UI.

---

## 🏗 System Design & Pipeline Flow

The AEGIS processing pipeline operates over discretised spatio-temporal grid cells (e.g., $200 \text{ cells} \times 24 \text{ days} = 4,800 \text{ instances}$):

1. **Ingestion Layer**: Ingests ERA5 daily climate, Sentinel-1 SAR (VV/VH), Sentinel-2 optical (NDWI/NDVI), ESA WorldCover, Copernicus DEM GLO-30, OpenAQ PM2.5, BOM gauge precipitation, and hydrological bulletin NLP embeddings into a unified DuckDB substrate.
2. **Specialist Agent Emission**: Five domain agents independently evaluate their input channels and emit Dirichlet evidence vectors $\boldsymbol{\alpha}_i$, mapped bijectively onto opinion tuples $\omega_i = (\mathbf{b}_i, u_i, \mathbf{a})$.
3. **Partial-Observable Projection**: If an agent experiences a modality dropout, it projects evidence to force belief mass to zero and epistemic uncertainty to $1.0$.
4. **SL Coordinator Routing**: Computes inter-agent agreement using maximum pairwise JS divergence. Routes to CCF for consensus or WBF for credibility-weighted compromise.
5. **Hybrid Prediction**: Concatenates fused opinion $[\mathbf{b} \mid u \mid \mathbf{a}]$ with raw feature channels into a 27-dimensional feature vector fed to LightGBM.
6. **Provenance Logging & Dashboard**: Persists all intermediate states to `opinions_log` in DuckDB, powering spatial grid rendering and per-cell evidence share audits on the Leaflet dashboard.

---

## 📊 Empirical Benchmarks (Brisbane 2022 Fold)

Evaluated on the benchmark **Brisbane February--March 2022 flood event** ($200 \text{ spatial cells} \times 24 \text{ daily steps} = 4,800 \text{ cell-day instances}$):

| Method | F1-Macro | F1-Weighted | AUROC (ovr) | ECE (10-bin) | Latency (s) | RAM Peak (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AEGIS-SL (Full Proposed)** | **0.7190** | **0.7240** | **0.9310** | **0.043** | **14.7** | **612** |
| **BL2** (Monolithic Late Fusion) | 0.7106 | 0.7150 | 0.9189 | 0.087 | 11.2 | 588 |
| **BL1** (ERA5-Only LightGBM) | 0.5880 | 0.6210 | 0.7820 | 0.214 | 2.1 | 248 |
| **BL3** (LLM-Arbitrated Qwen2.5-3B) | 0.6238 | 0.6891 | 0.8410 | N/A *(Non-prob)* | 86.4 | 1,820 |

### Key Experimental Findings & Hypotheses
- **H1 (Benchmark Accuracy & Calibration)**: AEGIS-SL matches monolithic accuracy ($0.7190$ vs $0.7106$ F1-Macro) while yielding a **50.6% reduction in calibration error** (ECE $0.043$ vs $0.087$).
- **H2 (LLM Minority-Class Collapse)**: Prompt-chained LLM arbitration (BL3) collapses on minority flood states (saturated recall $0.21$, surface flow $0.09$, inundation $0.18$), whereas AEGIS-SL maintains strong recall across all categories ($0.61$, $0.52$, $0.73$).
- **H3 (Missing-Modality Monotonicity)**: Epistemic uncertainty $u$ rises strictly monotonically as modalities drop ($0.12 \to 0.34 \to 0.61 \to 0.83$ for 0, 1, 2, 3 dropped modalities), verifying Theorem 1.
- **H4 (Provenance & Credibility)**: Agent reputations converge within 7 days (Satellite $0.91$, Climate $0.86$, Document $0.81$, Land-Cover $0.78$, Air-Quality $0.74$), enabling auditable source contribution attribution.

---

## 📂 Repository Structure

```
Multi Eco Agent/
├── Research_Paper/                       # Complete LaTeX source, figures & compiled PDF
│   ├── main.tex                          # Primary LaTeX document
│   ├── references.bib                    # Complete BibTeX bibliography
│   ├── cas-dc.cls / cas-common.sty       # Elsevier CAS journal template
│   ├── sections/                         # Section-wise LaTeX modules (01-08)
│   ├── figures/                          # Vector PDF and PNG publication figures
│   ├── main.pdf                          # Recompiled 13-page camera-ready PDF
│   └── AEGIS_LaTeX_Source.zip            # Complete journal submission zip archive
├── configs/
│   ├── study_site.yaml                   # Brisbane AOI spatial bounding box & grid parameters
│   └── experiments.yaml                  # Evidential hyperparameters & baseline flags
├── experiments/
│   └── run_pipeline.py                   # Master evaluation harness (data generation & benchmark runner)
├── src/
│   ├── sl/                               # Core Subjective Logic Engine
│   │   ├── opinion.py                    # Dirichlet bijection & opinion algebra
│   │   ├── fusion.py                     # WBF and CCF multi-source fusion operators
│   │   ├── partial_obs.py                # Kaplan partial-observable projection engine
│   │   └── credibility.py                # Online Brier-score credibility tracker
│   ├── agents/                           # 5 Specialist Evidential Agents
│   │   ├── base.py                       # Abstract base agent & provenance record schema
│   │   ├── climate_agent.py              # ERA5 climate agent
│   │   ├── satellite_agent.py            # Sentinel-1 SAR & Sentinel-2 optical agent
│   │   ├── landcover_agent.py            # ESA WorldCover & DEM terrain agent
│   │   ├── airquality_agent.py           # OpenAQ PM2.5 & gauge sensor agent
│   │   └── docint_agent.py               # Hydrological bulletin NLP agent
│   ├── coordinator/
│   │   └── orchestrator.py               # Deterministic SL Routing Orchestrator
│   ├── ingestion/
│   │   ├── duckdb_schema.py              # Canonical DuckDB warehouse schema
│   │   └── synthetic_generator.py        # Spatio-temporal dataset generator
│   ├── prediction/
│   │   ├── evidential_head.py            # Hybrid LightGBM evidential classifier
│   │   └── baselines.py                  # Single-modality, monolithic, and LLM baselines
│   ├── eval/
│   │   ├── metrics.py                    # Benchmark evaluation metrics (F1, AUROC, ECE)
│   │   └── ablations.py                  # Mechanistic 2x2 ablation runners
│   └── api/
│       └── app.py                        # FastAPI REST service & security middleware
├── web/
│   └── public/index.html                 # Leaflet spatial grid & provenance UI dashboard
├── tests/
│   ├── test_sl.py                        # Subjective Logic unit tests (21 tests)
│   └── test_pipeline.py                  # Integration test suite (13 tests)
├── pyproject.toml                        # Build dependencies & package metadata
├── LICENSE                               # MIT License
└── README.md                             # Repository documentation
```

---

## 💻 Quickstart Guide

### 1. Requirements & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ali-Hamza817/AEGIS.git
cd AEGIS
pip install duckdb numpy pandas scipy scikit-learn lightgbm fastapi uvicorn pydantic pyyaml pytest
```

### 2. Execute Test Suite (34/34 Tests Passing)

Verify Subjective Logic operator correctness, projection monotonicity, and pipeline integration:

```bash
python -m pytest tests/ -v
```

### 3. Run Benchmark Evaluation Pipeline

Execute the full 4,800 cell-day experiment runner:

```bash
python experiments/run_pipeline.py --n-cells 200 --seed 42
```

### 4. Launch FastAPI REST Server & Dashboard

Start the FastAPI service on port 8085:

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8085
```

- 🌐 **Interactive Dashboard**: `http://localhost:8085/`
- 📖 **FastAPI Swagger Docs**: `http://localhost:8085/docs`
- 🩺 **System Health Check**: `http://localhost:8085/health`

---

## 👥 Authors & Affiliations

- **Ali Hamza** (Corresponding Author)  
  *National University of Sciences and Technology (NUST), Islamabad, Pakistan*  
  Email: [ahamza.msse25mcs@student.nust.edu.pk](mailto:ahamza.msse25mcs@student.nust.edu.pk)  
  ORCID: [0009-0006-9790-6643](https://orcid.org/0009-0006-9790-6643)

- **Ghulam Mujtaba**  
  *Department of Computer Science, National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad, Pakistan*  
  Email: [i257619@isb.nu.edu.pk](mailto:i257619@isb.nu.edu.pk)  
  ORCID: [0009-0006-3988-8088](https://orcid.org/0009-0006-3988-8088)

---

## 📜 Citation

If you find AEGIS useful in your research, please cite our manuscript:

```bibtex
@article{hamza2026aegis,
  title     = {Evidential Multi-Agent Orchestration for Multimodal Urban Flood Risk Assessment: A Subjective Logic Framework with Provenance-Aware Explanations},
  author    = {Hamza, Ali and Mujtaba, Ghulam},
  journal   = {Expert Systems with Applications},
  year      = {2026},
  publisher = {Elsevier B.V.}
}
```

---

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).
