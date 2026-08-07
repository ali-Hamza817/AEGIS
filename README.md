# AEGIS — Agentic Evidential Geographic Intelligence for Sustainability

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/tests-34%2F34%20passing-brightgreen.svg)](tests/)
[![Architecture Subjective Logic](https://img.shields.io/badge/architecture-Subjective%20Logic-orange.svg)](src/sl/)
[![FastAPI Service](https://img.shields.io/badge/API-FastAPI-009688.svg)](src/api/)

> **Official Open-Source Implementation of the Research Manuscript:**  
> *AEGIS: Evidential Multi-Agent Orchestration for Multimodal Urban Flood Assessment*  
> **Target Journal:** *Cognitive Systems Research*, Elsevier B.V. (Impact Factor: 4.3, CiteScore 8.5)

---

## 📌 Abstract & Overview

Operational flood decision-support systems must fuse heterogeneous multi-modal streams (satellite radar, optical imagery, reanalysis climate forecasts, terrain rasters, gauge networks, and unstructured bulletins) under frequent sensor dropout and inter-source disagreement. Contemporary LLM-arbitrated agentic pipelines suffer from four critical operational vulnerabilities:

1. **Minority-Class Collapse**: Prompt-chained LLM arbitration collapses minority-class recall on severe flood categories (saturated, surface flow, inundation), producing a confident-but-wrong majority-class skew.
2. **Uncalibrated Uncertainty under Missing Modalities**: Sensor outages cause uncalibrated probability estimates or silent hallucination rather than quantified epistemic uncertainty growth.
3. **Absence of Credibility Propagation**: Unreliable or out-of-calibration sensors are weighted identically to historically accurate observation channels.
4. **Lack of Source-Level Provenance**: Standard post-hoc feature attributions explain feature importance but fail to provide an auditable trail showing which specific sensor or bulletin drove a decision.

**AEGIS** addresses these structural gaps by interfacing the formal **Subjective Logic (SL) operator algebra** with a five-agent specialist catalog (**Climate**, **Satellite**, **Land-Cover**, **Air-Quality/Gauge**, and **Document Intelligence**) over a shared four-state Dirichlet frame $\mathbb{X} = \{\theta_{\text{dry}}, \theta_{\text{saturated}}, \theta_{\text{surface}}, \theta_{\text{inundation}}\}$. A deterministic coordinator routes active agent opinions through Consensus & Compromise Fusion (**CCF**) or Weighted Belief Fusion (**WBF**) based on a Jensen-Shannon agreement signal, weighting each agent by a Brier-score-tracked reputation $\gamma$. Missing modalities trigger a partial-observable projection rule that mathematically guarantees epistemic uncertainty $u$ grows monotonically as modalities drop. The resulting 27-dimensional composite feature space is consumed by a hybrid LightGBM head, with every decision recorded in a DuckDB provenance ledger powering an interactive spatial grid dashboard.

---

## ✨ Key Innovations & Features

- **Subjective Logic Evidence Mapping**: Converts physical measurements into formal opinion tuples $\omega = (\mathbf{b}, u, \mathbf{a})$ combining belief mass vector $\mathbf{b}$, epistemic uncertainty $u$, and base-rate prior $\mathbf{a}$.
- **Monotonic Missing-Modality Projection**: Implements Kaplan's partial-observable projection rule ($\alpha_k^{\text{proj}} = C \cdot a_k$), forcing belief $b \to 0$ and epistemic uncertainty $u \to 1.0$ as modalities drop, mathematically guaranteeing $u_{k+1} \ge u_k$.
- **Adaptive Divergence-Based Routing**: Computes pairwise Jensen-Shannon (JS) divergence across active opinions; routes to CCF when $\text{JS} < \tau_{\text{low}}$ (high agreement) and to WBF when $\text{JS} \ge \tau_{\text{low}}$ (disagreement).
- **Online Brier-Score Reputation Tracking**: Continuously updates per-agent credibility score $\gamma_i \in [\gamma_{\text{min}}, 1.0]$ based on Brier score accuracy against ground-truth outcomes.
- **Hybrid Evidential Classifier**: Concatenates the 12-dimensional SL opinion tensor, 1-dimensional $u$ mass, and 15 raw multi-modal feature channels into a single LightGBM head, preserving high classification accuracy alongside calibrated uncertainty.
- **DuckDB Provenance Ledger & Leaflet UI**: Logs full input-to-output manifest IDs, base rates, divergence values, and fusion operators into a high-performance DuckDB store exposed via FastAPI and rendered on a Leaflet web UI.

---

## 📁 Repository Structure

```
AEGIS/
├── configs/
│   ├── study_site.yaml                   # Brisbane AOI spatial bounding box & grid parameters
│   └── experiments.yaml                  # Evidential hyperparameters & baseline flags
├── experiments/
│   └── run_pipeline.py                   # Master evaluation harness (data generation & benchmark runner)
├── src/
│   ├── sl/                               # Core Subjective Logic Engine (opinion algebra, operators)
│   ├── agents/                           # 5 Specialist Evidential Agents
│   ├── coordinator/                      # Deterministic SL Routing Orchestrator
│   ├── ingestion/                        # DuckDB warehouse schema & synthetic generator
│   ├── prediction/                       # Hybrid LightGBM classifier & baselines
│   ├── eval/                             # Evaluation metrics & ablations
│   └── api/                              # FastAPI REST service
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

## 📊 Empirical Benchmarks (Brisbane 2022 Fold)

Evaluated on the benchmark **Brisbane February–March 2022 flood event** ($200 \text{ spatial cells} \times 24 \text{ daily steps} = 4,800 \text{ cell-day instances}$):

| Method | F1-Macro | F1-Weighted | AUROC (ovr) | ECE (10-bin) | Latency (s) | RAM Peak (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AEGIS-SL (Full Proposed)** | **0.7190** | **0.7240** | **0.9310** | **0.043** | **14.7** | **612** |
| **BL2** (Monolithic Late Fusion) | 0.7106 | 0.7150 | 0.9189 | 0.087 | 11.2 | 588 |
| **BL1** (ERA5-Only LightGBM) | 0.5880 | 0.6210 | 0.7820 | 0.214 | 2.1 | 248 |
| **BL3** (LLM-Arbitrated Qwen2.5-3B) | 0.6238 | 0.6891 | 0.8410 | N/A *(Non-prob)* | 86.4 | 1,820 |

---

## 👥 Authors & Affiliations

- **Ali Hamza** (Corresponding Author)  
  *National University of Sciences and Technology (NUST), Islamabad, Pakistan*  
  Email: [ahamza.msse25mcs@student.nust.edu.pk](mailto:ahamza.msse25mcs@student.nust.edu.pk)  
  ORCID: [0009-0006-9790-6643](https://orcid.org/0009-0006-9790-6643)

- **Ghulam Mujtaba**  
  *National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad, Pakistan*  
  Email: [i257619@isb.nu.edu.pk](mailto:i257619@isb.nu.edu.pk)  
  ORCID: [0009-0006-3988-8088](https://orcid.org/0009-0006-3988-8088)

---

## 📜 Citation

If you find AEGIS useful in your research, please cite our manuscript:

```bibtex
@article{hamza2026aegis,
  title     = {AEGIS: Evidential Multi-Agent Orchestration for Multimodal Urban Flood Assessment},
  author    = {Hamza, Ali and Mujtaba, Ghulam},
  journal   = {Cognitive Systems Research},
  year      = {2026},
  publisher = {Elsevier B.V.}
}
```

---

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).
