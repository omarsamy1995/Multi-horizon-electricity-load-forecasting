# Weekly Individual Contribution Log

> **Project #13:** Multi-Horizon Electricity Load Forecasting on AEP Hourly Dataset  
> **Course:** Deep Learning Final Project  
> **Repository:** `Deep_Learning`

This document tracks weekly individual contributions, engineering tasks, time spent, concrete code evidence (files/commits), and problems solved. Every team member must update their section prior to weekly synchronization meetings.

---

## Team Members
- **Member 1 (Lead / Architecture):** [Full Name / Student ID / GitHub Username]
- **Member 2 (Data / Modeling):** [Full Name / Student ID / GitHub Username]
- **Member 3 (Evaluation / Web App):** [Full Name / Student ID / GitHub Username]

---

## Contribution Log Table

| Week | Team Member | Primary Task / Module | Hours Spent | Concrete Evidence (Files, Commits, PRs) | Problems Encountered & Resolution |
| :---: | :---: | :--- | :---: | :--- | :--- |
| **W1** | All | Repository Scaffolding & Problem Definition | 6h | `data/preprocess.py`, `src/datasets/aep_dataset.py`, configs | Aligning sliding window tensor dimensions (168h lookback, 24h horizon). Fixed off-by-one calendar indexing. |
| **W2** | [Member Name] | Custom PyTorch Layers & Loss Implementation | 8h | `src/models/custom_layers.py` (GRN, GLU, QuantileLoss) | Numerical stability in pinball loss and ensuring non-crossing quantile heads via Softplus offsets. |
| **W3** | [Member Name] | Baseline & Seq2Seq GRU with Attention | 10h | `src/models/baseline.py`, `src/models/seq2seq_gru.py`, `configs/` | Decoder input step conditioning on known future calendar features rather than autoregressive feedback alone. |
| **W4** | [Member Name] | Time-Series Transformer & TFT Advanced Model | 12h | `src/models/transformer.py`, `src/models/tft_advanced.py` | Causal masking in future horizon decoder self-attention to avoid temporal lookahead leakage. |
| **W5** | [Member Name] | Multi-Seed Training Harness & Metrics | 8h | `train.py`, `evaluate.py`, `experiments/` | Logging exact parameters, memory size, and hardware profile to complete Table 7. |
| **W6** | [Member Name] | Calendar Feature Ablation & Error Analysis | 8h | `evaluate.py --ablation_no_calendar`, horizon error plots | Measuring degradation when future calendar signals are masked. Quantifying peak load errors. |
| **W7** | [Member Name] | Interactive Streamlit Dashboard & Deployment | 7h | `demo/app.py`, Plotly confidence ribbon | Seamless rendering of 168-hour historical context + 24-hour predictive uncertainty band. |
| **W8** | All | Final Technical Report, Slide Deck & Polish | 10h | `report/final_report.pdf`, presentation slides | Synthesizing empirical results across the 3 random seeds into final report tables. |

---

## Instructions for Updating this Log
1. **Be Honest and Specific:** List the exact module or issue worked on rather than generic descriptions like "did coding".
2. **Commit Evidence:** Include the git commit hash or relative file path modified.
3. **Document Blockers:** Explicitly mention bugs or blockers solved, such as GPU OOM, convergence issues, or data leakage bugs.
