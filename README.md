# Card-Fraud Triage: GMM Density Scoring vs Autoencoder Anomaly Detector

**Project 26 — CSAI Advanced Machine Learning (Phase 1) & Deep Learning (Phase 2)**

**Team ID:** T068  ·  **Section:** B

| # | Name | Enrollment No. |
|---|---|---|
| 1 | Sharma Piyush | 2401010437 |
| 2 | Saransh Singh Tomar | 2401010418 |
| 3 | Satwik Mani Tripathi | 2401020056 |
| 4 | Gourav Tanwar | 2401010173 |
| 5 | Alok Kumar | 2401010058 |

Who owns which part: [`docs/team_roles.md`](docs/team_roles.md)

| | |
|---|---|
| **Dataset** | [Credit Card Fraud Detection — ULB Machine Learning Group / Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 frauds) |
| **Phase 1 report** | [`reports/phase1_ml/main.pdf`](reports/phase1_ml/main.pdf) (LaTeX source alongside, IEEE-style references) |
| **Phase 2 report** | [`reports/phase2_dl/main.pdf`](reports/phase2_dl/main.pdf) (LaTeX source alongside, IEEE-style references) |

## The problem in one paragraph

Fraud is about **0.17 %** of card transactions. It changes over time, and it is labelled only weeks later. Instead of
learning "what fraud looks like" from scarce labels, we learn **what legitimate behaviour looks like**, without
showing any fraud to the models. We then send the *least normal-looking* transactions to a limited human review
queue (**triage**). We compare:
* a **Gaussian Mixture Model** density score (Phase 1, ML);
* **autoencoder** reconstruction error (Phase 2, DL);
* several **hybrid** ML+DL designs.

All models use the same chronological split and the same legitimate-only training rows. We report triage metrics:
**PR-AUC**, **recall at fixed false-positive budgets**, and **precision at a review capacity**. Never accuracy: a model
that flags nothing is 99.83 % "accurate".

## Results (test set, last 20 % of the timeline, 74 frauds)

| Model | PR-AUC [95 % CI] | Recall @ 0.1 % FPR | Recall @ 1 % FPR | Precision top-50 | Precision top-100 |
|---|---|---|---|---|---|
| **GMM** (Phase 1, K = 4, quantile inputs) | **0.686** [0.57–0.81] | 0.76 | 0.78 | 0.90 | 0.56 |
| Single Gaussian (K = 1), same inputs | **0.691** [0.58–0.80] | 0.76 | 0.77 | 0.92 | 0.56 |
| Isolation Forest | **0.050** [0.04–0.07] | 0.01 | 0.43 | 0.02 | 0.06 |
| **Undercomplete autoencoder** (Phase 2) | **0.618** [0.50–0.74] | 0.70 | 0.77 | 0.72 | 0.52 |
| **Denoising autoencoder** (Phase 2) | **0.661** [0.55–0.78] | 0.70 | 0.80 | 0.84 | 0.55 |
| **Best ML+DL hybrid** — H3 augmented GMM [x, log e] | **0.695** [0.59–0.81] | 0.76 | 0.77 | 0.92 | 0.56 |
| Supervised logistic regression † | **0.739** [0.65–0.83] | 0.77 | 0.85 | 0.88 | 0.56 |
| Supervised gradient boosting † | **0.778** [0.69–0.86] | 0.78 | 0.86 | 0.96 | 0.58 |
| Semi-supervised hybrid H5 (GBM + GMM/AE scores) † | **0.771** [0.68–0.86] | 0.76 | 0.86 | 0.94 | 0.56 |

† uses fraud labels — reference only, not a fair competitor to the label-free models. Test base rate = 0.130 %. Thresholds for the recall columns are chosen on validation and applied unchanged to test.

### Key findings

1. **Feature engineering made Phase 1 work.** A quantile (marginal-Gaussian) transform turns a useless Gaussian
   detector (PR-AUC 0.16 with standard scaling) into a strong one (0.69), and makes EM stable across random seeds.
2. **More mixture components ≠ better detection.** AIC/BIC keep asking for more components (K = 24), but K = 1–4 detect
   fraud as well or better, and larger K is less stable.
3. **Autoencoders need careful, validated design.** The default network scores ≈ 0.08 on validation. A 16-unit
   bottleneck plus Gaussian input noise (σ = 0.2) gives a **denoising autoencoder on par with the GMM** (0.661 vs 0.686,
   difference not significant) that is stable across seeds. Stacking every individually helpful change *breaks* the
   model, because factors interact.
4. **No ML+DL hybrid significantly beats the GMM.** The best hybrid (GMM on raw features + reconstruction error) ties
   it. DAGMM-style latent-space densities fail, because the encoder maps fraud to normal-looking codes: the evidence
   lives in the reconstruction residual.
5. **Labels matter most.** The label-free GMM is worth about 17 labelled frauds. With all 342 training frauds,
   supervised gradient boosting is significantly better (+0.09 PR-AUC).

**Recommendation:** deploy the label-free GMM first, with a threshold set by the review budget (0.1 % FP budget → 94
alerts in 7.7 h, 76 % of frauds caught). Switch to a supervised model once a few dozen frauds are confirmed.

## Repository structure

```
Project-26-Card-Fraud-Triage/
├── notebooks/                          ← executed on Google Colab, outputs included
│   ├── 01_EDA_and_Feature_Engineering.ipynb     data quality, EDA, chronological split, features
│   ├── 02_Phase1_GMM_Density_Scoring.ipynb      Phase 1: GMM, EM, AIC/BIC, baselines, thresholds, stability
│   ├── 03_Phase2_Autoencoder.ipynb              Phase 2: AE + denoising AE, ablations, diagnosis
│   └── 04_Phase2_Hybrid_LatentGMM.ipynb         Phase 2: hybrid ML+DL study, label efficiency, final comparison
├── figures/                            ← every figure used in the reports (generated by the notebooks)
├── results/                            ← every number used in the reports (JSON written by the notebooks)
├── reports/
│   ├── phase1_ml/   main.tex, main.pdf, numbers.tex   IEEE report, Phase 1
│   ├── phase2_dl/   main.tex, main.pdf, numbers.tex   IEEE report, Phase 2
│   ├── t068report.sty                 shared report style
│   └── references.bib
├── presentation/                       ← Phase1_ML_Presentation.pptx, Phase2_DL_Presentation.pptx
├── docs/
│   ├── literature_review.md            ML + DL literature review (32 references)
│   ├── viva_prep.md                    questions & answers for the individual viva
│   └── team_roles.md                   who owns what
├── requirements.txt
└── README.md
```

## How to run

**Google Colab (recommended).** Open a notebook from this repository in Colab (File → Open notebook → GitHub, paste
the repo URL) and choose *Runtime → Run all*.
* Every notebook is **self-contained**: it downloads the dataset itself from a public mirror and verifies its SHA-256
  checksum.
* Notebooks 01–02 need only a CPU. Notebooks 03–04 are set to use a **T4 GPU**.
* Approximate run times on Colab: 01 ≈ 2 min · 02 ≈ 35 min (CPU) · 03 ≈ 35 min (T4 GPU) · 04 ≈ 15 min (T4 GPU).

**Locally.**
```bash
pip install -r requirements.txt
```
```bash
jupyter lab notebooks/
```
The dataset (150 MB) is not stored in the repository (GitHub's file limit is 100 MB). The notebooks download it on
first run. To use a local copy, set `FRAUD_DATA=/path/to/creditcard.csv`.

**Reports.** The PDFs are already built. To rebuild, compile `reports/*/main.tex` with **XeLaTeX** (on Overleaf:
upload `reports/` and `figures/`, then set *Menu → Compiler → XeLaTeX*). All numbers in the reports are pulled from
`results/*.json` through `numbers.tex`.

## Experimental protocol (identical for every model)

1. Drop 1,081 exact duplicate rows (19 frauds) **before** splitting.
2. Order rows by `Time` and split into **train 60 % / validation 20 % / test 20 %** (past → future).
3. Feature engineering: `log(1+Amount)`, time of day on the unit circle (`hour_sin`, `hour_cos`), V1–V28 kept, raw
   `Time` dropped. Scalers are fitted on **legitimate training rows only**.
4. Unsupervised models are fitted on **legitimate training rows only**. The fraud label is used once, to remove frauds.
5. Hyper-parameters and **thresholds** are chosen on validation, then frozen and applied to test **once**.
6. Every comparison is averaged over several random seeds. Test differences are checked with a paired stratified
   bootstrap (1,000 resamples).

## Reproducibility

* Fixed seeds (`SEED = 42`, plus explicit seed lists for multi-seed studies).
* Dataset integrity: SHA-256 `76274b69…51a89`.
* Tested on Google Colab (Python 3, scikit-learn, PyTorch with CUDA) and locally (Python 3.12, scikit-learn 1.9,
  PyTorch 2.14).
