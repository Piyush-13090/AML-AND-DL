# Team Roles and Ownership

**Team ID:** T068 · **Section:** B · **Project 26:** Card-Fraud Triage: GMM Density Scoring vs Autoencoder Anomaly
Detector

The viva is **individual**. Every member must be able to explain the *whole* pipeline, but each member **owns** one
area in depth and presents it. The same split appears in the "Team Contributions" appendix of both reports.

| # | Name | Enrollment No. | Phase 1 (ML) — owns and presents | Phase 2 (DL) — owns and presents | Main files |
|---|---|---|---|---|---|
| 1 | Sharma Piyush | 2401010437 | Data quality, EDA, chronological split, leakage control | DL data handling: legitimate-only training, label-free holdout, contamination study | `notebooks/01_…`, `03_…` §2 and A7 |
| 2 | Saransh Singh Tomar | 2401010418 | Feature engineering and scaling ablation (quantile transform, cyclic hour, log amount) | Autoencoder input preprocessing and scaling ablation (A0) | `notebooks/01_…` §3–6, `02_…` §2 |
| 3 | Satwik Mani Tripathi | 2401020056 | GMM theory, EM algorithm, AIC/BIC model selection, seed-stability study | Hybrid second stage: latent GMMs, augmented GMM, evidence analysis | `notebooks/02_…` §3–5 and §8, `04_…` §3–5 |
| 4 | Gourav Tanwar | 2401010173 | Reference baselines (single Gaussian, Isolation Forest, supervised reference), error analysis | Autoencoder and denoising autoencoder: architecture, ablations A1–A6, model selection | `notebooks/02_…` §5 and §9, `03_…` |
| 5 | Alok Kumar | 2401010058 | Evaluation protocol, thresholds, cost analysis, report and slides | Significance tests, label efficiency (H5), final comparison, report and slides | `notebooks/02_…` §6–7, `04_…` §6–8, `reports/`, `presentation/` |

## Suggested Git workflow (so the history shows everyone's contribution)
Each member makes their own commits on the area they own, for example:
* Sharma Piyush: README data section, notebook 01
* Saransh Singh Tomar: feature-engineering text and figures
* Satwik Mani Tripathi: GMM / EM and hybrid sections of the reports
* Gourav Tanwar: autoencoder sections of the Phase 2 report
* Alok Kumar: evaluation sections, slides, viva notes

Graders look at the GitHub history, so commits from all five accounts are better than one person pushing everything.

## What *everyone* must be able to answer
See `docs/viva_prep.md`. At minimum:
1. Why accuracy is meaningless here, and what PR-AUC, recall at a false-positive budget and precision at a review
   capacity mean.
2. Why the models are trained on legitimate rows only, and where labels are (and are not) used.
3. What EM does in one sentence per step, and what AIC/BIC trade off.
4. What an autoencoder learns, why the bottleneck matters, and what "denoising" adds.
5. Which hybrids were tried, why the latent-space hybrids failed, and what the label-efficiency study shows.
