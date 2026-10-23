# Viva Preparation — Project 26: Card Fraud Triage

Short, plain-English answers. Every member should be comfortable with **all** of sections 1–4. Numbers for our own
results are in section 6 (they come from the Colab runs in `notebooks/` and `results/`).

---

## 1. Problem framing

**Q1. What is the project in one sentence?**
We rank card transactions by how *abnormal* they look compared to legitimate behaviour, so a small team of analysts
reviews the most suspicious ones first. We compare a Gaussian Mixture density score (ML), an autoencoder
reconstruction score (DL), and a hybrid of the two.

**Q2. Why treat fraud as anomaly detection instead of normal classification?**
(1) Fraud is extremely rare (0.17 %), so there are very few positive examples. (2) Fraud patterns change: a
classifier learns *past* fraud, but a model of *normal* behaviour can flag new kinds of fraud. (3) In practice, labels
arrive late (chargebacks take weeks). Legitimate data is plentiful and available immediately.

**Q3. What does "triage" mean here?**
Sorting alerts by priority for a limited review team. The question is not "fraud or not?" but "which k transactions
should analysts look at first?". That is why we report precision at a review capacity.

**Q4. Where exactly are labels used?**
Only in three places: (1) to *remove* frauds from the training block (the legit-only protocol in the brief),
(2) on the **validation** block to choose hyper-parameters and thresholds, (3) on the **test** block, once, to report
results. The GMM and the autoencoder never see a label during fitting. Early stopping uses a *label-free* random 10 %
of the legitimate training rows.

## 2. Data and features

**Q5. What are V1–V28?**
The bank applied PCA to confidential original features, so V1–V28 are principal components. They are centred, nearly
uncorrelated over the whole dataset, and ordered by variance. We cannot interpret them individually.

**Q6. Why drop duplicates *before* splitting?**
An identical row in both train and test would be "memorised", which inflates test scores (leakage). There were 1,081
duplicates, 19 of them fraud.

**Q7. Why a chronological split instead of a random one?**
In production you train on the past and score the future. A random split lets the model peek at future patterns and
overestimates performance. Ours is 60 % / 20 % / 20 % by time.

**Q8. Why encode hour of day as sine and cosine?**
Hour 23 and hour 0 are neighbours, but the numbers 23 and 0 are far apart. Mapping the hour onto a circle
(sin, cos) keeps them close. Fraud *rate* is much higher at night, so the hour carries signal.

**Q9. Why log(1 + Amount)?**
Amount is extremely right-skewed (skewness > 15). The log compresses the long tail so a few huge purchases do not
dominate distances and variances. The "+1" handles zero amounts.

**Q10. What does the quantile transformer do, and why did it help the GMM so much?**
It maps each feature through its own empirical CDF and then the inverse normal CDF. Every feature's *marginal*
distribution becomes standard normal on the training data. Our legitimate features are very heavy-tailed (excess
kurtosis up to ≈ 966). With standard scaling, a few extreme *legitimate* values dominate the Gaussian components and
EM becomes unstable. After the transform, the GMM's Gaussian assumption is much more reasonable. One caveat: it clips
values beyond the training range.

**Q11. Why fit scalers on legitimate training rows only?**
So that no information from validation/test (or from fraud) leaks into preprocessing. The scaler is part of the model.

## 3. Phase 1 — Gaussian Mixture Model

**Q12. What is a GMM?**
A weighted sum of K Gaussian distributions: p(x) = Σ πₖ N(x | μₖ, Σₖ). Each component models one "kind" of normal
transaction.

**Q13. How is it fitted? Explain EM.**
- **E-step:** for each point, compute the responsibility γᵢₖ, the probability it came from component k (using current
  parameters).
- **M-step:** re-estimate each component's weight, mean and covariance as responsibility-weighted averages.

Repeat until the log-likelihood stops improving. Each iteration never decreases the likelihood, but EM can stop at a
*local* optimum. That is why the random seed matters.

**Q14. What is the anomaly score?**
s(x) = −log p(x). A transaction in a low-density region of legitimate behaviour gets a high score.

**Q15. What are AIC and BIC?**
AIC = 2p − 2 log L̂ and BIC = p log n − 2 log L̂, where p is the number of parameters and n the number of rows. Both
reward fit (likelihood) and penalise complexity. BIC penalises more (log n > 2 when n > 7), so it prefers smaller
models.

**Q16. How many parameters does a full-covariance GMM have?**
(K − 1) weights + K·d means + K·d(d+1)/2 covariance entries. Our final inputs are V1–V28 (d = 28), so each component
costs 28 + 406 = 434 parameters (527 with all 31 features). K = 24 is over 10,000 parameters.

**Q17. Full vs diagonal covariance?**
Full captures correlations between features (it can model tilted ellipses). Diagonal assumes features are independent
within a component: far fewer parameters, but it misses correlations. Fraud moves several components *together*, so
full covariance worked much better.

**Q18. What does `reg_covar` do?**
It adds λI to every covariance matrix. This keeps it invertible when a component collapses onto a few nearly-identical
points (a singular covariance would make the likelihood infinite).

**Q19. Why did BIC disagree with the best detector?**
With about 150,000 rows, the BIC penalty p·log n is small compared to the likelihood gain from extra components, so
BIC keeps preferring larger K. But extra components start modelling *rare legitimate* behaviour and the sparse
regions where fraud lives, which raises fraud's density and *hurts* detection. "Best density model of legit" ≠ "best
fraud detector". We report the BIC choice and select K on validation PR-AUC.

**Q20. What is the "gap" of the GMM that Phase 2 addresses?**
(1) It assumes each cluster is an ellipse (Gaussian), but legitimate data is skewed, heavy-tailed and curved.
(2) In 28 dimensions each component needs 434 parameters, and EM's result depends on the random start (with standard
scaling, test PR-AUC ranged from 0.15 to 0.74 across seeds). (3) Better likelihood does not mean better detection.

## 4. Phase 2 — Autoencoders and hybrid

**Q21. What is an autoencoder?**
A neural network trained to output its own input. An encoder compresses x into a small code z, and a decoder rebuilds
x̂ from z. Trained only on legitimate data, it rebuilds normal transactions well and unusual ones badly.

**Q22. Why must it be undercomplete?**
If the code were as big as the input, the network could learn the identity function and rebuild *everything*,
including fraud. The bottleneck forces it to keep only the structure common to legitimate data.

**Q23. Linear autoencoder vs PCA?**
A linear AE trained with MSE learns the same subspace as PCA (Baldi & Hornik, 1989). Non-linear activations let it
follow a *curved* manifold. We test this in ablation A6.

**Q24. What is a denoising autoencoder?**
During training we add noise to the input (Gaussian, or masking random features) and ask the network to rebuild the
*clean* input. It must learn to pull points back onto the data manifold. This is a regulariser that also makes
features more robust. At test time no noise is added.

**Q25. How do you prevent over-fitting?**
The bottleneck, weight decay (L2), dropout, input noise (DAE), and early stopping on a label-free holdout. Each one is
ablated in Notebook 03.

**Q26. Why ELU and a linear output layer?**
ELU is smooth and non-zero for negative inputs, so there are no "dead" units. The inputs are standardised and
unbounded, so the output must be linear (a sigmoid output could not produce −3 or +4).

**Q27. What hybrids did you build, and why are they not just "combining two models"?**
Each one tests a specific idea, and the naive combination (late fusion, H4) is included as the baseline any real
hybrid must beat.
* **H1** — a GMM on the AE's latent code z: is density easier to model in a learned low-dimensional space? (DAGMM idea)
* **H2** — a GMM on [z, log reconstruction error, cosine(x, x̂)]: combine *where* a point is on the manifold with *how
  far* it is from it (DAGMM).
* **H3** — a GMM on [raw features x, log reconstruction error]: keep all the input information and add the learned
  error as an extra dimension.
* **H4** — late fusion: average of rank-normalised GMM and AE scores.
* **H5** — semi-supervised: gradient boosting on [x, GMM score, AE error] using analyst labels.

**Q28. Why two-stage instead of DAGMM's joint training?**
Joint training is known to be fragile: the GMM energy term can pull the encoder into degenerate solutions. Two-stage
training is stable, and each stage can be validated separately. Joint training is listed as future work.

**Q29. How did you avoid leakage in the hybrids?**
* **H1–H4:** the legitimate training rows are split into halves. Half A trains the AE; half B (40k rows) fits every
  stage-2 GMM. The AE never scores its own training rows, where its errors would be unrealistically small.
* **H5:** GMM and AE scores for training rows are computed out-of-fold (2-fold cross-fitting), so the supervised model
  never sees an "in-sample" score.

## 5. Evaluation

**Q30. Why not accuracy?**
Predicting "legit" for everything is 99.83 % accurate and catches zero fraud.

**Q31. What is PR-AUC (average precision)?**
The area under the precision–recall curve: average precision over all recall levels as the threshold moves. A random
ranker scores the base rate (≈ 0.0013 on test). It focuses on the rare positive class.

**Q32. Why is ROC-AUC misleading here?**
The false-positive rate divides by ~56,000 legitimate rows, so even hundreds of false alarms look like a tiny FPR. ROC
can be 0.95+ while most alerts are false. PR curves expose this.

**Q33. Explain "recall at a 0.1 % false-positive budget".**
We set the threshold on *validation* so that only 0.1 % of legitimate customers would be flagged. Then we apply that
same frozen threshold to the test period and measure what fraction of frauds are caught. The *realised* test FPR is
reported too, because it can drift.

**Q34. Explain "precision at a review capacity of k".**
Analysts can open only k alerts. P@k is the fraction of the top-k scored transactions that are real fraud.

**Q35. How do you know a difference between models is real?**
Paired stratified bootstrap. We resample the test set 1,000 times, compute PR-AUC(A) − PR-AUC(B) on each resample,
and check whether the 95 % interval excludes 0. We also report the spread over several random seeds.

**Q36. Limitations?**
(1) Two days of data from 2013, so drift over months is untested. (2) Features are anonymised PCA components, so
there is no domain-level feature engineering (merchant, customer history). (3) Validation and test fall in daytime
hours, so night-time fraud is only in training. (4) Only 74 test frauds, so confidence intervals are wide.

## 6. Our results (filled from the Colab runs)

**Q37. What are your headline numbers?** (test set: last 20 % of the timeline, 74 frauds, base rate 0.13 %)
* Phase-1 GMM (K = 4, quantile inputs): **PR-AUC 0.686**, 76 % of frauds caught at a 0.1 % false-positive budget,
  top-50 queue 90 % fraud.
* Phase-2 denoising AE: **0.661** (not significantly different from the GMM). Plain AE: 0.618 (significantly worse).
* Best hybrid (GMM on [x, log reconstruction error]): **0.695**, a tie with the GMM.
* Supervised gradient boosting with all labels (reference): 0.778.

**Q38. Why K = 4 and not what BIC said (K = 24)?**
BIC kept improving up to the largest K we tried, because with ~40,000–170,000 rows the penalty p·log n is small. But
validation PR-AUC did not improve with K and became less stable across EM seeds, so we chose K on validation. In our
Colab run the K = 24 model scored 0.626 on test; in a local re-run it scored 0.73. That variation is itself evidence of
instability at large K.

**Q39. Your single Gaussian (K = 1) scored about the same as K = 4. So why a mixture?**
Honest answer: after the quantile transform the data is close enough to Gaussian that one component already works well.
The mixture is the method the brief asked for; we report that its *number of components* matters little, and that the
*feature engineering* matters a lot. That is a finding, not a failure.

**Q40. Why did you use a 40,000-row subsample for model selection?**
The selection studies fit ~100 GMMs. On Colab's free CPU one full-data fit took over a minute, so the full study would
not fit in a session. Selection on a random subsample is standard practice; the final model and all baselines use all
169,894 legitimate training rows.

**Q41. Which autoencoder settings mattered most?**
Bottleneck size (16 ≫ 2–8), non-linearity (the linear/PCA-like AE was the worst), and Gaussian input noise σ = 0.2. Dropout
hurt, because it makes reconstructions noisy and noise is what we score. Leaving the 0.2 % training frauds in the
training data destroyed the model.

**Q42. Why not just combine all the best ablation settings?**
We tried it: validation PR-AUC fell to ≈ 0.02. Factors interact, so we compared a short list of candidates (default,
each single change, top-2, all combined) on validation and picked the best, which was the single change m = 16.

**Q43. Why did the DAGMM-style (latent) hybrids fail?**
At a 1 % FPR the latent-space GMM caught only ≈ 4 % of frauds, against ≈ 77 % for the GMM on raw inputs. The encoder,
trained only on legitimate data, maps frauds close to normal codes (|z| 3.5 vs 2.9). The anomaly evidence is in the
*reconstruction residual*, not in the code.

**Q44. If the hybrid didn't win, what is its value?**
1. It shows *where* the information is: the residual, not the code.
2. It shows that the AE and the GMM find the *same* frauds (ρ ≈ 0.95), so combining them cannot catch new ones.
3. The label-efficiency study quantifies the value of labels: the label-free GMM ≈ 17 labelled frauds.

All three are evidence-based design guidance for a real triage system.

**Q45. What would you deploy?**
The label-free GMM on day one, with the threshold set by review capacity: for example, a 0.1 % budget gives 94 alerts
in 7.7 h and catches 76 % of frauds. Once a few dozen frauds are confirmed, switch to (or add) a supervised model, and
retrain regularly because fraud drifts.

**Q46. What is the cost-based view?**
With an illustrative review cost of 5 per alert, and each missed fraud costing its amount, the total cost is lowest
when reviewing the top ≈ 61 alerts: ≈ 2,763 against 7,728 with no model.
