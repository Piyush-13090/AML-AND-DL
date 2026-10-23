# Literature Review — Project 26: Card Fraud Triage

This review supports both evaluation phases:
* **Part A (Phase 1, ML):** fraud detection as anomaly detection, Gaussian mixtures and EM, model selection, evaluation under extreme imbalance.
* **Part B (Phase 2, DL):** autoencoder-based anomaly detection, denoising, and hybrid deep-density models.

Each entry gives *what the work did* and *how we use it*. Full references are at the end, and the same list is in
`reports/*/references.bib`.

---

## Part A — Machine-learning foundations (Phase 1)

### A1. Fraud detection as a learning problem
* **Bolton & Hand (2002)** review statistical fraud detection. They separate *supervised* methods (learn from labelled
  past fraud) from *unsupervised* methods (flag departures from normal behaviour). They argue unsupervised methods
  matter because fraud patterns change and labels arrive late and incomplete. → *This is the motivation for our
  legitimate-only training protocol.*
* **Dal Pozzolo et al. (2015)** introduced the ULB credit-card dataset we use: 284,807 transactions, 492 frauds, with
  PCA-anonymised features. They studied how undersampling distorts predicted probabilities. **Dal Pozzolo et al.
  (2018)** describe a realistic fraud-detection system: alerts are reviewed by a limited team of investigators, so the
  relevant metric is **precision among the top-k alerts**, and feedback labels are delayed. → *We report precision
  at a review capacity (P@k) for exactly this reason.*
* **Carcillo et al. (2021)** combine unsupervised outlier scores with supervised classifiers on card transactions and
  show that the unsupervised scores add information. → *This supports reporting a supervised reference and treating
  anomaly scores as triage signals.*

### A2. Anomaly / novelty detection
* **Chandola, Banerjee & Kumar (2009)** survey anomaly detection and classify techniques as statistical (density),
  distance, clustering, classification, and spectral. Statistical techniques assume "normal instances occur in high
  probability regions of a stochastic model". → *That is precisely the GMM score −log p(x).*
* **Pimentel et al. (2014)** review *novelty detection*: training on normal data only and flagging departures. They
  identify GMM-based density estimation as the canonical probabilistic approach and note its sensitivity to
  dimensionality and model order. → *This is the gap stated in the project brief.*
* **Liu, Ting & Zhou (2008)** propose Isolation Forest: anomalies are isolated by fewer random axis-aligned splits. It
  needs no density model. → *We use it as a non-density unsupervised reference baseline.*

### A3. Gaussian mixtures, EM and model order
* **Dempster, Laird & Rubin (1977)** formalised the EM algorithm for maximum likelihood with latent variables and
  proved that each iteration does not decrease the likelihood. → *This is our fitting procedure. It converges only to
  a local optimum, which we measure (seed-stability study).*
* **McLachlan & Peel (2000)** and **Bishop (2006, Ch. 9)** give the standard treatment of finite mixtures:
  responsibilities, covariance structures (full/diagonal/tied), the singularity problem when a component collapses
  onto few points, and regularisation of covariances. → *This motivates our `reg_covar` λI term and its sensitivity
  study.*
* **Akaike (1974)** (AIC) and **Schwarz (1978)** (BIC) give the information criteria used to choose the number of
  components. BIC is consistent for the true order when the model family is correct. → *We use both, as the brief
  asks. We also show that with n ≈ 150k rows and a misspecified (non-Gaussian) family they keep favouring more
  components, while fraud-detection quality does not improve. That is an important nuance.*

### A4. Evaluation under extreme class imbalance
* **Davis & Goadrich (2006)** show the relationship between ROC and PR space. A curve dominates in ROC space if and
  only if it dominates in PR space, but ROC-AUC can look excellent while precision is poor.
* **Saito & Rehmsmeier (2015)** show empirically that PR plots are more informative than ROC plots on imbalanced data.
  → *Our primary metric is PR-AUC (average precision). We never report accuracy (a do-nothing model scores 99.83 %).*

---

## Part B — Deep-learning foundations (Phase 2)

### B1. Autoencoders and their relation to PCA
* **Baldi & Hornik (1989)** prove that a linear autoencoder trained with squared error has no spurious local minima,
  and its optimum spans the principal subspace. A linear AE therefore *is* PCA. → *Our ablation A6 ("linear"
  activation) tests whether non-linearity adds value beyond PCA-like reconstruction.*
* **Hinton & Salakhutdinov (2006)** show deep non-linear autoencoders give much better low-dimensional codes than PCA.
* **Goodfellow, Bengio & Courville (2016, Ch. 14)** give a textbook treatment of undercomplete, regularised, sparse and
  denoising autoencoders. → *Our architecture follows this.*

### B2. Reconstruction error as an anomaly score
* **Sakurada & Yairi (2014)** use autoencoders for anomaly detection by training on normal data and scoring by
  reconstruction error. They show this captures non-linear structure that linear PCA misses. → *This is our Phase-2
  baseline, as named in the project brief.*
* **An & Cho (2015)** propose a variational-AE "reconstruction probability" score. **Chalapathy & Chawla (2019)** and
  **Pang et al. (2021)** survey deep anomaly detection. They list reconstruction-based models as the dominant family
  and point out their main weakness: a model that generalises too well can also reconstruct anomalies.
* **Pumsirirat & Yan (2018)** apply autoencoders (and RBMs) to credit-card fraud data, including the ULB dataset, as
  unsupervised detectors.

### B3. Denoising and regularisation
* **Vincent et al. (2008, 2010)** introduce denoising autoencoders: corrupt the input, reconstruct the clean version.
  This forces robust features and prevents trivial identity mappings, even without a bottleneck.
* **Alain & Bengio (2014)** show the optimal DAE reconstruction r(x) satisfies r(x) − x ∝ σ²∇ₓ log p(x) for small
  noise. The DAE implicitly learns the *score* of the data density. → *This is a theoretical bridge between the
  Phase-1 density view and the Phase-2 reconstruction view.*
* **Srivastava et al. (2014)** (dropout), **Kingma & Ba (2015)** (Adam) and **Clevert et al. (2016)** (ELU) are the
  training components we use and ablate.

### B4. Hybrid deep + density models
* **Zong et al. (2018) — DAGMM** couple a compression network (autoencoder) with an estimation network that fits a
  GMM on the concatenation of the latent code and reconstruction-error features (relative Euclidean distance, cosine
  similarity). Both are trained jointly. DAGMM beat stand-alone AE and GMM baselines on several benchmarks,
  including KDDCUP99. → *Our hybrid follows the same representation idea, [z, reconstruction features] → GMM, but
  trains it in two stages for stability and transparency.*
* **Ruff et al. (2018) — Deep SVDD** learn a representation in which normal data falls inside a minimum-volume
  hypersphere. This is another "deep representation + classical one-class criterion" hybrid, and a possible future
  extension.

### Summary — where our project sits

| Approach | Captures | Misses | Our use |
|---|---|---|---|
| GMM on raw features | full probabilistic density, interpretable | non-Gaussian / curved structure; unstable in high-d | Phase 1 |
| Autoencoder (AE / DAE) | non-linear manifold of normal data | *where* a point lies on the manifold; uses only a scalar error | Phase 2 |
| Latent-space GMM (DAGMM-style) | both manifold distance and on-manifold density | joint training is unstable → two-stage | Phase 2 hybrid |

---

## References

1. H. Akaike, "A new look at the statistical model identification," *IEEE Trans. Automatic Control*, vol. 19, no. 6, pp. 716–723, 1974.
2. G. Alain and Y. Bengio, "What regularized auto-encoders learn from the data-generating distribution," *JMLR*, vol. 15, pp. 3563–3593, 2014.
3. J. An and S. Cho, "Variational autoencoder based anomaly detection using reconstruction probability," SNU Data Mining Center, Tech. Rep., 2015.
4. P. Baldi and K. Hornik, "Neural networks and principal component analysis: Learning from examples without local minima," *Neural Networks*, vol. 2, no. 1, pp. 53–58, 1989.
5. C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.
6. R. J. Bolton and D. J. Hand, "Statistical fraud detection: A review," *Statistical Science*, vol. 17, no. 3, pp. 235–255, 2002.
7. F. Carcillo, Y.-A. Le Borgne, O. Caelen, Y. Kessaci, F. Oblé, and G. Bontempi, "Combining unsupervised and supervised learning in credit card fraud detection," *Information Sciences*, vol. 557, pp. 317–331, 2021.
8. R. Chalapathy and S. Chawla, "Deep learning for anomaly detection: A survey," arXiv:1901.03407, 2019.
9. V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," *ACM Computing Surveys*, vol. 41, no. 3, art. 15, 2009.
10. D.-A. Clevert, T. Unterthiner, and S. Hochreiter, "Fast and accurate deep network learning by exponential linear units (ELUs)," in *Proc. ICLR*, 2016.
11. A. Dal Pozzolo, O. Caelen, R. A. Johnson, and G. Bontempi, "Calibrating probability with undersampling for unbalanced classification," in *Proc. IEEE SSCI*, 2015, pp. 159–166.
12. A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, "Credit card fraud detection: A realistic modeling and a novel learning strategy," *IEEE Trans. Neural Networks and Learning Systems*, vol. 29, no. 8, pp. 3784–3797, 2018.
13. J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," in *Proc. ICML*, 2006, pp. 233–240.
14. A. P. Dempster, N. M. Laird, and D. B. Rubin, "Maximum likelihood from incomplete data via the EM algorithm," *J. Royal Statistical Society B*, vol. 39, no. 1, pp. 1–38, 1977.
15. I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.
16. G. E. Hinton and R. R. Salakhutdinov, "Reducing the dimensionality of data with neural networks," *Science*, vol. 313, no. 5786, pp. 504–507, 2006.
17. D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015.
18. F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. IEEE ICDM*, 2008, pp. 413–422.
19. G. McLachlan and D. Peel, *Finite Mixture Models*. Wiley, 2000.
20. G. Pang, C. Shen, L. Cao, and A. van den Hengel, "Deep learning for anomaly detection: A review," *ACM Computing Surveys*, vol. 54, no. 2, art. 38, 2021.
21. F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
22. A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Proc. NeurIPS*, 2019.
23. M. A. F. Pimentel, D. A. Clifton, L. Clifton, and L. Tarassenko, "A review of novelty detection," *Signal Processing*, vol. 99, pp. 215–249, 2014.
24. A. Pumsirirat and L. Yan, "Credit card fraud detection using deep learning based on auto-encoder and restricted Boltzmann machine," *Int. J. Advanced Computer Science and Applications*, vol. 9, no. 1, 2018.
25. L. Ruff *et al.*, "Deep one-class classification," in *Proc. ICML*, 2018, pp. 4393–4402.
26. T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLoS ONE*, vol. 10, no. 3, e0118432, 2015.
27. M. Sakurada and T. Yairi, "Anomaly detection using autoencoders with nonlinear dimensionality reduction," in *Proc. MLSDA Workshop*, 2014, pp. 4–11.
28. G. Schwarz, "Estimating the dimension of a model," *Annals of Statistics*, vol. 6, no. 2, pp. 461–464, 1978.
29. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *JMLR*, vol. 15, pp. 1929–1958, 2014.
30. P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, "Extracting and composing robust features with denoising autoencoders," in *Proc. ICML*, 2008, pp. 1096–1103.
31. P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, and P.-A. Manzagol, "Stacked denoising autoencoders," *JMLR*, vol. 11, pp. 3371–3408, 2010.
32. B. Zong *et al.*, "Deep autoencoding Gaussian mixture model for unsupervised anomaly detection," in *Proc. ICLR*, 2018.
