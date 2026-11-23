# DIALECTICA — Robust Experimental Design References

1. **Callahan, J., & Catanach, T. (2026).** On the Misinformation in a Statistical Experiment. Proceedings of AISTATS 2026, PMLR 300, 2287–2295. https://proceedings.mlr.press/v300/callahan26a.html. Defines EGIG, the posterior-mismatch penalty, and a KL-instantiated robust f-information measure. PDF: https://openreview.net/pdf?id=kO66LSLmB8.
2. **Tang, R., Sloman, S. J., & Kaski, S. (2026).** Representative, Informative, and De-Amplifying: Requirements for Robust Bayesian Active Learning under Model Misspecification. Proceedings of AISTATS 2026, PMLR 300, 3016–3024. https://proceedings.mlr.press/v300/tang26d.html. Derives R-I and R-IDeA acquisition functions and studies representativeness, informativeness, and error de-amplification.
3. **Lindley, D. V. (1956).** On a Measure of the Information Provided by an Experiment. The Annals of Mathematical Statistics, 27(4), 986–1005. DOI: https://doi.org/10.1214/aoms/1177728069.
4. **Catanach, T. A., & Das, N. (2023).** Metrics for Bayesian Optimal Experiment Design under Model Misspecification. arXiv:2304.07949. https://arxiv.org/abs/2304.07949. Earlier development of EGIG and Expected Discriminatory Information.
5. **Gretton, A., et al. (2012).** A Kernel Two-Sample Test. Journal of Machine Learning Research, 13, 723–773. https://www.jmlr.org/papers/v13/gretton12a.html. Foundation for MMD as a distribution discrepancy measure used by R-I.
6. **Atkinson, A. C., & Fedorov, V. V. (1975).** Optimal Design: Experiments for Discriminating between Several Models. Biometrika, 62(2), 289–303. DOI: https://doi.org/10.1093/biomet/62.2.289.

## Reproduction code

- R-IDeA supplementary implementation: https://github.com/TrbingWY/robust-boed
- EGIG AISTATS 2026 supplementary implementation: https://github.com/jcallahan4/misinformation-statistical-experiment
