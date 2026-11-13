# Classical Machine Learning

Study guides for classical machine learning: own explanations, original diagrams (`assets/`, drawn with `scripts/classical_ml_diagrams.py`), and runnable scikit-learn snippets with expected output and a baseline. Neural networks live in [deep-learning](../deep-learning/INDEX.md); bandits and RL in [reinforcement-learning](../reinforcement-learning/INDEX.md); the maths in [math-for-ml](../math-for-ml/INDEX.md). Up: [machine-learning](../INDEX.md).

## Machine Learning A-Z (Udemy)

- [data-preprocessing](data-preprocessing.md) — Sections 1–4 (Part 1): workflow, split first, imputation, one-hot, normalization vs standardization (blue/purple/red example), leakage-free pipelines, re-scaling test data by its own stats goes wrong, Python ↔ R.
- [regression-models](regression-models.md) — Part 2: linear regression and assumptions, dummy trap as a rank-deficient matrix, p-values and backward elimination vs adjusted R², polynomial, SVR, tree and forest regression, the level-6.5 salary case, R² on held-out data.
- [classification-models](classification-models.md) — Part 3: logistic, k-NN, linear and kernel SVM, naive Bayes (wrench example), tree, forest on one dataset vs baseline; confusion matrix, accuracy paradox, CAP curve and accuracy ratio.
- [clustering](clustering.md) — Part 4: k-means, elbow vs silhouette, k-means++ and n_init, hierarchical clustering, dendrogram gap rule, linkage choice (two moons).
- [association-rules](association-rules.md) — Part 5: support, confidence, lift (movie example), Apriori with apyori, Eclat, thresholds from business volumes.
- Part 6 (UCB, Thompson sampling) → [multi-armed-bandits](../reinforcement-learning/multi-armed-bandits.md).
- [nlp-bag-of-words](nlp-bag-of-words.md) — Part 7: cleaning, stop words and keeping "not", stemming, CountVectorizer, naive Bayes vs logistic vs bigrams, R tm pipeline.
- Part 8 (ANN, CNN) → [artificial-neural-networks](../deep-learning/artificial-neural-networks.md) and [convolutional-neural-networks](../deep-learning/convolutional-neural-networks.md), "Also in Machine Learning A-Z" sections.
- [dimensionality-reduction](dimensionality-reduction.md) — Part 9: PCA by hand on wine, LDA, kernel PCA, when PCA discards the signal.
- [model-selection-boosting](model-selection-boosting.md) — Part 10: lucky splits, k-fold CV, bias–variance, grid search and nested CV, XGBoost vs logistic/forest/tree.
- [aws-ml-data-preparation](aws-ml-data-preparation.md) — Part 11: S3, Glue, DataBrew, Data Wrangler, cleanup.
- [aws-ml-model-development](aws-ml-model-development.md) — Part 12: XGBoost gain by hand, LightGBM tricks, CatBoost ordered target encoding vs leaky encoding, Canvas models, Model Registry, AI services.
- [aws-ml-deployment-mlops](aws-ml-deployment-mlops.md) — Parts 13–15: inference types, containers, IaC, CI/CD with a testing build step, Model Monitor and drift (PSI vs KS), observability and cost control.
