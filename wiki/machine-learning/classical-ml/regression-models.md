---
name: regression-models
description: Regression models for predicting numbers, with runnable scikit-learn and statsmodels code — simple and multiple linear regression (least squares, the five assumptions, dummy variables and the dummy variable trap shown as a rank-deficient matrix), p-values and statistical significance, five model-building strategies and backward elimination with OLS p-values (and where it disagrees with adjusted R²), polynomial regression as linear in the coefficients, support vector regression (ε-insensitive tube, slack variables, scaling X and y), decision tree regression (constant leaves) and random forest regression (averaged trees), the course's "truth or bluff" salary-at-level-6.5 case, R² and adjusted R², negative test R², comparing six regressors on held-out data. Study guide for Machine Learning A-Z, Part 2 (sections 5–14).
created: 2026-11-13
updated: 2026-11-13
tags: [machine-learning, regression, linear-regression, polynomial-regression, svr, decision-tree, random-forest, r-squared, backward-elimination, p-values, scikit-learn, statsmodels, study-guide, ml-a-z]
---
# Regression Models: Lines, Curves, Tubes and Trees

> **How to read this note.** Study guide for *Machine Learning A-Z*, Part 2 (sections 5–14): simple linear regression (`Salary_Data.csv`, salary vs years of experience), multiple linear regression (`50_Startups.csv`: R&D, administration and marketing spend plus state → profit), polynomial, SVR, decision-tree and random-forest regression on the 10-row `Position_Salaries.csv` "truth or bluff" case, R² and adjusted R², and a model-selection exercise on a power-plant dataset. Each model is built in Python and R. The snippets use stand-in data of the same shape; the course's numbers are quoted alongside. The quizzes aren't included. The least-squares derivation is in [artificial-neural-networks](../deep-learning/artificial-neural-networks.md) §5.1–5.2.

## Why This Exists

**Problem.** "Fit a regression" hides choices that change the answer: which features to keep, how to encode categories, whether the curve should bend, whether to scale, and how to tell a good fit from an overfit one.

**Key insight.** All five models answer the same question — *what number goes with these features?* — with different shapes: a plane (linear), a curve that is still linear in its coefficients (polynomial), a plane that ignores small errors (SVR), or a staircase (trees, averaged in a forest). Judge them all the same way: R² on data they haven't seen.

**Reach for this when:**
- Predicting a quantity (price, salary, profit, energy output) from tabular features.
- Asked which features "matter" in a linear model.
- Choosing between linear and non-linear regressors.

**Don't reach for this when:**
- Predicting a category — [classification-models](classification-models.md).
- Forecasting a time series where order matters — see [recurrent-neural-networks](../deep-learning/recurrent-neural-networks.md) and naive baselines.
- You want causal effects — regression coefficients on observational data aren't causal.

## Diagrams

![Least squares minimising vertical squared errors; SVR's ε-insensitive tube with a slack distance for a point outside; a tree's staircase of constant leaves; notes on polynomial regression being linear in the coefficients, the dummy trap, scaling, and R² vs adjusted R²](assets/regression.svg)

## 1. Linear regression

**Simple:** $y = b_0 + b_1 x$, with $b_0, b_1$ chosen to minimise the sum of squared vertical errors (**ordinary least squares**). The course's intuition: potato yield vs fertiliser; its build: salary vs experience (about $100k predicted at 8 years). **Multiple:** $y = b_0 + b_1 x_1 + \dots + b_k x_k$.

**Assumptions** (the course lists five, plus checking outliers): linearity, homoscedasticity (constant error spread), multivariate normality of errors, independence of errors, and no multicollinearity. They matter for p-values and confidence intervals; predictions can still be useful when they're bent.

**Dummy variables and the trap.** A state column becomes one 0/1 column per state, but you include **one fewer** than the number of categories: the left-out one is the baseline absorbed by $b_0$. Include all three and the dummies always sum to the intercept column:

```python
import numpy as np
np.set_printoptions(suppress=True)
from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(0)
n = 50                                                    # 50 startups: R&D, admin, marketing spend, state -> profit
rd, admin, mkt = rng.uniform(0, 165e3, n), rng.uniform(50e3, 180e3, n), rng.uniform(0, 470e3, n)
state = rng.integers(0, 3, n); D = np.eye(3)[state]       # one-hot: NY, CA, FL
profit = 50e3 + 0.8 * rd + 0.03 * mkt + rng.normal(0, 9e3, n)   # truth: only R&D (and a little marketing) matter
X_trap = np.c_[np.ones(n), D, rd, admin, mkt]             # intercept + ALL three dummies
print("columns:", X_trap.shape[1], " rank:", np.linalg.matrix_rank(X_trap), " -> dummies sum to the intercept column")
X_ok = np.c_[D[:, 1:], rd, admin, mkt]                    # drop one dummy: NY becomes the baseline
m = LinearRegression().fit(X_ok, profit)
print("coefficients CA, FL, R&D, admin, mkt:", m.coef_.round(3))
m_trap = LinearRegression().fit(np.c_[D, rd, admin, mkt], profit)   # sklearn still "fits" the trap version...
print("with all 3 dummies, state coefficients:", np.round(m_trap.coef_[:3]), "(not unique: add c to all, subtract c from intercept)")
# Expected output:
# columns: 7  rank: 6  -> dummies sum to the intercept column
# coefficients CA, FL, R&D, admin, mkt: [ -217.458 -1151.95      0.766     0.062     0.037]
# with all 3 dummies, state coefficients: [ 456.  239. -695.] (not unique: add c to all, subtract c from intercept)
```

Seven columns, rank six: no unique solution. scikit-learn still returns *a* fit (a minimum-norm one), so predictions are fine, but the state coefficients are arbitrary — the course's claim that scikit-learn "takes care of" the trap is wrong; it just doesn't complain. statsmodels and R's `lm` drop or flag the redundant column.

## 2. Which features matter: p-values and backward elimination

**P-values**, the course's way: assume the null hypothesis (a fair coin, or a coefficient of zero) and ask how surprising the data would be. Six heads in a row has probability 1.6%; at significance level 5%, you'd stop believing the coin is fair. A small p-value means *the data are unlikely if the coefficient were zero* — not "95% sure the feature matters", and not "bigger effect". (One lecture says the highest p-value is the most significant; it's the reverse.)

**Building a model** — five strategies: all-in, backward elimination, forward selection, bidirectional elimination, all possible models ($2^n - 1$ of them: 1,023 for 10 features). **Backward elimination**: fit with all features, drop the one with the highest p-value above the threshold, refit, repeat:

```python
import numpy as np, statsmodels.api as sm
rng = np.random.default_rng(0)
n = 50
rd, admin, mkt = rng.uniform(0, 165e3, n), rng.uniform(50e3, 180e3, n), rng.uniform(0, 470e3, n)
state = rng.integers(0, 3, n); D = np.eye(3)[state][:, 1:]
profit = 50e3 + 0.8 * rd + 0.03 * mkt + rng.normal(0, 9e3, n)
cols = {"CA": D[:, 0], "FL": D[:, 1], "R&D": rd, "admin": admin, "marketing": mkt}
while True:                                               # backward elimination at significance level 0.05
    fit = sm.OLS(profit, sm.add_constant(np.column_stack(list(cols.values())))).fit()   # ordinary least squares with p-values
    p = dict(zip(cols, fit.pvalues[1:]))
    worst = max(p, key=p.get)
    print(f"kept {list(cols)}  adj R² {fit.rsquared_adj:.4f}  worst: {worst} p={p[worst]:.3f}")
    if p[worst] <= 0.05: break
    del cols[worst]
# Expected output:
# kept ['CA', 'FL', 'R&D', 'admin', 'marketing']  adj R² 0.9443  worst: CA p=0.947
# kept ['FL', 'R&D', 'admin', 'marketing']  adj R² 0.9456  worst: FL p=0.743
# kept ['R&D', 'admin', 'marketing']  adj R² 0.9466  worst: admin p=0.076
# kept ['R&D', 'marketing']  adj R² 0.9440  worst: marketing p=0.001
```

The two state dummies go first (p ≈ 0.95 and 0.74), then administration (p = 0.076, just above 0.05) — but adjusted R² *drops* when administration is removed (0.9466 → 0.9440). The 5% line is arbitrary, and the two criteria disagree at the margin; the course meets the same situation with marketing spend (p ≈ 0.06) and resolves it with adjusted R². The course's R run on `50_Startups`: state dummies p ≈ 0.99 and 0.94, administration 0.60, marketing 0.06 — leaving R&D spend alone (or R&D + marketing). It runs elimination on all 50 rows, including the test rows; do selection on training data only. The Python version (an optional article) uses `statsmodels.OLS` with a hand-added column of ones, since scikit-learn reports no p-values.

## 3. Bending the line: polynomial, SVR, trees

**Polynomial regression** adds powers of a feature: $y = b_0 + b_1 x + b_2 x^2 + \dots + b_n x^n$. It's still *linear regression* — linear in the $b$'s — fitted on the columns $[x, x^2, \dots, x^n]$ (`PolynomialFeatures`).

**Support vector regression** fits a line with a tube of half-width ε around it; points inside the tube cost nothing, points outside cost their distance to the tube edge (**slack variables**). With an RBF kernel the "line" becomes a curve. SVR has no built-in scale, so both X and y must be standardised (separate scalers), and predictions mapped back with `inverse_transform`.

**Decision tree regression** splits the feature space into regions and predicts each region's **average** (a staircase in 1-D). Splits are chosen to reduce variance (squared error) — the course says "entropy", which is the classification criterion. **Random forest regression** averages many trees grown on bootstrap samples — the course's analogy is averaging many people's guesses of the number of sweets in a jar.

The course's case: a candidate claims a previous salary of 160k at a level between 6 (150k) and 7 (200k). Same question on my own salary table (110k at level 6, 150k at 7):

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
level = np.arange(1, 11).reshape(-1, 1)                   # job level 1..10 -> salary, rising steeply at the top
salary = np.array([40, 48, 58, 70, 85, 110, 150, 220, 380, 800]) * 1e3
q = [[6.5]]                                               # candidate between level 6 (110k) and 7 (150k)
poly = PolynomialFeatures(degree=4)                       # columns 1, x, x², x³, x⁴ -> still LINEAR in the coefficients
sx, sy = StandardScaler().fit(level), StandardScaler().fit(salary.reshape(-1, 1))
svr = SVR(kernel="rbf").fit(sx.transform(level), sy.transform(salary.reshape(-1, 1)).ravel())   # SVR needs both scaled
preds = {"linear": LinearRegression().fit(level, salary).predict(q)[0],
         "polynomial (degree 4)": LinearRegression().fit(poly.fit_transform(level), salary).predict(poly.transform(q))[0],
         "SVR (RBF, scaled X and y)": sy.inverse_transform(svr.predict(sx.transform(q)).reshape(-1, 1))[0, 0],
         "decision tree": DecisionTreeRegressor(random_state=0).fit(level, salary).predict(q)[0],
         "random forest (10 trees)": RandomForestRegressor(10, random_state=0).fit(level, salary).predict(q)[0]}
for k, v in preds.items(): print(f"{k:27s} predicts {v/1e3:7.1f}k for level 6.5")
# Expected output:
# linear                      predicts   258.2k for level 6.5
# polynomial (degree 4)       predicts   115.1k for level 6.5
# SVR (RBF, scaled X and y)   predicts   130.3k for level 6.5
# decision tree               predicts   110.0k for level 6.5
# random forest (10 trees)    predicts   124.0k for level 6.5
```

The straight line is far off (it's dragged up by the top levels); the curve-capable models land between the neighbouring levels; the single tree just returns level 6's salary. The course's results on its own table: linear ~330k, polynomial (degree 4) ~158k, SVR 170k (R: 177k), decision tree 150k in Python but 250k in R (rpart prunes the tree unless `minsplit` is lowered), random forest 167k with 10 trees (R: 141k, 166k and 160k for 10, 100 and 500 trees). With 10 points and no test set, all of these are interpolations from possibly overfit curves.

## 4. Scoring regressions: R² and adjusted R²

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y - \hat y)^2}{\sum (y - \bar y)^2}$$

compares your errors with those of always predicting the mean: 1 is perfect, 0 is no better than the mean, and on test data it can go **negative**. Adding any feature never lowers training R², so **adjusted R²** penalises the number of features $k$: $1 - (1 - R^2)\frac{n - 1}{n - k - 1}$.

The course's model-selection exercise trains all models on a power-plant dataset and compares test R²: multiple linear 0.93, polynomial 0.946, SVR 0.948, decision tree 0.922, random forest 0.96. On a standard non-linear benchmark:

```python
import numpy as np
from sklearn.datasets import make_friedman1                # a standard nonlinear benchmark: y depends on 5 of 10 features
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
X, y = make_friedman1(n_samples=1000, noise=1.0, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
def adj_r2(r2, n, k): return 1 - (1 - r2) * (n - 1) / (n - k - 1)   # penalise every extra feature
models = {"predict the mean": None, "multiple linear": LinearRegression(),
          "polynomial (deg 2)": make_pipeline(PolynomialFeatures(2), LinearRegression()),
          "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=10)), "decision tree": DecisionTreeRegressor(random_state=0),
          "random forest": RandomForestRegressor(300, random_state=0)}
for name, m in models.items():
    pred = np.full(len(yte), ytr.mean()) if m is None else m.fit(Xtr, ytr).predict(Xte)
    r2 = 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()   # R² = 1 − SS_res / SS_tot on the test set
    print(f"{name:20s} test R² {r2:6.3f}   adjusted (k=10) {adj_r2(r2, len(yte), 10):6.3f}")
# Expected output:
# predict the mean     test R² -0.009   adjusted (k=10) -0.062
# multiple linear      test R²  0.732   adjusted (k=10)  0.717
# polynomial (deg 2)   test R²  0.885   adjusted (k=10)  0.879
# SVR (RBF)            test R²  0.904   adjusted (k=10)  0.899
# decision tree        test R²  0.640   adjusted (k=10)  0.621
# random forest        test R²  0.849   adjusted (k=10)  0.841
```

The ranking depends on the data: here a tuned SVR wins and a single tree loses badly. Gaps of 0.002 (like the course's 0.946 vs 0.948) are within the noise of one split — use cross-validation to choose ([model-selection-boosting](model-selection-boosting.md)).

## 5. Course extras (articles)

- **Single predictions and the final equation** (two bonus notebooks): scikit-learn expects a 2-D input even for one row — `regressor.predict([[12]])` for 12 years of experience, and for the startup example the one-hot state columns come first (`[[1, 0, 0, 160000, 130000, 300000]]` — California is the first column in the encoder's alphabetical order). The fitted equation is `regressor.intercept_` plus `regressor.coef_` times each feature.
- **Conclusion of Part 2:** a pros-and-cons cheat sheet for the five regressors, regularisation slides, and the pointer that models have learned parameters (coefficients) and **hyperparameters** (λ, C) left at defaults until Part 10's tuning ([model-selection-boosting](model-selection-boosting.md) §3).

## Trade-offs

- **Linear vs flexible models.** Interpretable coefficients and p-values vs capturing curves (and overfitting small data).
- **Backward elimination vs regularisation.** Simple, readable vs Lasso/Ridge, which shrink coefficients continuously and are more stable.
- **SVR.** Robust to small errors, smooth curves vs needs scaling and tuning (C, ε, gamma), slow on large data.
- **Tree vs forest.** Readable rules vs smooth, accurate, less interpretable averages.

## Common Pitfalls

- **All dummies plus an intercept.** Coefficients become arbitrary; drop one per categorical feature.
- **Reading p-values as effect size or certainty.** They measure surprise under the null hypothesis, nothing more.
- **Feature selection on the full dataset.** The test rows influence which features survive; select on training data.
- **Forgetting to scale y for SVR.** Predictions collapse toward the mean.
- **Extrapolating polynomials.** High-degree curves explode outside the training range.
- **Choosing the winner on one test split.** Use cross-validation; small R² differences are noise.
- **Causal reading of coefficients.** "R&D spend predicts profit" isn't "spending on R&D causes profit".

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| Need coefficients and significance | Multiple linear regression (statsmodels / R `lm`) | Random forest | Inference |
| Curved single-feature trend | Polynomial (low degree) or SVR | High-degree polynomial | Overfitting, extrapolation |
| Many features, non-linear | Random forest / gradient boosting | Single tree | Variance |
| Choose features in a linear model | Regularisation or CV-based selection | p-value threshold alone | Arbitrary cut-off |
| Compare models | Cross-validated R² / RMSE | One split's R² | Noise |

## References

- [scikit-learn — Linear models](https://scikit-learn.org/stable/modules/linear_model.html) · [SVR](https://scikit-learn.org/stable/modules/svm.html#regression) · [Decision trees](https://scikit-learn.org/stable/modules/tree.html)
- [statsmodels — OLS](https://www.statsmodels.org/stable/regression.html)
- [Smola & Schölkopf — A Tutorial on Support Vector Regression (2004)](https://link.springer.com/article/10.1023/B:STCO.0000035301.49549.88)
- [James et al. — An Introduction to Statistical Learning](https://www.statlearning.com/) — ch. 3 (linear regression), 6 (selection), 8 (trees).

## See Also

- [classification-models](classification-models.md) — Part 3: the same families for categories.
- [data-preprocessing](data-preprocessing.md) — encoding and scaling used here.
- [statistics-and-probability](../math-for-ml/statistics-and-probability.md) — hypothesis tests and p-values.
- [model-selection-boosting](model-selection-boosting.md) — cross-validation and XGBoost.
