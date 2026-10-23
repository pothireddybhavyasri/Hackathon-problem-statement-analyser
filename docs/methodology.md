# Mathematical & Algorithmic Methodology — Hackathon Insights

This document outlines the theoretical foundations, formulas, assumptions, and algorithmic implementations utilized across the **Hackathon Insights** platform.

---

## 1. Data Cleaning & Statistical Outlier Detection

### Interquartile Range (IQR) Method
To distinguish genuine values from anomalous entry noise without making Gaussian distribution assumptions:
- **First Quartile ($Q_1$)**: 25th percentile value ($P_{25}$).
- **Third Quartile ($Q_3$)**: 75th percentile value ($P_{75}$).
- **Interquartile Range**:
  $$\text{IQR} = Q_3 - Q_1$$
- **Inner Fences (Tukey's Standard)**:
  $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
  $$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
Any data point $x < \text{Lower Bound}$ or $x > \text{Upper Bound}$ is flagged as a statistical outlier.

### Imputation Strategies
- **Continuous Features with Skewed Distributions** (`estimated_solution_days`, `innovation_score`): Imputed using the **median**, which is robust against extreme outliers.
- **Categorical Attributes** (`data_availability`, `sub_domain`): Imputed using the **mode** (most frequent class).

---

## 2. Statistical Analysis & Hypothesis Testing

### Pearson Correlation Coefficient ($r$)
Measures linear association between continuous variables $X$ and $Y$:
$$r_{XY} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

### One-Way Analysis of Variance (ANOVA)
Tests equality of means across $k$ problem domain groups:
$$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} = \frac{\frac{\text{SS}_{\text{between}}}{k - 1}}{\frac{\text{SS}_{\text{within}}}{N - k}}$$

### Pearson Chi-Square Test of Independence ($\chi^2$)
Tests association between two categorical features (e.g., Domain and AI Requirement):
$$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{R_i \times C_j}{N}$$

---

## 3. Supervised Predictive Modeling

### Multiple Linear Regression & Gauss-Markov BLUE Properties
$$y = X\beta + \varepsilon, \quad \hat{\beta} = (X^T X)^{-1} X^T y$$

Under the classical Gauss-Markov assumptions, the OLS estimator $\hat{\beta}$ is **BLUE** (Best Linear Unbiased Estimator):
1. **$E[\varepsilon | X] = 0$**: Zero conditional mean of residuals.
2. **$\text{Var}(\varepsilon | X) = \sigma^2 I_n$**: Homoscedastic and uncorrelated errors.
3. **$\text{Rank}(X) = p$**: No exact collinearity among regressors.

#### Evaluation Metrics:
- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
- **Coefficient of Determination ($R^2$)**:
  $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
  *Note on Negative $R^2$*: When evaluated on out-of-sample test data, $R^2$ can legitimately be negative if the model predictions have larger squared error than the simple horizontal line representing the test set mean.

### Logistic Regression
Models the probability of binary outcome $Y \in \{0, 1\}$ using the logistic sigmoid link function:
$$P(Y = 1 | X) = \sigma(X\beta) = \frac{1}{1 + e^{-X\beta}}$$
The parameter vector $\beta$ is estimated via Maximum Likelihood Estimation (MLE):
$$\ell(\beta) = \sum_{i=1}^n \left[ y_i \log \sigma(x_i^T \beta) + (1 - y_i) \log(1 - \sigma(x_i^T \beta)) \right]$$

---

## 4. Unsupervised Segmentation & NLP

### K-Means Clustering & Inertia
Partitions $n$ observations into $k$ clusters by minimizing the Within-Cluster Sum of Squares (WCSS / Inertia):
$$J = \sum_{j=1}^k \sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$

### Silhouette Coefficient
Evaluates cluster cohesion and separation:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
Where $a(i)$ is the mean distance to all other points in the same cluster, and $b(i)$ is the mean distance to points in the nearest neighboring cluster.

### TF-IDF Vectorization & Cosine Similarity
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \left( \log \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} + 1 \right)$$
$$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

---

## 5. Time-Series Modeling (ARIMA)

$$\Phi(L)(1 - L)^d Y_t = \delta + \Theta(L)\varepsilon_t$$
For $\text{ARIMA}(1, 1, 0)$:
$$(1 - \phi_1 L)(Y_t - Y_{t-1}) = \delta + \varepsilon_t$$
- **Backtest Accuracy**:
  $$\text{MAPE} = \frac{100\%}{n} \sum_{t=1}^n \left| \frac{y_t - \hat{y}_t}{y_t} \right|$$
