# Viva Questions and Concise Answers — Hackathon Insights

This document contains 25+ comprehensive viva examination questions with concise, technically accurate answers grounded in academic data analytics theory (`CS513PE`) and this project's implementation.

---

### Q1: What is the primary difference between data cleaning and data preprocessing?
**Answer**: Data cleaning focuses on detecting and correcting errors, noise, duplicate rows, missing entries, and corrupted formats. Data preprocessing transforms clean data into representations optimal for machine learning algorithms (e.g., scaling numerical variables, one-hot encoding categorical features, and text vectorization).

### Q2: Why did you choose median imputation over mean imputation for numerical variables like `estimated_solution_days`?
**Answer**: Mean is highly sensitive to extreme outliers and skewed distributions. The median represents the 50th percentile and is a robust non-parametric measure of central tendency that preserves the true distributional center without inflation from extreme values.

### Q3: How does the Interquartile Range (IQR) method detect outliers, and why not use Z-scores?
**Answer**: The IQR method computes $\text{IQR} = Q_3 - Q_1$ and defines fences at $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$. It is non-parametric and does not assume a normal Gaussian distribution. Z-scores assume data is normally distributed and can be distorted by the very outliers they attempt to detect.

### Q4: When should you use Normalization versus Standardization?
**Answer**: Min-Max Normalization rescales features into $[0, 1]$ and is sensitive to outliers; it is ideal when bounded scales are required (e.g., neural networks or image pixels). Z-score Standardization transforms data to $\mu = 0, \sigma = 1$; it is robust to outliers and preferred for Linear Regression, Logistic Regression, PCA, and K-Means.

### Q5: What is the BLUE property in Linear Regression?
**Answer**: According to the Gauss-Markov Theorem, under classical assumptions (linearity, strict exogeneity $E[\varepsilon|X]=0$, homoscedasticity, no autocorrelation, and full rank), the Ordinary Least Squares (OLS) estimator is the **Best Linear Unbiased Estimator (BLUE)**—meaning it achieves minimum variance among all linear unbiased estimators.

### Q6: Can the $R^2$ score be negative on test data? Why?
**Answer**: Yes. On unseen test data, $R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$. If the trained model makes predictions with larger squared residual error than a horizontal baseline predicting the test set mean ($\bar{y}_{\text{test}}$), $SS_{\text{res}} > SS_{\text{tot}}$, producing a negative $R^2$. This indicates poor generalization or overfitting on training data.

### Q7: What is the difference between MAE and RMSE?
**Answer**: Mean Absolute Error (MAE) averages absolute errors $|y - \hat{y}|$, penalizing errors linearly and maintaining intuitive units. Root Mean Squared Error (RMSE) squares errors before taking the root, heavily penalizing large outlier errors.

### Q8: How does Logistic Regression differ from Linear Regression?
**Answer**: Linear Regression predicts an unbounded continuous target ($y \in \mathbb{R}$) using least squares. Logistic Regression predicts the probability of a categorical outcome ($P \in [0, 1]$) using the logistic sigmoid link function $\sigma(z) = \frac{1}{1 + e^{-z}}$ and is fitted via Maximum Likelihood Estimation.

### Q9: What is the "Accuracy Paradox" in classification?
**Answer**: In imbalanced datasets (e.g., 90% class 0 and 10% class 1), a naive model that always predicts class 0 achieves 90% accuracy but has 0% recall on the minority class. Therefore, precision, recall, F1-score, and ROC-AUC are essential to evaluate true model utility.

### Q10: How do Decision Trees split data, and what is the difference between Gini and Entropy?
**Answer**: Decision Trees recursively partition feature space to maximize purity. Gini Impurity ($1 - \sum p_i^2$) measures the probability of misclassifying a randomly chosen element. Entropy ($-\sum p_i \log_2 p_i$) measures information uncertainty. Gini is computationally faster; both produce comparable tree structures.

### Q11: What causes overfitting in Decision Trees, and how is it mitigated?
**Answer**: Without constraints, a Decision Tree will continue splitting until every training sample is isolated into pure leaves, memorizing noise. It is mitigated by pre-pruning (restricting `max_depth`, `min_samples_split`, or `min_samples_leaf`) or post-pruning via cost-complexity pruning ($\alpha$).

### Q12: How do Random Forests improve upon single Decision Trees?
**Answer**: Random Forests are an ensemble method utilizing **Bagging** (Bootstrap Aggregating) and random feature subspace selection. By training multiple de-correlated trees on bootstrapped subsets of data and averaging their predictions, Random Forests dramatically reduce variance without increasing bias.

### Q13: How does K-Means clustering work, and what does Inertia measure?
**Answer**: K-Means is an iterative unsupervised algorithm that assigns points to the nearest cluster centroid ($\mu_k$) and recomputes centroids as cluster means until convergence. Inertia (Within-Cluster Sum of Squares) measures total squared Euclidean distance from points to their assigned centroid: $\sum \sum \|x_i - \mu_k\|^2$.

### Q14: What does the Silhouette Score indicate?
**Answer**: The Silhouette Score evaluates cluster cohesion ($a$) versus separation from neighboring clusters ($b$): $s = \frac{b - a}{\max(a, b)}$. It ranges from $-1$ (incorrect cluster assignment) through $0$ (overlapping clusters) to $+1$ (dense, clearly separated clusters).

### Q15: Why is PCA used in clustering visualization, and what is its limitation?
**Answer**: Principal Component Analysis (PCA) performs orthogonal linear transformation to project high-dimensional features onto 2 or 3 principal axes maximizing variance. Its limitation is that non-linear manifold structures and fine-grained variance may be lost in low-dimensional projections.

### Q16: How does TF-IDF vectorization work?
**Answer**: TF-IDF combines Term Frequency (how often a term appears in a document) with Inverse Document Frequency ($\log \frac{N}{df}$, how rare the term is across the entire corpus). It gives high mathematical weight to terms that are discriminative for a specific document while penalizing ubiquitous stopwords.

### Q17: What does Cosine Similarity measure in text discovery?
**Answer**: Cosine Similarity measures the cosine of the angle between two normalized vectors: $\frac{u \cdot v}{\|u\| \|v\|}$. It evaluates directional orientation rather than vector magnitude, ensuring documents of differing lengths can be compared accurately on topical similarity.

### Q18: What is data leakage, and how did your project prevent it?
**Answer**: Data leakage occurs when information from outside the training partition influences model training (e.g., fitting scalers on the entire dataset before train/test splitting). Our project strictly splits raw data into 80% train and 20% test before calling `fit_transform()` exclusively on training data, using `transform()` on test data.

### Q19: What is the ARIMA model, and what do $(p, d, q)$ parameters represent?
**Answer**: Autoregressive Integrated Moving Average.
- $p$: Order of autoregressive terms (lags of the dependent variable).
- $d$: Degree of differencing required to achieve stationarity.
- $q$: Order of moving average error terms.

### Q20: Why was STL decomposition not applied directly to the annual time series?
**Answer**: Seasonal and Trend decomposition using Loess (STL) requires periodic seasonal data with at least two complete seasonal cycles ($m \ge 2$, e.g., quarterly or monthly timestamps). Annual data has only 1 observation per year, making sub-annual seasonal pattern estimation mathematically impossible.

### Q21: What are the five visualization categories in Unit V of the syllabus?
**Answer**:
1. **Pixel-Oriented**: Colors of individual cells represent values (Correlation Heatmap, Density Map).
2. **Geometric Projection**: Cartesian multidimensional mapping (Scatter Plots, 2D PCA Projections).
3. **Icon-Based**: Visual glyphs, cards, and badges representing multiple attributes.
4. **Hierarchical**: Nested dimensional breakdowns (Treemaps, Sunbursts).
5. **Complex Relations**: Relational and diagnostic graphs (Residual diagnostic plots, ROC curves, TF-IDF nearest-neighbor graphs).

### Q22: What role does SQLite play in this analytics project?
**Answer**: SQLite provides a lightweight, serverless relational database engine storing problem statements, discrete technology tags, and domain dimensions. It demonstrates SQL concepts (SELECT, WHERE, GROUP BY, HAVING, ORDER BY, and JOIN) and bridges file-based data with production relational storage.

### Q23: What are the limitations of synthetic data in this project?
**Answer**: Synthetic records simulate realistic attributes using domain heuristics and parametric formulas. While valuable for testing pipelines and demonstrating algorithms, synthetic target variables (`estimated_solution_days`) reflect heuristic assumptions rather than observed empirical human effort.

### Q24: How does Chi-Square test of independence work in this dataset?
**Answer**: It constructs a contingency table of observed frequencies ($O_{ij}$) between Problem Domain and AI Requirement, calculates expected frequencies ($E_{ij}$) under the null hypothesis of independence, and computes $\sum \frac{(O-E)^2}{E}$. A $p$-value $< 0.05$ confirms statistical dependence.

### Q25: How do you verify that the cleaning pipeline does not corrupt data upon repeated execution?
**Answer**: The cleaning function is idempotent and operates on an explicit copy of the raw dataset. Re-running cleaning on raw data consistently yields identical cleaned outputs without cumulative deletion of valid records.
