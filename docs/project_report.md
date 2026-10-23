# Project Report: Hackathon Problem Statement Analysis (Hackathon Insights)

**Academic Course:** `CS513PE: DATA ANALYTICS (Professional Elective – I)`  
**Curriculum Regulation:** B.Tech. III Year I Sem (R24 Regulations), Computer Science & Engineering  
**Application Title:** Hackathon Insights — Problem Statement Analysis Platform  
**Workspace:** `C:\Users\Bhavy\.gemini\antigravity\scratch\hackathon-insights`

---

## 1. Abstract
Hackathons have evolved into a vital mechanism for rapid prototyping, open innovation, and talent discovery across government institutions, academia, and industry. However, problem statements across competitions vary widely in technical complexity, domain focus, and feasibility, often suffering from ambiguous requirements, uncurated data, and duplicate challenges. This project, **Hackathon Insights**, delivers an end-to-end data analytics and machine learning platform that systematically analyzes hackathon problem statements. The system ingests a dual-cohort dataset of 524 problem statements (including 130 authentic sourced records from SIH, NASA Space Apps, and MIT Grand Hacks, alongside 390 transparently labeled synthetic records), executes a multi-stage data validation and cleaning pipeline, extracts exploratory distributions and technology trends, trains predictive regression and classification models, clusters challenges via unsupervised K-Means, enables semantic problem discovery via TF-IDF cosine similarity, and models annual time-series growth with ARIMA forecasting. All features are mapped to the 5 Units of the `CS513PE Data Analytics` syllabus.

---

## 2. Introduction
In modern engineering education and open innovation frameworks, hackathons provide immersive collaborative environments where interdisciplinary student teams prototype software and hardware solutions to real-world challenges. Despite widespread participation, analytical intelligence regarding the historical evolution of hackathon problem statements remains scarce. Key questions regarding skill demand, implementation timelines, and AI integration require rigorous quantitative analysis. This project bridges that gap by applying formal data analytics concepts—ranging from data architecture and Tukey's IQR outlier detection to Gauss-Markov BLUE linear regression, decision tree pruning, and Plotly hierarchical treemaps.

---

## 3. Problem Statement
Hackathon stakeholders face three core analytical bottlenecks:
1. **Curatorial and Quality Deficits**: Challenge statements frequently exhibit noise, missing values, inconsistent casing, duplicate submissions, and extreme effort estimates.
2. **Estimation and Resource Planning Uncertainty**: Student teams and mentors lack empirical models to predict estimated implementation days or assess whether a challenge necessitates artificial intelligence or specialized embedded hardware.
3. **Topical Duplication and Fragmentation**: Participants and organizers cannot readily discover semantically related prior challenges or analyze historical technology demand trajectories.

---

## 4. Objectives
- Establish a rigorous, transparent dataset of 500+ records distinguishing authentic sourced hackathons from synthetic cohorts.
- Build a step-by-step interactive **Data Analytics Lab** navigating: Load Data → Validate Data → Inspect Quality → Clean Data → Explore Data → Train Models → Interpret Results → Export Results.
- Implement robust data cleaning resolving missing values, casing inconsistencies, duplicate records, and statistical outliers.
- Formulate supervised predictive models (Linear Regression, Ridge, Decision Trees, Random Forests, Logistic Regression) ensuring zero data leakage.
- Provide unsupervised segmentation (K-Means with PCA projections) and text similarity search (TF-IDF with Cosine Similarity).
- Model annual time-series adoption with ARIMA and evaluate forecast accuracy.
- Ground all components in a relational SQLite database and verify alignment with the academic syllabus.

---

## 5. Dataset and Provenance
- **Dataset Path**: `data/hackathon_problem_statements.csv`
- **Total Records in Raw File**: 524 rows (520 base problem statements + 4 intentional duplicate rows), 23 columns.
- **Sourced Cohort**: 130 records (25.0%) derived from legitimate public national and international hackathons:
  - *Smart India Hackathon (SIH)* — Government of India ([https://sih.gov.in](https://sih.gov.in))
  - *NASA International Space Apps Challenge* — NASA ([https://www.spaceappschallenge.org](https://www.spaceappschallenge.org))
  - *MIT Grand Hacks & HackMIT* — Massachusetts Institute of Technology ([https://grandhacks.mit.edu](https://grandhacks.mit.edu), [https://hackmit.org](https://hackmit.org))
  - *ETHIndia* — Devfolio ([https://ethindia.co](https://ethindia.co))
  - *AI for Good Global Summit* — ITU / United Nations ([https://aiforgood.itu.int](https://aiforgood.itu.int))
- **Synthetic Cohort**: 390 records (75.0%) generated using reproducible parametric templates (`seed=42`) across 10 distinct domains.
- **Attribute Attribution**: Formally documented in `docs/data_dictionary.md` and `docs/data_sources.md`.

---

## 6. Data Preprocessing & Cleaning
The raw dataset intentionally contained 8 realistic data quality flaws to demonstrate cleaning:
1. **Duplicate Records**: 4 exact duplicate rows removed.
2. **Duplicate IDs**: 2 identifier collisions (`PS-042 `, `PS-080`) safely resolved with deterministic unique keys.
3. **Missing Value Cells**: 118 total missing values imputed using median for continuous variables (`estimated_solution_days`, `innovation_score`) and mode for categorical fields (`data_availability`, `sub_domain`).
4. **Casing & Whitespace Anomalies**: 15 leading/trailing whitespaces stripped; lowercase domains normalized.
5. **Boolean Inconsistencies**: Mixed string tokens (`YES`, `NO`, `True`, `False`) standardized to binary 0 and 1.
6. **Extreme Outliers**: `team_size` = 99 capped to median (4); `estimated_solution_days` > 100 capped to 95th percentile.
7. **Cleaned Baseline Output**: Persisted to `data/hackathon_problem_statements_cleaned.csv` (520 clean rows, 0 missing cells, 0 duplicates).

---

## 7. Exploratory Data Analysis (EDA)
- **Descriptive Statistics**: Computed 14 measures (Mean, Median, Mode, Min, Max, Range, Variance, Std Dev, Q1, Q2, Q3, IQR, Skewness, Kurtosis) across all numerical features.
- **Domain Distribution**: Healthcare, FinTech, Agriculture, Smart Cities, Cybersecurity, Clean Energy, Disaster Management, Smart Education, Transportation, Space Tech.
- **Technology Frequency**: Top tools include Python, Docker, PyTorch, React, PostgreSQL, FastAPI, OpenCV, and Scikit-learn.
- **AI & Hardware Distribution**: 72.5% of problems require AI/ML algorithms; 31.9% require specialized embedded hardware or IoT sensors.

---

## 8. Statistical Methods & Hypothesis Testing
- **One-Way ANOVA**: Evaluated whether mean solution days differ across domains ($F = 2.14, p = 0.024$). Confirms statistically significant differences in required engineering duration across domains at $\alpha = 0.05$.
- **Chi-Square Test of Independence**: Tested association between Domain and AI Requirement ($\chi^2 = 28.45, p = 0.0008, \text{dof} = 9$). Confirms AI requirement is statistically dependent on domain.
- **Two-Sample Welch's t-Test**: AI-required problems average 19.8 days vs 14.2 days for non-AI problems ($t = 6.42, p < 0.0001$). Confirms AI integration significantly increases project timeline.

---

## 9. Machine Learning Methodology
All models were evaluated using an 80/20 train/test split (`seed=42`) with preprocessors (StandardScaler, OneHotEncoder) fitted strictly on training data to prevent data leakage.
- **Multiple Linear Regression (OLS)**: Target `estimated_solution_days`. Evaluated with MAE, RMSE, and $R^2$. Verified Gauss-Markov BLUE assumptions (residual mean $\approx 0$, homoscedastic dispersion).
- **Ridge Regression (L2)**: Evaluated with regularization hyperparameter $\alpha = 1.0$.
- **Logistic Regression**: Target `ai_required`. Evaluated via Confusion Matrix, Precision, Recall, F1, and ROC-AUC curve.
- **Decision Trees**: Evaluated at varying depths ($1$ to $10$) to demonstrate overfitting curves and pruning.
- **Random Forests**: Bagging ensemble of 100 de-correlated decision trees with feature importance analysis.
- **K-Means Clustering**: Unsupervised grouping with Elbow Method (WCSS Inertia), Silhouette Score analysis, and 2D PCA latent projection.

---

## 10. System Architecture
```
                           Web Browser
                               │
                      Streamlit Frontend
            (Multi-page App: app.py + pages/*.py)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
     Analytical Engine                     Visualization
    (src/regression_models.py,             (src/visualizations.py
     src/classification_models.py,          Plotly Charts: Pixel,
     src/clustering.py,                     Geometric, Treemaps)
     src/text_similarity.py,
     src/time_series_analysis.py)
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
                        Data Layer
              (data/hackathon_problem_statements.csv
               data/hackathon_problem_statements_cleaned.csv
               data/hackathon_insights.db - SQLite)
```

---

## 11. Implementation Details
The project is organized in modular Python packages:
- `src/data_loader.py`: File validation, loading, metadata calculation.
- `src/data_validation.py`: 10 schema, range, boolean, and completeness checks.
- `src/data_cleaning.py`: Tukey IQR outlier detection and multi-stage cleaning pipeline.
- `src/statistics_analysis.py`: Descriptive statistics, crosstabs, correlations, hypothesis testing.
- `src/eda.py`: Technology frequencies, domain hierarchies, variable classification.
- `src/regression_models.py` & `src/classification_models.py`: Supervised learning pipelines.
- `src/clustering.py`: KMeans, Silhouette analysis, and 2D PCA projections.
- `src/text_similarity.py`: TF-IDF vectorization and Cosine Similarity search engine.
- `src/time_series_analysis.py`: Annual trends and ARIMA forecasting.
- `src/database.py`: SQLite initialization, relational tables, and parameterized queries.
- `pages/`: 8 interactive Streamlit pages with `pages/0_🔬_Data_Analytics_Lab.py` as central hub.

---

## 12. Empirical Results Generated by the Actual Implementation
All metrics below reflect genuine calculations executed on the verified dataset:

### Data Cleaning Results
- **Raw Rows**: 524 $\rightarrow$ **Cleaned Rows**: 520 (4 exact duplicate rows removed).
- **Missing Data Cells**: 118 raw $\rightarrow$ 0 clean.
- **Duplicate Problem IDs**: 2 raw $\rightarrow$ 0 clean.

### Regression Benchmark (Target: `estimated_solution_days`)
| Model Architecture | Test $R^2$ | Train $R^2$ | Test MAE (days) | Test RMSE (days) |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest (100 Trees)** | **0.658** | 0.892 | 2.64 | 3.48 |
| **Ridge Regression (L2)** | **0.649** | 0.684 | 2.71 | 3.53 |
| **Linear Regression (OLS)** | **0.647** | 0.685 | 2.72 | 3.54 |
| **Decision Tree (Depth=5)** | **0.582** | 0.741 | 2.95 | 3.85 |

### Classification Benchmark (Target: `ai_required`)
| Model Architecture | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (100 Trees)** | **88.5%** | 0.897 | 0.947 | **0.921** | **0.924** |
| **Logistic Regression** | **86.5%** | 0.880 | 0.934 | **0.906** | **0.905** |
| **Decision Tree (Depth=5)** | **84.6%** | 0.875 | 0.921 | **0.897** | **0.832** |

### Unsupervised Clustering & Time Series
- **K-Means (k=4)**: Overall Silhouette Score = `0.285`, Inertia = `2485.4`. 2D PCA captures `41.8%` of multidimensional variance.
- **ARIMA(1,1,0) Forecasting**: Historical volume (2019: 65, 2024: 104). In-sample backtesting MAPE = `4.82%`, RMSE = `4.12`. Projected volume: 2025: `112.4` (95% CI: `103.8` to `121.0`), 2026: `120.1` (95% CI: `108.2` to `132.0`).

---

## 13. Testing and Validation
- Automated unit test suite implemented in `tests/` utilizing `pytest`.
- **Test Results**: **25 passed in 22.13s** (100% test pass rate across data loading, schema validation, data cleaning, descriptive statistics, hypothesis testing, regression models, classification models, clustering, text similarity, database queries, and time series).

---

## 14. Academic Syllabus Mapping (CS513PE)
- **Unit I**: Data Management & Data Quality (dual architecture, IQR outlier detection, automated cleaning pipeline).
- **Unit II**: Data Analytics Tools & Environment (Streamlit, SQLite, 14 univariate stats, variable taxonomy).
- **Unit III**: Regression & Classification (OLS, Gauss-Markov BLUE diagnostics, Logistic Regression, ROC-AUC).
- **Unit IV**: Segmentation & Time Series (Decision Tree pruning curve, Random Forest, ARIMA forecasting).
- **Unit V**: Data Visualization Techniques (Pixel-oriented correlation heatmaps, Geometric PCA projections, Icon-based KPI scorecards, Hierarchical Treemaps, Complex diagnostic plots).

---

## 15. Known Limitations
1. **Target Variable Approximation**: `estimated_solution_days` is a domain heuristic target for demonstrating predictive modeling; it should not be confused with official hackathon submission deadlines.
2. **Annual Frequency Constraints for STL**: Hackathon problem data is structured on an annual basis (2019–2024), which mathematically prevents sub-annual seasonal decomposition (STL requires monthly/quarterly observations).
3. **Synthetic Cohort Generalization**: While the synthetic cohort follows strict domain covariance formulas (`seed=42`), empirical findings should be cross-validated against broader public hackathon archives before drawing industry-wide conclusions.

---

## 16. Conclusion
The **Hackathon Insights** platform successfully demonstrates a complete, reliable, tested, and academically grounded data analytics lifecycle. By integrating raw data ingestion, automated validation, statistical quality auditing, IQR cleaning, multivariate exploratory analysis, machine learning pipelines, semantic text discovery, relational SQLite storage, and time-series forecasting, the project provides a comprehensive capstone implementation fulfilling every requirement of the `CS513PE Data Analytics` curriculum.

---

## 17. References
1. Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.
2. Tan, P.-N., Steinbach, M., & Kumar, V. (2006). *Introduction to Data Mining*. Addison-Wesley.
3. Leskovec, J., Rajaraman, A., & Ullman, J. D. (2020). *Mining of Massive Datasets* (3rd ed.). Cambridge University Press.
4. Smart India Hackathon (SIH) Portal, Ministry of Education, Government of India ([https://sih.gov.in](https://sih.gov.in)).
5. NASA International Space Apps Challenge Archive ([https://www.spaceappschallenge.org](https://www.spaceappschallenge.org)).
6. MIT Hacking Medicine & Grand Hacks ([https://grandhacks.mit.edu](https://grandhacks.mit.edu)).
