# Hackathon Insights — Problem Statement Analysis

> **A complete, working, tested, and presentation-ready Data Analytics platform** built for analyzing hackathon problem statements across domains, technical stacks, difficulty levels, and historical timelines. Mapped directly to the university academic curriculum **CS513PE: DATA ANALYTICS** (B.Tech III-I CSE, R24 Regulations).

---

## 📌 Executive Summary

- **Primary Objective**: Analyze hackathon problem statements through data management, data cleaning, exploratory data analysis, inferential statistics, machine learning, text similarity, visualization, and time-series forecasting.
- **Dataset Size**: **524 total records** in raw dataset (520 base problem statements + 4 intentional duplicate rows), 23 schema columns.
- **Data Provenance**: **130 verified sourced records (25.0%)** from authentic public hackathons (*Smart India Hackathon SIH*, *NASA International Space Apps Challenge*, *MIT Grand Hacks*, *HackMIT*, *ETHIndia*, *AI for Good Global Summit*) and **390 transparently labeled synthetic records (75.0%)** generated via reproducible parametric templates (`seed=42`).
- **Test Suite**: **25 automated tests passed in 22s** using `pytest`.
- **Database Engine**: Relational SQLite database with parameterized execution and pre-built SQL demonstrations (SELECT, WHERE, GROUP BY, HAVING, ORDER BY, JOIN).

---

## 🛠️ Technology Stack

| Category | Technology / Library | Purpose in Project |
| :--- | :--- | :--- |
| **Language** | Python 3.13 | Core application and analytical logic |
| **Data Manipulation** | Pandas, NumPy | Data loading, transformation, cleaning, matrix computations |
| **Machine Learning** | Scikit-learn | Pipeline preprocessing, Linear Regression, Logistic Regression, Decision Trees, Random Forests, K-Means Clustering, TF-IDF Vectorization, Cosine Similarity |
| **Statistical Analysis** | SciPy, Statsmodels | Descriptive statistics, ANOVA, Chi-Square, t-tests, ARIMA time-series forecasting |
| **Interactive UI** | Streamlit | Multi-page web analytics platform with session state management |
| **Visualizations** | Plotly Express & Graph Objects | Interactive charts: Pixel-oriented heatmaps, Geometric scatter/PCA, Hierarchical treemaps, Residual diagnostics, ROC curves |
| **Relational Database**| SQLite (`sqlite3`) | Local relational storage, junction tables, parameterized queries, SQL studio |
| **Testing** | Pytest | Automated test runner verifying all 8 submodules |

---

## 📂 Project Structure

```
hackathon-insights/
├── app.py                             # Main Streamlit application with Overview Dashboard
├── requirements.txt                   # Complete frozen dependencies
├── README.md                          # Comprehensive project documentation
├── data/
│   ├── hackathon_problem_statements.csv           # Raw dataset (with intentional flaws)
│   ├── hackathon_problem_statements_cleaned.csv   # Cleaned dataset (520 clean records)
│   └── hackathon_insights.db                      # Local relational SQLite database
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # File discovery, loading, and dataset metadata
│   ├── data_validation.py             # 10 automated schema, range, boolean, and format checks
│   ├── data_cleaning.py               # IQR outlier detection and multi-stage cleaning pipeline
│   ├── eda.py                         # Technology tags extraction, domain breakdowns, variable taxonomy
│   ├── statistics_analysis.py         # 14 descriptive stats, correlations, ANOVA, Chi-Square, t-tests
│   ├── regression_models.py           # OLS Linear Regression, Ridge, Decision Tree, Random Forest, BLUE diagnostics
│   ├── classification_models.py       # Logistic Regression, Decision Trees (pruning), Random Forests, ROC curves
│   ├── clustering.py                  # K-Means clustering, Elbow method, Silhouette score, 2D PCA projection
│   ├── text_similarity.py             # TF-IDF vectorization and Cosine Similarity semantic search
│   ├── time_series_analysis.py        # Annual trends, AI adoption trajectory, ARIMA(1,1,0) forecasting
│   ├── visualizations.py              # Plotly interactive charts for Unit V visualization types
│   └── database.py                    # SQLite tables, junction table, parameterized SQL demos
├── pages/
│   ├── 0_🔬_Data_Analytics_Lab.py     # CENTRAL LAB: 7-stage sequential analytical workflow
│   ├── 1_📊_Data_Explorer.py          # Multidimensional filtering and record deep-dive inspector
│   ├── 2_🧹_Data_Quality_and_Cleaning.py # Deep-dive data quality audit and IQR outlier sandbox
│   ├── 3_📈_Exploratory_Analysis.py   # Univariate statistics, hierarchical treemaps, hypothesis testing
│   ├── 4_🤖_Machine_Learning_Lab.py   # Supervised ML studio, pruning curves, clustering, and model benchmarks
│   ├── 5_🔍_Text_Similarity_and_Discovery.py # Semantic problem search engine with similarity scores
│   ├── 6_📅_Trends_and_Forecasting.py # Multi-year growth, AI adoption rate, ARIMA time-series projection
│   ├── 7_🎓_Syllabus_Mapping.py       # Full interactive syllabus mapping for Units I through V
│   └── 8_📖_Methodology_and_SQL.py    # Relational SQLite studio, SQL query editor, mathematical methodology
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_data_validation.py
│   ├── test_data_cleaning.py
│   ├── test_eda_and_stats.py
│   ├── test_ml_models.py
│   └── test_misc_modules.py           # Text similarity, SQLite, and time-series tests
├── docs/
│   ├── project_report.md              # Complete 17-section formal academic project report
│   ├── data_dictionary.md             # Complete schema documentation (all 23 fields)
│   ├── data_sources.md                # Provenance, authentic sources, synthetic methodology, limitations
│   ├── syllabus_mapping.md            # Verified 5-unit curriculum alignment (CS513PE)
│   ├── viva_questions.md              # 25+ comprehensive viva questions with concise answers
│   └── methodology.md                 # Mathematical and algorithmic formulas and assumptions
└── scripts/
    ├── generate_dataset.py            # Reproducible raw dataset generator with controlled flaws
    └── run_tests.py                   # Automated test execution script
```

---

## ⚡ Installation & Launch Instructions

### 1. Prerequisites
- Python 3.10+ installed.
- Windows, macOS, or Linux.

### 2. Activate Virtual Environment
Navigate to the project root directory:
```bash
cd "C:\Users\Bhavy\.gemini\antigravity\scratch\hackathon-insights"
```
Activate the virtual environment:
- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The application will start locally at **`http://localhost:8501`**.

---

## 🧪 Running Automated Tests

Run the full automated test suite with Pytest:
```bash
pytest -v
```
Or execute the automated test runner script:
```bash
python scripts/run_tests.py
```
**Current Execution Result**: `25 passed in 22.13s` (100% test pass rate).

---

## 🎓 Academic Syllabus Alignment (`CS513PE: DATA ANALYTICS`)

Verified against `CSE_R24 B.Tech III-I Syllabus 2026-2027 (1).pdf` (Pages 12–13):

1. **Unit I: Data Management & Data Quality**: Dual data pipeline architecture, missing values, noise, duplicate handling, Tukey's Interquartile Range (IQR) outlier detection, automated multi-stage cleaning.
2. **Unit II: Data Analytics & Modeling**: Introduction to tools (Streamlit, Scikit-learn, SQLite), variable classification taxonomy (nominal, ordinal, interval, ratio), missing-value imputation (median/mode), 14 univariate descriptive statistics.
3. **Unit III: Regression & Classification**: Multiple Linear Regression (OLS), Gauss-Markov BLUE assumptions, residual diagnostics, Logistic Regression, Sigmoid link, Confusion Matrix, ROC-AUC curve.
4. **Unit IV: Segmentation & Time Series**: Supervised vs Unsupervised learning, Decision Trees with pruning and overfitting curves, Random Forest ensembles, K-Means clustering, ARIMA(1,1,0) forecasting with 95% confidence intervals.
5. **Unit V: Data Visualization Techniques**: Pixel-oriented correlation heatmaps, Geometric projection scatter/PCA plots, Icon-based KPI scorecards, Hierarchical Plotly Treemaps, and complex model diagnostic graphs.

---

## 🎯 Faculty Demonstration Sequence

When presenting the project to faculty or viva examiners:

1. **Overview Dashboard (`app.py`)**: Show total records (524 raw, 520 clean), sourced vs synthetic cohorts, AI requirement rate, and project motivation.
2. **Data Analytics Lab (`pages/0_🔬_Data_Analytics_Lab.py`)**:
   - **Step 1 (Load)**: Click *Load Raw Data* to display raw records and file info.
   - **Step 2 (Validate)**: Click *Execute Data Validation Audit* to show 10 structural rule checks.
   - **Step 3 (Inspect Quality)**: Click *Generate Quality Report* to show missing values and IQR outlier detection.
   - **Step 4 (Clean)**: Click *Run Cleaning Pipeline* to show missing values dropping to 0, duplicates eliminated, and review the operations log.
   - **Step 5 (Explore & Transform)**: Inspect 14 univariate statistics and the correlation heatmap.
   - **Step 6 (Train Models)**: Train OLS Linear Regression (predicting solution days), Logistic Regression (predicting AI requirement), and K-Means clustering.
   - **Step 7 (Export)**: Download the cleaned dataset, audit JSON, and model coefficients.
3. **Machine Learning Lab (`pages/4_🤖_Machine_Learning_Lab.py`)**: Show the Decision Tree pruning curve demonstrating overfitting across depths 1 to 10.
4. **Text Similarity & Discovery (`pages/5_🔍_Text_Similarity_and_Discovery.py`)**: Enter keywords (e.g., *"wildfire LoRaWAN sensor alerts"*) to demonstrate TF-IDF and Cosine Similarity matching.
5. **Trends & Forecasting (`pages/6_📅_Trends_and_Forecasting.py`)**: Show historical annual problem counts and the ARIMA forecast for 2025–2026 with 95% confidence bands.
6. **Syllabus Mapping (`pages/7_🎓_Syllabus_Mapping.py`)**: Display the verified 5-unit curriculum table with direct demonstration shortcuts.
7. **Relational SQL Studio (`pages/8_📖_Methodology_and_SQL.py`)**: Execute custom SQL queries demonstrating SELECT, WHERE, GROUP BY, HAVING, and JOIN.

---

## 📜 License & Attribution

- Built as an academic hands-on capstone project for B.Tech III Year I Semester.
- Authentic hackathon challenge records sourced under fair-use open education from Smart India Hackathon ([sih.gov.in](https://sih.gov.in)), NASA Space Apps Challenge ([spaceappschallenge.org](https://www.spaceappschallenge.org)), and MIT Grand Hacks ([grandhacks.mit.edu](https://grandhacks.mit.edu)).
