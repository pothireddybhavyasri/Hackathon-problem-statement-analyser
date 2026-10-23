# Data Sources and Provenance — Hackathon Insights

## 1. Provenance Overview

The **Hackathon Insights** project addresses problem statement discovery and multidimensional analytics for student, open-source, and corporate hackathon ecosystems.

To adhere strictly to academic integrity and prevent fabricating unverified official records:
- **Total Records in Dataset**: 524 records in raw file (520 base problem statements + 4 intentional duplicate rows).
- **Verified Sourced Records**: 130 records (25.0%) derived from legitimate public national, international, and collegiate hackathons across 2019–2024.
- **Transparent Synthetic Records**: 390 records (75.0%) generated via reproducible parametric templates (`seed=42`) using domain-specific ontologies, standard technical stacks, and realistic attribute correlation.
- **Field-Level Attribution**: The explicit column `data_type` with values `'sourced'` and `'synthetic'` allows users to filter, compare, and isolate records dynamically in every analysis and visualization.

---

## 2. Verified Sourced Hackathon Competitions

The sourced subset reflects official challenge themes, problem statements, and technical requirements from the following public hackathon repositories and governmental challenge portals:

1. **Smart India Hackathon (SIH)**
   - *Host*: Ministry of Education's Innovation Cell (MIC), Government of India, AICTE.
   - *Years*: 2019, 2020, 2021, 2022, 2023, 2024.
   - *Domains*: Smart Cities, Disaster Management, Agriculture, Transportation, Cybersecurity.
   - *Public Portal*: [https://sih.gov.in](https://sih.gov.in)
   - *Nature of Data*: Real problem statements submitted by central/state ministries and public sector undertakings.

2. **NASA International Space Apps Challenge**
   - *Host*: National Aeronautics and Space Administration (NASA), Space Apps Global Organizing Team.
   - *Years*: 2019, 2020, 2021, 2022, 2023, 2024.
   - *Domains*: Space Tech, Clean Energy, Earth Observation, Disaster Forecasting.
   - *Public Portal*: [https://www.spaceappschallenge.org](https://www.spaceappschallenge.org)
   - *Nature of Data*: Earth observation, lunar navigation, and climate challenges released under Open Government data licensing.

3. **MIT Grand Hacks & HackMIT**
   - *Host*: Massachusetts Institute of Technology (MIT Hacking Medicine & MIT TechX).
   - *Years*: 2019, 2020, 2021, 2022, 2023, 2024.
   - *Domains*: Healthcare, Biomedical Informatics, Clean Energy, Ambient Computing.
   - *Public Portals*: [https://grandhacks.mit.edu](https://grandhacks.mit.edu), [https://hackmit.org](https://hackmit.org)
   - *Nature of Data*: Clinical triage, assistive health technology, and telemetry challenges.

4. **ETHIndia & Web3 Hackathons**
   - *Host*: Devfolio & Ethereum Community.
   - *Years*: 2021, 2022, 2023, 2024.
   - *Domains*: FinTech, Decentralized Identity, Verifiable Carbon Markets.
   - *Public Portal*: [https://ethindia.co](https://ethindia.co)

5. **AI for Good Global Summit**
   - *Host*: International Telecommunication Union (ITU) in partnership with UN sister agencies.
   - *Years*: 2020, 2021, 2022, 2023, 2024.
   - *Domains*: Agriculture, Multilingual NLP, Assistive Technologies for Global Health.
   - *Public Portal*: [https://aiforgood.itu.int](https://aiforgood.itu.int)

6. **India AI & Open Government Data Challenges**
   - *Host*: National Informatics Centre (NIC), Ministry of Electronics and Information Technology (MeitY).
   - *Public Portal*: [https://indiaai.gov.in](https://indiaai.gov.in), [https://data.gov.in](https://data.gov.in)

---

## 3. Synthetic Data Generation Methodology

To ensure statistical power and adequate sample size for machine learning without scraping restricted enterprise portals:
- **Reproducibility**: Script `scripts/generate_dataset.py` utilizes NumPy and Python's `random` module with explicit `SEED = 42`.
- **Domain Ontologies**: 10 distinct domains (`Healthcare`, `FinTech`, `Agriculture`, `Smart Cities`, `Cybersecurity`, `Clean Energy`, `Disaster Management`, `Smart Education`, `Transportation`, `Space Tech`). Each contains realistic sub-domains, verified industry classifications, keywords, and realistic technology stack pools.
- **Attribute Correlation**: Estimated solution days are modeled with genuine covariance:
  $$\text{Days} \sim \mathcal{N}\left(9.0 + 5.5 \times \text{AI} + 7.5 \times \text{HW} + 11.0 \times \text{Adv}, \, 3.8^2\right)$$
  ensuring linear regression models reflect realistic latent relationships rather than pure random white noise.

---

## 4. Field Attribution and Estimation Notice

- **Officially Provided**: `hackathon_name`, `year`, `organizer_type`, `problem_title`, `problem_description`, and canonical `source_url` for sourced records.
- **Standardized / Categorized**: `domain`, `sub_domain`, `industry`, `technology_tags`.
- **Inferred / Estimated Rubrics**:
  - `estimated_solution_days`: Synthetic target variable representing estimated engineering effort for an undergraduate MVP. It is **not** an official competition deadline metric.
  - `innovation_score` & `impact_score`: Heuristic rubrics (1.0 to 10.0) based on algorithmic complexity and societal relevance.
  - `implementation_feasibility`: Rule-based assignment based on solution days and hardware dependencies.

---

## 5. Known Limitations and Potential Biases

1. **English Language Bias**: All problem titles and descriptions are in English; multilingual vernacular hackathons are represented via English translations.
2. **STEM & Software Over-Representation**: Hackathon challenges inherently emphasize software, algorithms, and prototype development over pure non-technical policy frameworks.
3. **Synthetic Target Approximation**: Regression performance on `estimated_solution_days` demonstrates algorithmic learning on domain heuristics and should be interpreted as proof-of-concept modeling rather than definitive empirical human project effort.
