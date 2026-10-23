# Data Dictionary — Hackathon Problem Statements Analysis

This document describes the schema, data types, statistical properties, domain meanings, and provenance for all fields in the **Hackathon Insights** dataset.

## Primary Dataset Attributes

| Column Name | Data Type (Raw) | Data Type (Cleaned) | Description / Semantic Meaning | Valid Range / Categories | Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `problem_id` | String | String | Unique problem statement identifier (e.g., `PS-001`). Raw data contains whitespace and duplicate flaws. | Format: `PS-[0-9]{3}` | System generated / assigned identifier |
| `hackathon_name` | String | String | Name of the hosting hackathon competition. | E.g., Smart India Hackathon, NASA Space Apps Challenge, MIT Grand Hacks, HackMIT, ETHIndia, AI for Good Global Summit | Officially sourced / Curated |
| `year` | Integer | Integer | Calendar year in which the hackathon took place. | 2019 – 2024 | Officially provided / Sourced |
| `organizer_type` | String | Category | Organizational taxonomy of the organizing body. | `Government`, `Academic`, `Corporate`, `Open Source / NGO` | Officially sourced / Categorized |
| `problem_title` | String | String | Concise title summarizing the hackathon challenge. Raw data contains leading/trailing whitespaces. | Text (3 to 20 words) | Curated / Generated |
| `problem_description`| String | String | Detailed problem narrative explaining constraints, requirements, and domain background. | Text (25 to 120 words) | Curated / Generated |
| `domain` | String | Category | Primary thematic domain of the problem statement. Raw data contains mixed casing (e.g. `healthcare`, `fintech`). | `Healthcare`, `FinTech`, `Agriculture`, `Smart Cities`, `Cybersecurity`, `Clean Energy`, `Disaster Management`, `Smart Education`, `Transportation`, `Space Tech` | Sourced / Standardized |
| `sub_domain` | String | String | Specific specialized operational area within the domain. Raw data contains missing values. | E.g., `Critical Care`, `Wildfire Alert`, `Marine Ecology`, `Decentralized Finance` | Sourced / Generated |
| `industry` | String | Category | Target industrial sector adopting or sponsoring the solution. | E.g., `Medical & Life Sciences`, `Public Sector`, `Financial Services`, `AgriTech`, `Defense & Security` | Industry taxonomy mapped |
| `technology_tags` | String | String | Comma-delimited list of technologies, frameworks, and programming languages associated with the challenge. | E.g., `Python, PyTorch, React, OpenCV, Docker` | Sourced / Tagged |
| `ai_required` | Object (Mixed) | Integer (Binary) | Indicator flag if Artificial Intelligence / Machine Learning is needed to solve the problem. Raw data contains mixed strings (`YES`, `NO`, `1`, `0`). | `0` (No), `1` (Yes) | Inferred / Tagged |
| `hardware_required`| Object (Mixed) | Integer (Binary) | Indicator flag if embedded systems, IoT sensors, or specialized hardware are required. Raw data contains mixed strings (`True`, `False`). | `0` (No), `1` (Yes) | Inferred / Tagged |
| `data_availability` | String | Category | Accessibility level of training or validation data provided to hackathon participants. Raw data contains nulls. | `Open Data`, `Restricted`, `Simulated`, `None` | Sourced / Inferred |
| `difficulty_level` | String | Category | Perceived algorithmic and structural complexity level of the challenge. | `Beginner`, `Intermediate`, `Advanced` | Assigned by domain criteria |
| `estimated_solution_days` | Float | Float | Estimated calendar days required for an experienced student team to engineer an MVP. Target variable for regression models. Raw data contains missing values and extreme outliers. | Continuous, valid realistic range: `2.0` to `50.0` days (Raw contains outliers up to 365) | Domain heuristic estimate |
| `team_size` | Integer | Integer | Recommended team size for the hackathon team. Raw data contains outlier `99`. | Valid range: `2` to `6` members | Sourced / Curated |
| `innovation_score` | Float | Float | Evaluated technical novelty and uniqueness score (scale 1 to 10). Raw data contains nulls. | `1.0` to `10.0` | Algorithmic heuristic / expert rubric |
| `impact_score` | Float | Float | Evaluated societal and commercial impact score (scale 1 to 10). | `1.0` to `10.0` | Algorithmic heuristic / expert rubric |
| `implementation_feasibility` | String | Category | Assessed feasibility of completing a working MVP within standard hackathon constraints. | `High`, `Medium`, `Low` | Derived from estimated days & hardware dependencies |
| `problem_length_words` | Integer | Integer | Word count of the problem description. Recomputed dynamically during cleaning. | Positive integer | Derived feature |
| `number_of_technologies` | Integer | Integer | Count of discrete technology tools parsed from `technology_tags`. Recomputed dynamically during cleaning. | Non-negative integer | Derived feature |
| `source_url` | String | String | Canonical URL of the hosting competition or open challenge archive. | Valid URL format | Legitimate public hackathon portals for sourced records |
| `data_type` | String | Category | Transparent attribution label distinguishing verified public hackathon records from generated demonstration records. | `sourced`, `synthetic` | Dataset provenance flag |

---

## Controlled Data Quality Issues Injected in Raw Dataset

To rigorously evaluate the Data Quality Analysis (Step 3) and Data Cleaning Pipeline (Step 4), the raw CSV contains deliberate real-world flaws:
1. **Duplicate Records**: 4 exact duplicate rows appended at the end of the file.
2. **Duplicate Identifiers**: Reused IDs with slight trailing whitespace (`PS-042 `) or identical keys (`PS-080`).
3. **Missing Values**:
   - `sub_domain`: 8 missing entries.
   - `data_availability`: 12 missing entries.
   - `technology_tags`: 6 missing entries.
   - `estimated_solution_days`: 7 missing numerical values.
   - `innovation_score`: 5 missing numerical values.
4. **Casing Inconsistencies**: Mixed lowercase/uppercase domain names (e.g., `healthcare`, `fintech`, `smart cities`, `CYBERSECURITY`).
5. **Whitespace Anomalies**: 15 problem titles and domain strings with leading/trailing spaces.
6. **Mixed Data Types**: `ai_required` containing string `"YES"`/`"NO"` alongside integers, and `hardware_required` containing string `"True"`/`"False"`.
7. **Numerical Outliers**: Extreme values in `team_size` (`99`) and `estimated_solution_days` (`365.0`, `280.0`).
8. **Short / Low Quality Text**: Problem descriptions with placeholder strings (`TBD soon.`, `Under review.`).
