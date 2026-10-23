# Streamlit Community Cloud Deployment Guide

This guide details the exact steps to deploy the **Hackathon Insights** platform to **Streamlit Community Cloud** from the GitHub repository.

---

## 1. Prerequisites

- A GitHub account with access to:
  `https://github.com/pothireddybhavyasri/Hackathon-problem-statement-analyser`
- A free account on [Streamlit Community Cloud](https://share.streamlit.io/).
- The project files committed and pushed to the `main` branch.

---

## 2. Deployment Steps

1. **Log in to Streamlit Community Cloud**:
   - Navigate to [https://share.streamlit.io/](https://share.streamlit.io/)
   - Sign in using the GitHub account `pothireddybhavyasri`.

2. **Initiate New App**:
   - In the Streamlit workspace dashboard, click the blue button **Create app** (or **New app**).
   - If prompted, select **"I already have an app"** (GitHub repository deploy).

3. **Configure Repository Parameters**:
   - **Repository**: `pothireddybhavyasri/Hackathon-problem-statement-analyser`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: You may customize your public subdomain (e.g., `hackathon-problem-statement-analyser.streamlit.app` or `hackathon-insights.streamlit.app`).

4. **Advanced Settings (Optional)**:
   - Click **Advanced settings**.
   - **Python version**: Select `3.11` or `3.12` or `3.13`.
   - **Secrets**: No external API keys or secrets are required. The platform operates self-contained with bundled datasets and embedded SQLite storage.

5. **Deploy**:
   - Click **Deploy!**.
   - Streamlit Cloud will launch a Debian container, clone the repository, install packages from `requirements.txt`, and start the app.

---

## 3. Post-Deployment Verification Checklist

Once the container finishes launching and the URL displays:
- [ ] Confirm the main Executive Dashboard loads with KPI scorecards (Total Problems: `520`, Verified Sourced: `130`, AI Required: `72.5%`).
- [ ] Test the sidebar navigation links to all pages.
- [ ] Navigate to **0 🔬 Data Analytics Lab** and click **Run Cleaning Pipeline** to verify in-memory computations.
- [ ] Navigate to **4 🤖 Machine Learning Lab** and verify model evaluation tables.
- [ ] Navigate to **5 🔍 Text Similarity & Discovery** and test a semantic keyword query.
- [ ] Navigate to **8 📖 Methodology & SQL** and execute a sample SQL query to verify the SQLite relational engine.
