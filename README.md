# BiasLens — Streamlit Dashboard

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Current state

This is a **UI-only build** — pages are wired up with placeholder/sample structure.
Real data connections are marked with `TODO` comments in `app.py`. To populate it:

1. **Race/Ethnicity page & Geographic page** — use the file uploader on each page to
   upload the CSVs exported by `scorer.py` (`looker_bias_metrics.csv` and
   `geo_looker_bias_metrics.csv`). The charts and tables will populate automatically
   once a CSV with the expected columns is uploaded:
   `group_name, control_approval_rate, counterfactual_approval_rate, disparity_pp, p_value, significance_flag`

2. **Run New Audit page** — currently a form-only placeholder. To make it live, call
   your existing `pair_generator.py` → `target_agent.py` → `scorer.py` pipeline from
   the `if submitted:` block in `app.py` and display the real result.

3. **Executive Summary boxes** — load the `explanation` field from
   `data/statistical_report.json` / `data/geo_statistical_report.json` and display it
   with `st.write()` in place of the current placeholder `st.info()` calls.

## Deploying (for the demo)

Easiest option: [share.streamlit.io](https://share.streamlit.io) — connect your GitHub
repo containing this folder, and it deploys for free with a public link you can pull up
during the jury presentation.
