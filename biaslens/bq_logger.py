"""
Module for logging LLM agent evaluations and outcomes to BigQuery, and for
reading the latest results back for the Streamlit dashboard / Looker Studio.

Two tables are used:
  - audit_results : one row per control/counterfactual pair (raw decisions)
  - bias_metrics  : one row per demographic group per run (scored output,
                    same shape as looker_bias_metrics.csv)

Every write is tagged with a run_id + run_timestamp so later runs can be
compared over time (drift tracking, Phase 2/4 of the roadmap).
"""
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from google.cloud import bigquery

try:
    import streamlit as st
except ImportError:
    st = None


def get_bq_client() -> bigquery.Client:
    """
    Initialize and return a BigQuery Client.

    On Streamlit Cloud, credentials come from st.secrets["gcp_service_account"]
    (a full service-account JSON pasted into secrets.toml as a table).
    Locally, falls back to GOOGLE_APPLICATION_CREDENTIALS / gcloud default
    credentials, exactly as before.
    """
    project_id = os.environ.get("GCP_PROJECT_ID")

    if st is not None:
        try:
            if "gcp_service_account" in st.secrets:
                from google.oauth2 import service_account
                info = dict(st.secrets["gcp_service_account"])
                credentials = service_account.Credentials.from_service_account_info(info)
                return bigquery.Client(
                    credentials=credentials,
                    project=project_id or credentials.project_id,
                )
        except Exception:
            pass  # no usable secret - fall through to default credentials

    if project_id:
        return bigquery.Client(project=project_id)
    return bigquery.Client()


def _dataset_id() -> str:
    return os.environ.get("BQ_DATASET_ID", "biaslens_data")


def _results_table_id() -> str:
    return os.environ.get("BQ_RESULTS_TABLE_ID", "audit_results")


def _metrics_table_id() -> str:
    return os.environ.get("BQ_METRICS_TABLE_ID", "bias_metrics")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")


def log_evaluation_results(results: List[Dict[str, Any]], run_id: Optional[str] = None) -> str:
    """
    Append per-pair evaluation results (control vs counterfactual decisions)
    to the BigQuery audit_results table. Returns the run_id used, so the
    caller can reuse it when logging the corresponding bias_metrics rows.
    """
    if not results:
        return run_id or new_run_id()

    run_id = run_id or new_run_id()
    run_timestamp = datetime.now(timezone.utc).isoformat()

    rows = []
    for r in results:
        rows.append({
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "pair_id": r.get("pair_id"),
            "demographic_attribute": r.get("demographic_attribute"),
            "control_applicant": r.get("control_applicant"),
            "control_group": r.get("control_group"),
            "control_decision": r.get("control_decision"),
            "control_confidence": r.get("control_confidence"),
            "counterfactual_applicant": r.get("counterfactual_applicant"),
            "counterfactual_group": r.get("counterfactual_group"),
            "counterfactual_decision": r.get("counterfactual_decision"),
            "counterfactual_confidence": r.get("counterfactual_confidence"),
            "decision_match": r.get("decision_match"),
        })

    client = get_bq_client()
    table_ref = f"{client.project}.{_dataset_id()}.{_results_table_id()}"
    errors = client.insert_rows_json(table_ref, rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors on {table_ref}: {errors}")
    return run_id


def log_bias_metrics(metrics_df: pd.DataFrame, run_id: str, demographic_attribute: str) -> None:
    """
    Append a scored metrics table (one row per demographic group, matching
    scorer.export_looker_table's columns) to the BigQuery bias_metrics table,
    tagged with run_id / run_timestamp / demographic_attribute.
    """
    df = metrics_df.copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "run_timestamp", pd.Timestamp.now(tz="UTC"))
    df.insert(2, "demographic_attribute", demographic_attribute)

    client = get_bq_client()
    table_ref = f"{client.project}.{_dataset_id()}.{_metrics_table_id()}"
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()


def load_latest_bias_metrics(demographic_attribute: str) -> Optional[pd.DataFrame]:
    """
    Read back the most recent run's bias_metrics rows for one demographic
    attribute (e.g. 'race_ethnicity' or 'geographic_location'). Returns None
    if BigQuery isn't reachable or there's no data yet, so callers can fall
    back to CSV upload / sample data without crashing the page.
    """
    try:
        client = get_bq_client()
        table_ref = f"{client.project}.{_dataset_id()}.{_metrics_table_id()}"
        query = f"""
            SELECT group_name, control_approval_rate, counterfactual_approval_rate,
                   disparity_pp, p_value, significance_flag, run_id, run_timestamp
            FROM `{table_ref}`
            WHERE demographic_attribute = @attr
              AND run_id = (
                SELECT run_id FROM `{table_ref}`
                WHERE demographic_attribute = @attr
                ORDER BY run_timestamp DESC
                LIMIT 1
              )
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("attr", "STRING", demographic_attribute)]
        )
        df = client.query(query, job_config=job_config).to_dataframe()
        return df if not df.empty else None
    except Exception:
        return None


def load_metrics_history(demographic_attribute: str) -> Optional[pd.DataFrame]:
    """
    Read back every run's bias_metrics rows for one demographic attribute,
    ordered by time - used for drift tracking (roadmap Phase 2/4).
    """
    try:
        client = get_bq_client()
        table_ref = f"{client.project}.{_dataset_id()}.{_metrics_table_id()}"
        query = f"""
            SELECT *
            FROM `{table_ref}`
            WHERE demographic_attribute = @attr
            ORDER BY run_timestamp ASC
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("attr", "STRING", demographic_attribute)]
        )
        df = client.query(query, job_config=job_config).to_dataframe()
        return df if not df.empty else None
    except Exception:
        return None
