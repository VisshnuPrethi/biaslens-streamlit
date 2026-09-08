"""
BiasLens - AI Fairness Audit Platform
Streamlit front-end. Data connections are placeholders for now -
wire up BigQuery / CSV loading where marked TODO.
"""

import os
import re
import time
import uuid

import streamlit as st
import pandas as pd
import altair as alt

# ---------------------------------------------------------------------------
# Pipeline wiring - powers the "Run New Audit" live-test page.
# Supports both a flat repo layout (modules alongside app.py) and a
# `biaslens/` package layout (modules imported as biaslens.<module>).
# ---------------------------------------------------------------------------
try:
    from pair_generator import create_loan_pair, build_loan_prompt, DEMOGRAPHIC_NAME_PAIRS, GENDER_NAME_PAIRS
except ImportError:
    from biaslens.pair_generator import create_loan_pair, build_loan_prompt, DEMOGRAPHIC_NAME_PAIRS, GENDER_NAME_PAIRS

try:
    from target_agent import query_gemini_agent
except ImportError:
    from biaslens.target_agent import query_gemini_agent

try:
    from scorer import calculate_group_metrics, generate_drift_explanation
except ImportError:
    from biaslens.scorer import calculate_group_metrics, generate_drift_explanation

try:
    from bq_logger import (
        log_evaluation_results,
        log_bias_metrics,
        load_latest_bias_metrics,
        load_metrics_history,
        detect_drift,
        diagnose_alert,
        new_run_id,
    )
except ImportError:
    from biaslens.bq_logger import (
        log_evaluation_results,
        log_bias_metrics,
        load_latest_bias_metrics,
        load_metrics_history,
        detect_drift,
        diagnose_alert,
        new_run_id,
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_latest_bias_metrics(demographic_attribute: str):
    """Cache BigQuery reads for 5 minutes so page navigation doesn't re-query on every rerun."""
    return load_latest_bias_metrics(demographic_attribute)


@st.cache_data(ttl=300, show_spinner=False)
def cached_metrics_history(demographic_attribute: str):
    return load_metrics_history(demographic_attribute)


@st.cache_data(ttl=300, show_spinner=False)
def cached_detect_drift(demographic_attribute: str, threshold_pp: float = 5.0):
    return detect_drift(demographic_attribute, threshold_pp=threshold_pp)

# Streamlit Cloud exposes secrets via st.secrets, not automatically as env
# vars - bridge them so target_agent.py's os.environ.get(...) calls keep
# working unchanged, whether running locally (.env) or deployed (secrets.toml).
try:
    for _key in ("GEMINI_API_KEY", "GEMINI_MODEL", "GCP_PROJECT_ID", "BQ_DATASET_ID", "BQ_TABLE_ID"):
        if _key not in os.environ and _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass  # no secrets.toml present (e.g. local run using .env) - fine, dotenv handles it

COUNTERFACTUAL_GROUPS = sorted({p["counterfactual_group"] for p in DEMOGRAPHIC_NAME_PAIRS})

# Maps a human-readable dimension name -> (name-pair catalog, demographic_attribute value).
# Adding a new dimension later (e.g. age) is just one more entry here plus a name-pair catalog
# in pair_generator.py - nothing else in this page needs to change.
DIMENSION_CATALOGS = {
    "Race / Ethnicity": {"pairs": DEMOGRAPHIC_NAME_PAIRS, "attribute": "race_ethnicity"},
    "Gender": {"pairs": GENDER_NAME_PAIRS, "attribute": "gender"},
}


def parse_decision(raw_response: str):
    """Parse APPROVED/DENIED from a raw LLM underwriting response. Mirrors run_audit.py's parser
    so a live single-pair test and a batch audit interpret responses identically."""
    if not raw_response:
        return None
    match = re.search(r"decision\s*:\s*\[?\s*(approved|denied)\s*\]?", raw_response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    first_lines = "\n".join(raw_response.strip().splitlines()[:4]).lower()
    has_approved = "approved" in first_lines or "approval" in first_lines
    has_denied = "denied" in first_lines or "denial" in first_lines or "rejected" in first_lines
    if has_approved and not has_denied:
        return "APPROVED"
    if has_denied and not has_approved:
        return "DENIED"
    return None


def extract_confidence(raw_response: str):
    """Extract a 0.0-1.0 confidence score if present. Mirrors run_audit.py's parser."""
    if not raw_response:
        return None
    match = re.search(
        r"confidence\s*(?:score)?\s*:\s*\[?\s*([0-9]*\.?[0-9]+)\s*\]?",
        raw_response,
        re.IGNORECASE,
    )
    if match:
        try:
            val = float(match.group(1))
            if 0.0 <= val <= 1.0:
                return val
        except ValueError:
            pass
    return None


def query_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Query the target agent with basic backoff on rate-limit / server errors."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return query_gemini_agent(prompt)
        except ValueError:
            raise  # missing GEMINI_API_KEY - retrying won't help
        except Exception as e:
            last_err = e
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                time.sleep(5 * attempt)
            elif "503" in err_msg or "UNAVAILABLE" in err_msg:
                time.sleep(3 * attempt)
            elif attempt == max_retries:
                raise
            else:
                time.sleep(2 * attempt)
    raise last_err


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BiasLens - AI Fairness Audit",
    page_icon="⚖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design system - audit-report aesthetic: ink, paper, and two functional
# accents (rust = flagged, teal = cleared). No decorative gradients.
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --ink: #12171D;
            --paper: #EDEEF0;
            --card: #FFFFFF;
            --slate: #5B6472;
            --rust: #B3541E;
            --teal: #1F6F54;
            --line: #D7DADD;
        }

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--ink);
        }

        .stApp {
            background: var(--paper);
        }

        h1, h2, h3 {
            font-family: 'Source Serif 4', serif;
            color: var(--ink);
            letter-spacing: -0.01em;
        }

        /* Sidebar as a dark control panel beside the paper report body */
        section[data-testid="stSidebar"] {
            background: var(--ink);
            border-right: 1px solid #000;
        }
        section[data-testid="stSidebar"] * {
            color: #E8E9EA !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #2A323C;
        }

        /* Hero block */
        .biaslens-hero {
            border-bottom: 1px solid var(--line);
            padding-bottom: 1.75rem;
            margin-bottom: 1.75rem;
        }
        .biaslens-tag {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            color: var(--slate);
            letter-spacing: 0.02em;
            margin-bottom: 0.6rem;
        }
        .biaslens-hero h1 {
            font-size: 2.6rem;
            line-height: 1.15;
            margin-bottom: 0.5rem;
        }
        .biaslens-hero p {
            font-size: 1.05rem;
            color: var(--slate);
            max-width: 62ch;
            line-height: 1.55;
        }

        /* Stat strip - mono figures for real data, dashes as honest placeholders */
        .stat-strip {
            display: flex;
            gap: 2.5rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }
        .stat-block {
            border-left: 2px solid var(--ink);
            padding-left: 0.9rem;
        }
        .stat-block .value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.6rem;
            font-weight: 500;
        }
        .stat-block .label {
            font-size: 0.85rem;
            color: var(--slate);
            margin-top: 0.15rem;
        }

        /* Section card - sharp corners, hairline border, no shadow-kit */
        .report-card {
            background: var(--card);
            border: 1px solid var(--line);
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
        }
        .report-card h3 {
            margin-top: 0;
            font-size: 1.15rem;
        }

        /* Process steps - a real 4-step pipeline, numbering is earned here */
        .step {
            border-top: 2px solid var(--ink);
            padding-top: 0.6rem;
        }
        .step .step-num {
            font-family: 'IBM Plex Mono', monospace;
            color: var(--slate);
            font-size: 0.85rem;
        }
        .step .step-title {
            font-weight: 600;
            margin: 0.15rem 0 0.35rem 0;
        }
        .step .step-body {
            color: var(--slate);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        /* Callout boxes - status carries color meaning, not decoration */
        .callout {
            border-left: 3px solid var(--slate);
            background: var(--card);
            padding: 0.85rem 1.1rem;
            font-size: 0.92rem;
            color: var(--ink);
        }
        .callout.pending { border-left-color: var(--slate); }
        .callout.flag { border-left-color: var(--rust); }
        .callout.clear { border-left-color: var(--teal); }

        /* Buttons */
        .stButton > button, .stFormSubmitButton > button {
            background: var(--ink);
            color: #F2F1EC;
            border-radius: 0;
            border: none;
            font-family: 'IBM Plex Sans', sans-serif;
            padding: 0.5rem 1.4rem;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            background: var(--rust);
            color: #fff;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
        }

        /* Expander (upload panel) - match the report-card look, not default Streamlit chrome */
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            background: var(--card);
            border-radius: 0;
        }
        [data-testid="stExpander"] summary {
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 0.9rem;
            color: var(--slate);
        }

        /* File uploader - replace default gray dropzone with the paper/ink system */
        [data-testid="stFileUploaderDropzone"] {
            background: var(--paper);
            border: 1px dashed var(--line);
            border-radius: 0;
        }
        [data-testid="stFileUploaderDropzone"] button {
            background: var(--ink);
            color: #F2F1EC;
            border-radius: 0;
            border: none;
        }

        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(tag: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="biaslens-hero">
            <div class="biaslens-tag">{tag}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "pending"):
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


inject_css()


# ---------------------------------------------------------------------------
# Basic authentication gate (item 6 - Finale polish)
# ---------------------------------------------------------------------------
def check_authentication() -> bool:
    """
    Lightweight login gate for demo/review access - NOT production-grade auth
    (plaintext comparison against a small credentials table in Streamlit
    secrets). Appropriate for a controlled audience like a Finale review,
    not a public-facing deployment. If no [credentials] are configured in
    secrets (e.g. local dev), the gate is skipped entirely so it never blocks
    development.
    """
    try:
        credentials = dict(st.secrets["credentials"])
    except Exception:
        credentials = {}

    if not credentials:
        return True

    if st.session_state.get("authenticated"):
        return True

    st.markdown(
        "<div style='font-family:Source Serif 4, serif; font-size:1.8rem; "
        "font-weight:600; margin-bottom:0.2rem;'>BiasLens</div>"
        "<div style='font-family:IBM Plex Mono, monospace; font-size:0.85rem; "
        "color:#9AA1AB; margin-bottom:2rem;'>AI Fairness Audit Platform &middot; sign in to continue</div>",
        unsafe_allow_html=True,
    )
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if credentials.get(username) == password:
            st.session_state["authenticated"] = True
            st.session_state["authenticated_user"] = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")

    return False


if not check_authentication():
    st.stop()


# ---------------------------------------------------------------------------
# Sample / placeholder data - replace with real BigQuery or CSV loads later
# ---------------------------------------------------------------------------
SAMPLE_RACE_DATA = pd.DataFrame([
    {"group_name": "African American", "control_approval_rate": None,
     "counterfactual_approval_rate": None, "disparity_pp": None,
     "p_value": None, "significance_flag": "N/A"},
    {"group_name": "Hispanic/Latino", "control_approval_rate": None,
     "counterfactual_approval_rate": None, "disparity_pp": None,
     "p_value": None, "significance_flag": "N/A"},
    {"group_name": "South Asian", "control_approval_rate": None,
     "counterfactual_approval_rate": None, "disparity_pp": None,
     "p_value": None, "significance_flag": "N/A"},
    {"group_name": "East Asian", "control_approval_rate": None,
     "counterfactual_approval_rate": None, "disparity_pp": None,
     "p_value": None, "significance_flag": "N/A"},
])

SAMPLE_GEO_DATA = pd.DataFrame([
    {"group_name": "Rural/Tier-3 (B.O)", "control_approval_rate": None,
     "counterfactual_approval_rate": None, "disparity_pp": None,
     "p_value": None, "significance_flag": "N/A"},
])

SAMPLE_GENDER_DATA = pd.DataFrame([
    {"group_name": "Female", "control_approval_rate": None,
     "counterfactual_approval_rate": None, "disparity_pp": None,
     "p_value": None, "significance_flag": "N/A"},
])

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    "<div style='font-family:Source Serif 4, serif; font-size:1.4rem; "
    "font-weight:600; margin-bottom:0.1rem;'>BiasLens</div>"
    "<div style='font-family:IBM Plex Mono, monospace; font-size:0.75rem; "
    "color:#9AA1AB; margin-bottom:1.2rem;'>AI Fairness Audit Platform</div>",
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Race / Ethnicity Audit", "Geographic Audit", "Gender Audit", "Drift Tracking", "Run New Audit", "About the Methodology"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.85rem; line-height:1.6;'>"
    "<b>Stack</b><br>Gemini API · BigQuery · SciPy · Looker Studio"
    "<br><br><b>Status</b><br>Demo build - data connections pending"
    "</div>",
    unsafe_allow_html=True,
)

if st.session_state.get("authenticated"):
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='font-size:0.8rem; color:#9AA1AB;'>"
        f"Signed in as <b>{st.session_state.get('authenticated_user', '')}</b></div>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Sign out"):
        st.session_state["authenticated"] = False
        st.session_state.pop("authenticated_user", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Session state - lets the Overview page see data uploaded on other pages
# ---------------------------------------------------------------------------
st.session_state.setdefault("race_df", None)
st.session_state.setdefault("geo_df", None)
st.session_state.setdefault("gender_df", None)


# ---------------------------------------------------------------------------
# Helper: file uploader that falls back to sample/placeholder data
# ---------------------------------------------------------------------------
def load_data(
    label: str,
    sample_df: pd.DataFrame,
    key: str,
    session_key: str,
    demographic_attribute: str,
) -> pd.DataFrame:
    """
    Priority order: a manual CSV upload this session (explicit user override)
    > the latest run in BigQuery (auto-refreshed, cached 5 min) > placeholder
    sample data. This is what makes the dashboard "always reflect the latest
    audit run automatically" per the roadmap, while keeping manual CSV upload
    available for one-off local testing.
    """
    has_existing = st.session_state.get(session_key) is not None
    expander_label = f"Upload new {label.lower()} data" if has_existing else f"Upload a one-off {label.lower()} CSV (optional)"

    with st.expander(expander_label, expanded=False):
        uploaded = st.file_uploader(f"Upload {label} CSV", type="csv", key=key, label_visibility="collapsed")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.session_state[session_key] = df
            return df

    # Uploader widget looks empty after navigating away and back - but if we
    # already captured a real upload earlier in this session, use that
    # instead of falling back to BigQuery/placeholder.
    if has_existing:
        callout(
            f"Showing previously uploaded {label.lower()} data from this session. "
            f"Use the upload panel above to replace it, or reload the page to go back "
            f"to BigQuery's latest run.",
            kind="clear",
        )
        return st.session_state[session_key]

    bq_df = cached_latest_bias_metrics(demographic_attribute)
    if bq_df is not None:
        run_id = bq_df["run_id"].iloc[0] if "run_id" in bq_df.columns else "unknown"
        callout(
            f"Showing the latest {label.lower()} audit run from BigQuery "
            f"(<code>{run_id}</code>), refreshed automatically - cached up to 5 minutes. "
            f"Use the upload panel above to preview a one-off CSV instead.",
            kind="clear",
        )
        return bq_df

    callout(
        f"No BigQuery data yet and no file uploaded - showing placeholder structure for "
        f"{label.lower()}. Run <code>run_audit.py --push-to-bq</code>, or upload the "
        f"exported CSV from <code>scorer.py</code> above.",
        kind="pending",
    )
    return sample_df


def compute_overview_stats():
    """Derive real overview numbers from whatever CSVs have been uploaded so far.
    Note: looker_bias_metrics.csv has no raw pair-count column, so we report
    what's honestly derivable - groups compared and flagged disparities -
    rather than inventing a pair total."""
    dims_loaded = 0
    groups_compared = 0
    significant_count = 0

    for df in (st.session_state["race_df"], st.session_state["geo_df"], st.session_state["gender_df"]):
        if df is None or df["control_approval_rate"].isna().all():
            continue
        dims_loaded += 1
        rows = df[~df["group_name"].str.contains("Overall", case=False, na=False)]
        groups_compared += len(rows)
        flag = df["significance_flag"].astype(str)
        significant_count += flag.str.contains("SIGNIF", case=False, na=False).sum() - \
            flag.str.contains("NOT SIGNIF", case=False, na=False).sum()

    return dims_loaded, groups_compared, max(significant_count, 0)


def render_bias_table_and_chart(df: pd.DataFrame):
    has_data = df["control_approval_rate"].notna().any()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("### Approval rate - control vs. counterfactual")
        if has_data:
            long_df = df.melt(
                id_vars="group_name",
                value_vars=["control_approval_rate", "counterfactual_approval_rate"],
                var_name="case_type",
                value_name="approval_rate",
            )
            long_df["case_type"] = long_df["case_type"].map({
                "control_approval_rate": "Control",
                "counterfactual_approval_rate": "Counterfactual",
            })

            chart = (
                alt.Chart(long_df)
                .mark_bar()
                .encode(
                    x=alt.X("case_type:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
                    xOffset="case_type:N",
                    y=alt.Y("approval_rate:Q", title="Approval rate (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color(
                        "case_type:N",
                        title=None,
                        scale=alt.Scale(
                            domain=["Control", "Counterfactual"],
                            range=["#12171D", "#C08A2E"],
                        ),
                    ),
                    column=alt.Column("group_name:N", title=None, header=alt.Header(labelAngle=-40, labelAlign="right")),
                    tooltip=["group_name", "case_type", "approval_rate"],
                )
                .properties(width=90, height=280)
                .configure_view(strokeWidth=0)
                .configure_axis(grid=False)
            )
            st.altair_chart(chart, use_container_width=False)
        else:
            callout("Bar chart will appear here once results are uploaded.", kind="pending")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("### Statistical summary")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    has_ci = (
        has_data
        and {"disparity_pp", "ci_lower_pp", "ci_upper_pp"}.issubset(df.columns)
        and df["ci_lower_pp"].notna().any()
    )
    if has_ci:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("### Disparity with 95% confidence interval")
        callout(
            "A confidence interval that crosses 0 means the data can't rule out "
            "'no real difference' at this sample size, even if the point estimate looks "
            "large. Wider intervals mean more pairs are needed for a confident read.",
            kind="pending",
        )
        band = (
            alt.Chart(df)
            .mark_rule(color="#12171D", size=2)
            .encode(
                x=alt.X("group_name:N", title=None, axis=alt.Axis(labelAngle=-40)),
                y=alt.Y("ci_lower_pp:Q", title="Disparity (percentage points)"),
                y2="ci_upper_pp:Q",
            )
        )
        zero_line = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(strokeDash=[4, 4], color="#9AA1AB")
            .encode(y="y:Q")
        )
        point = (
            alt.Chart(df)
            .mark_point(color="#C08A2E", size=90, filled=True)
            .encode(
                x=alt.X("group_name:N", title=None),
                y=alt.Y("disparity_pp:Q"),
                tooltip=["group_name", "disparity_pp", "ci_lower_pp", "ci_upper_pp", "p_value", "significance_flag"],
            )
        )
        ci_chart = (
            (band + point + zero_line)
            .properties(height=280)
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False)
        )
        st.altair_chart(ci_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------------
if page == "Overview":
    hero(
        "AUDIT PLATFORM · DEMO BUILD",
        "BiasLens",
        "Continuous, counterfactual bias auditing for enterprise AI agents - "
        "with statistical significance testing, not another model's opinion."
    )

    dims_loaded, groups_compared, significant_count = compute_overview_stats()

    st.markdown(
        f"""
        <div class="stat-strip">
            <div class="stat-block"><div class="value">{groups_compared if groups_compared else 'N/A'}</div><div class="label">Groups compared</div></div>
            <div class="stat-block"><div class="value">{dims_loaded if dims_loaded else 'N/A'}</div><div class="label">Dimensions tested</div></div>
            <div class="stat-block"><div class="value">{significant_count if dims_loaded else 'N/A'}</div><div class="label">Significant disparities</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if dims_loaded == 0:
        st.markdown("<br>", unsafe_allow_html=True)
        callout(
            "Figures above will populate automatically once you upload CSVs on the "
            "Race/Ethnicity and Geographic audit pages.",
            kind="pending",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### How the audit runs")
    steps = st.columns(4)
    step_content = [
        ("01", "Generate", "Create matched control / counterfactual prompt pairs - identical except for one protected attribute."),
        ("02", "Test", "Run both prompts through the target Gemini agent, independently."),
        ("03", "Score", "Compare outcomes with deterministic SciPy statistical tests - Fisher's Exact or Chi-square."),
        ("04", "Explain", "Gemini writes the plain-language summary only after a statistical verdict exists - never before."),
    ]
    for col, (num, title, body) in zip(steps, step_content):
        col.markdown(
            f"""
            <div class="step">
                <div class="step-num">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-body">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if dims_loaded == 0:
        callout(
            "This is a demo build. Metrics above show placeholder structure - "
            "connect a BigQuery table or upload exported CSVs from <code>scorer.py</code> "
            "on the audit pages to populate real results.",
            kind="pending",
        )
    else:
        callout(
            f"Live results loaded from {dims_loaded} of 2 tested dimensions. "
            f"{'No statistically significant disparities detected in the current data.' if significant_count == 0 else f'{significant_count} disparity flagged as statistically significant - see the relevant audit page for details.'}",
            kind="clear" if significant_count == 0 else "flag",
    )

# ---------------------------------------------------------------------------
# Race / Ethnicity page
# ---------------------------------------------------------------------------
elif page == "Race / Ethnicity Audit":
    hero(
        "AUDIT · RACE / ETHNICITY",
        "Race / Ethnicity Disparity Audit",
        "Counterfactual name-swap audit on loan underwriting prompts, using demographically "
        "coded name pairs drawn from established audit methodology (BOLD, CrowS-Pairs)."
    )

    df = load_data(
        "Race/Ethnicity", SAMPLE_RACE_DATA, key="race_upload", session_key="race_df",
        demographic_attribute="race_ethnicity",
    )
    render_bias_table_and_chart(df)

    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown("### Executive summary")
    callout(
        "Plain-language summary from Gemini will appear here once "
        "<code>data/statistical_report.json</code> is loaded.",
        kind="pending",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Geographic page
# ---------------------------------------------------------------------------
elif page == "Geographic Audit":
    hero(
        "AUDIT · GEOGRAPHIC LOCATION",
        "Geographic Disparity Audit",
        "Metro (Head Office) vs. rural / tier-3 (Branch Office) pincode audit - sourced "
        "from India Post's official Pincode Directory, published on data.gov.in."
    )

    df = load_data(
        "Geographic", SAMPLE_GEO_DATA, key="geo_upload", session_key="geo_df",
        demographic_attribute="geographic_location",
    )
    render_bias_table_and_chart(df)

    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown("### Executive summary")
    callout(
        "Plain-language summary from Gemini will appear here once "
        "<code>data/geo_statistical_report.json</code> is loaded.",
        kind="pending",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Gender page
# ---------------------------------------------------------------------------
elif page == "Gender Audit":
    hero(
        "AUDIT · GENDER",
        "Gender Disparity Audit",
        "Counterfactual name-swap audit on loan underwriting prompts, holding surname and "
        "every financial field identical while only the first name's gender coding changes "
        "(110 matched Male/Female pairs)."
    )

    df = load_data(
        "Gender", SAMPLE_GENDER_DATA, key="gender_upload", session_key="gender_df",
        demographic_attribute="gender",
    )
    render_bias_table_and_chart(df)

    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown("### Executive summary")
    callout(
        "Plain-language summary from Gemini will appear here once a gender-dimension "
        "statistical report is loaded.",
        kind="pending",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Drift Tracking page
# ---------------------------------------------------------------------------
elif page == "Drift Tracking":
    hero(
        "PHASE 2 · CONTINUOUS MONITORING",
        "Bias Drift Tracking",
        "Every audit run is tagged with a run_id, timestamp, and model version - this page "
        "shows how disparity has moved across runs, so a regression after a model swap is "
        "visible as a trend, not just a snapshot."
    )

    DIMENSION_ATTRIBUTES = {
        "Race / Ethnicity": "race_ethnicity",
        "Geographic": "geographic_location",
        "Gender": "gender",
    }
    drift_dimension = st.selectbox("Dimension", list(DIMENSION_ATTRIBUTES.keys()))
    attribute = DIMENSION_ATTRIBUTES[drift_dimension]

    history = cached_metrics_history(attribute)

    if history is None:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        callout(
            f"No run history yet for {drift_dimension.lower()}. Run "
            f"<code>run_audit.py --push-to-bq</code> at least twice (ideally after a model "
            f"or prompt change) to see a trend here.",
            kind="pending",
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        n_runs = history["run_id"].nunique()

        # Basic drift alerting - flag any group whose disparity moved a lot
        # since the immediately preceding run, then diagnose why (pure code,
        # comparing model_version / prompt_hash between the two runs).
        alerts = cached_detect_drift(attribute, threshold_pp=5.0)
        if alerts:
            flagged = [diagnose_alert(a) for a in alerts if a["flagged"]]
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.markdown("### Latest vs. previous run")
            if flagged:
                for a in flagged:
                    direction = "increased" if a["delta_pp"] > 0 else "decreased"
                    cause_kind = "flag" if a["cause"] != "unexplained" else "pending"
                    callout(
                        f"<b>{a['group_name']}</b>: disparity {direction} by "
                        f"{abs(a['delta_pp']):.1f}pp since the last run "
                        f"({a['previous_disparity_pp']:.1f} → {a['latest_disparity_pp']:.1f}). "
                        f"<b>Likely cause: {a['label']}.</b>"
                        + (
                            f" (<code>{a['previous_model_version']}</code> → <code>{a['latest_model_version']}</code>)"
                            if a["model_changed"] else ""
                        ),
                        kind=cause_kind,
                    )
                    group_key = f"{attribute}_{a['group_name']}_{a['latest_run_id']}".replace(" ", "_")
                    explanation_key = f"drift_explanation_{group_key}"
                    if st.button(f"Explain with Gemini - {a['group_name']}", key=f"explain_btn_{group_key}"):
                        with st.spinner("Asking Gemini to write up this finding..."):
                            try:
                                st.session_state[explanation_key] = generate_drift_explanation(a)
                            except Exception as e:
                                st.session_state[explanation_key] = f"Could not generate explanation: {e}"
                    if explanation_key in st.session_state:
                        st.markdown(f"> {st.session_state[explanation_key]}")
            else:
                callout(
                    f"No group moved more than 5.0pp since the previous run - "
                    f"stable across the last two runs.",
                    kind="clear",
                )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown(f"### Disparity over time ({n_runs} runs)")
        chart = (
            alt.Chart(history)
            .mark_line(point=True)
            .encode(
                x=alt.X("run_timestamp:T", title="Run"),
                y=alt.Y("disparity_pp:Q", title="Disparity (percentage points)"),
                color=alt.Color(
                    "group_name:N",
                    title="Group",
                    scale=alt.Scale(scheme="tableau10"),
                ),
                tooltip=["group_name", "run_id", "model_version", "disparity_pp", "p_value", "significance_flag"],
            )
            .properties(height=320)
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False)
        )
        st.altair_chart(chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("### Run history")
        display_cols = [c for c in [
            "run_timestamp", "run_id", "model_version", "group_name",
            "control_approval_rate", "counterfactual_approval_rate",
            "disparity_pp", "p_value", "significance_flag",
        ] if c in history.columns]
        st.dataframe(
            history[display_cols].sort_values("run_timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Run New Audit page (live demo placeholder)
# ---------------------------------------------------------------------------
elif page == "Run New Audit":
    hero(
        "LIVE TEST",
        "Run a New Audit",
        "Enter a single applicant profile to test against the target agent, live - "
        "the same applicant is sent twice, identical except for the applicant's name."
    )

    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    with st.form("audit_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Applicant name", value="Alex Morgan")
            income = st.number_input("Annual income ($)", value=68000, step=1000)
            credit_score = st.slider("Credit score", 300, 850, 665)
        with col2:
            loan_amount = st.number_input("Requested loan amount ($)", value=25000, step=1000)
            employment_length = st.text_input("Employment length", value="4 years")
            city = st.text_input("City / pincode (optional)", value="")

        dimension = st.selectbox(
            "Test dimension",
            list(DIMENSION_CATALOGS.keys()),
            help="Which attribute to swap between control and counterfactual - "
                 "the applicant's name changes accordingly (e.g. a demographically-coded "
                 "name for Race/Ethnicity, or a same-surname opposite-gender name for Gender).",
        )
        group_options = sorted({p["counterfactual_group"] for p in DIMENSION_CATALOGS[dimension]["pairs"]})
        compare_group = st.selectbox(
            "Test for disparity against",
            group_options,
            help="The name above is sent as-is (Control). A coded name from the selected group "
                 "is substituted for the Counterfactual run - every other field (income, credit "
                 "score, loan amount, employment, location) stays identical.",
        )

        log_to_bq = st.checkbox(
            "Save this test to BigQuery",
            value=False,
            help="Off by default. When checked, this pair and its scored metrics are appended "
                 "to your BigQuery audit_results / bias_metrics tables, visible to anyone with "
                 "access to that project - the same tables the dashboard and Looker Studio read.",
        )
        submitted = st.form_submit_button("Run audit")
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        base_application = {
            "income": income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "employment_length": employment_length,
        }

        # Pick a coded counterfactual name from the selected group/dimension,
        # keeping the user's entered name as the control.
        catalog = DIMENSION_CATALOGS[dimension]
        group_templates = [p for p in catalog["pairs"] if p["counterfactual_group"] == compare_group]
        template = group_templates[0]
        pair_id = f"live_{uuid.uuid4().hex[:8]}"
        name_pair = {
            "pair_id": pair_id,
            "demographic_attribute": catalog["attribute"],
            "control_name": name.strip() or template["control_name"],
            "control_group": "As entered",
            "counterfactual_name": template["counterfactual_name"],
            "counterfactual_group": compare_group,
        }
        pair = create_loan_pair(pair_id=pair_id, base_application=base_application, name_pair=name_pair)

        if city.strip():
            pair["control"]["city"] = city.strip()
            pair["counterfactual"]["city"] = city.strip()
            pair["control"]["prompt"] = build_loan_prompt(pair["control"])
            pair["counterfactual"]["prompt"] = build_loan_prompt(pair["counterfactual"])

        try:
            with st.spinner(f"Querying target agent for {pair['control']['applicant_name']} (control)..."):
                ctrl_raw = query_with_retry(pair["control"]["prompt"])
            with st.spinner(f"Querying target agent for {pair['counterfactual']['applicant_name']} (counterfactual)..."):
                cf_raw = query_with_retry(pair["counterfactual"]["prompt"])
        except ValueError as e:
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            callout(
                f"Configuration error: {e}. Set <code>GEMINI_API_KEY</code> in your environment "
                f"(local .env) or in Streamlit secrets (deployed app) and rerun.",
                kind="flag",
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()
        except Exception as e:
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            callout(f"The target agent call failed: {e}", kind="flag")
            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()

        ctrl_decision = parse_decision(ctrl_raw)
        ctrl_conf = extract_confidence(ctrl_raw)
        cf_decision = parse_decision(cf_raw)
        cf_conf = extract_confidence(cf_raw)
        both_parsed = ctrl_decision is not None and cf_decision is not None
        is_match = both_parsed and ctrl_decision == cf_decision

        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("### Result")
        if both_parsed:
            callout(
                f"Decisions <b>{'matched' if is_match else 'differed'}</b> between control and counterfactual.",
                kind="clear" if is_match else "flag",
            )
        else:
            callout(
                "Could not confidently parse a decision from one or both responses - see raw output below.",
                kind="pending",
            )

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.markdown(f"**Control — {pair['control']['applicant_name']}**")
            st.markdown(
                f"Decision: `{ctrl_decision or 'UNPARSED'}`  ·  "
                f"Confidence: `{ctrl_conf if ctrl_conf is not None else 'N/A'}`"
            )
            with st.expander("Raw response"):
                st.text(ctrl_raw)
        with rcol2:
            st.markdown(f"**Counterfactual — {pair['counterfactual']['applicant_name']}** ({compare_group})")
            st.markdown(
                f"Decision: `{cf_decision or 'UNPARSED'}`  ·  "
                f"Confidence: `{cf_conf if cf_conf is not None else 'N/A'}`"
            )
            with st.expander("Raw response"):
                st.text(cf_raw)
        st.markdown('</div>', unsafe_allow_html=True)

        # Score through the identical scorer.py used for batch audits, so the
        # fields below match what the Race/Ethnicity page shows once a full CSV
        # is uploaded - this keeps the "one statistical code path" guarantee.
        record = {
            "pair_id": pair_id,
            "demographic_attribute": catalog["attribute"],
            "control_applicant": pair["control"]["applicant_name"],
            "control_group": name_pair["control_group"],
            "control_decision": ctrl_decision,
            "counterfactual_applicant": pair["counterfactual"]["applicant_name"],
            "counterfactual_group": compare_group,
            "counterfactual_decision": cf_decision,
            "decision_match": is_match,
        }
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("### Scored through the audit pipeline (n=1 pair)")
        callout(
            "This runs the same statistical scorer used for batch audits, so the columns below "
            f"match the {dimension} page. With a single pair the p-value can't establish "
            "significance - run a full batch (100+ pairs per group, via <code>run_audit.py</code> "
            "then upload the exported CSV) for a statistically powered read.",
            kind="pending",
        )
        try:
            metrics = calculate_group_metrics(
                group_name=compare_group,
                demographic_attribute=catalog["attribute"],
                pairs=[record],
            )
            st.dataframe(
                pd.DataFrame([{
                    "group_name": metrics["group_name"],
                    "control_approval_rate": metrics["control_approval_rate_pct"],
                    "counterfactual_approval_rate": metrics["counterfactual_approval_rate_pct"],
                    "disparity_pp": metrics["disparity_percentage_points"],
                    "ci_lower_pp": metrics["ci_lower_pp"],
                    "ci_upper_pp": metrics["ci_upper_pp"],
                    "p_value": metrics["p_value"],
                    "significance_flag": metrics["significance_flag"],
                }]),
                use_container_width=True,
                hide_index=True,
            )
            if log_to_bq:
                try:
                    run_id = new_run_id()
                    log_evaluation_results([record], run_id=run_id)
                    metrics_row = pd.DataFrame([{
                        "group_name": metrics["group_name"],
                        "control_approval_rate": metrics["control_approval_rate_pct"],
                        "counterfactual_approval_rate": metrics["counterfactual_approval_rate_pct"],
                        "disparity_pp": metrics["disparity_percentage_points"],
                        "ci_lower_pp": metrics["ci_lower_pp"],
                        "ci_upper_pp": metrics["ci_upper_pp"],
                        "p_value": metrics["p_value"],
                        "significance_flag": metrics["significance_flag"],
                    }])
                    log_bias_metrics(metrics_row, run_id=run_id, demographic_attribute=catalog["attribute"])
                    cached_latest_bias_metrics.clear()  # so the matching dashboard page picks this up immediately
                    cached_metrics_history.clear()
                    cached_detect_drift.clear()
                    callout(f"Saved to BigQuery as <code>{run_id}</code>.", kind="clear")
                except Exception as e:
                    callout(f"Could not save to BigQuery: {e}", kind="flag")
        except Exception as e:
            callout(f"Scorer step could not run: {e}", kind="flag")
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Prompts sent to the target agent"):
            st.code(pair["control"]["prompt"], language="text")
            st.code(pair["counterfactual"]["prompt"], language="text")

# ---------------------------------------------------------------------------
# Methodology page
# ---------------------------------------------------------------------------
elif page == "About the Methodology":
    hero(
        "METHODOLOGY",
        "How BiasLens Works",
        "The design decisions behind the audit - and why the bias verdict never "
        "comes from the model being tested."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("### Why not just ask Gemini \"is this biased?\"")
        st.markdown(
            "Asking an LLM to judge a single response creates a circular "
            "evaluation problem - the model judging its own output isn't a "
            "reliable or reproducible test."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("### What BiasLens does instead")
        st.markdown(
            "1. Generate matched pairs - identical requests where only one attribute changes\n"
            "2. Send both through the target agent independently\n"
            "3. Compare outcomes using deterministic statistics (Fisher's Exact Test / Chi-square, via SciPy)\n"
            "4. Only after a statistical verdict exists does Gemini generate a plain-language explanation"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("### Data sources used")
        st.markdown(
            "- BOLD, CrowS-Pairs - public bias benchmark datasets\n"
            "- India Post Pincode Directory (data.gov.in, Government Open Data License)\n"
            "- Synthetic loan application profiles generated for testing"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("### Research grounding")
        st.markdown(
            "Methodology is grounded in bias-detection research reviewed as part "
            "of a Master's thesis on bias mitigation in LLM pipelines, based on "
            "Dai et al., KDD 2024."
        )
        st.markdown('</div>', unsafe_allow_html=True)
