"""
BiasLens - AI Fairness Audit Platform
Streamlit front-end. Data connections are placeholders for now -
wire up BigQuery / CSV loading where marked TODO.
"""

import streamlit as st
import pandas as pd
import altair as alt

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
    ["Overview", "Race / Ethnicity Audit", "Geographic Audit", "Run New Audit", "About the Methodology"],
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

# ---------------------------------------------------------------------------
# Session state - lets the Overview page see data uploaded on other pages
# ---------------------------------------------------------------------------
st.session_state.setdefault("race_df", None)
st.session_state.setdefault("geo_df", None)


# ---------------------------------------------------------------------------
# Helper: file uploader that falls back to sample/placeholder data
# ---------------------------------------------------------------------------
def load_data(label: str, sample_df: pd.DataFrame, key: str, session_key: str) -> pd.DataFrame:
    has_existing = st.session_state.get(session_key) is not None
    expander_label = f"Upload new {label.lower()} data" if has_existing else f"Upload {label.lower()} data"

    with st.expander(expander_label, expanded=not has_existing):
        uploaded = st.file_uploader(f"Upload {label} CSV", type="csv", key=key, label_visibility="collapsed")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.session_state[session_key] = df
            return df

    # Uploader widget looks empty after navigating away and back - but if we
    # already captured a real upload earlier in this session, use that
    # instead of falling back to the placeholder.
    if has_existing:
        callout(
            f"Showing previously uploaded {label.lower()} data from this session. "
            f"Use the upload panel above to replace it.",
            kind="clear",
        )
        return st.session_state[session_key]

    callout(
        f"No file uploaded yet - showing placeholder structure for {label.lower()}. "
        f"Upload the exported CSV from <code>scorer.py</code> above to populate this view.",
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

    for df in (st.session_state["race_df"], st.session_state["geo_df"]):
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

    df = load_data("Race/Ethnicity", SAMPLE_RACE_DATA, key="race_upload", session_key="race_df")
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

    df = load_data("Geographic", SAMPLE_GEO_DATA, key="geo_upload", session_key="geo_df")
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
# Run New Audit page (live demo placeholder)
# ---------------------------------------------------------------------------
elif page == "Run New Audit":
    hero(
        "LIVE TEST",
        "Run a New Audit",
        "Enter a single applicant profile to test against the target agent."
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

        submitted = st.form_submit_button("Run audit")
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        callout(
            "This is a UI placeholder - connect this form to "
            "<code>pair_generator.py</code> → <code>target_agent.py</code> → "
            "<code>scorer.py</code> to run a live counterfactual test.",
            kind="pending",
        )
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("### Submitted profile")
        st.code(
            f"Applicant:         {name}\n"
            f"Income:            ${income:,}\n"
            f"Credit score:      {credit_score}\n"
            f"Loan amount:       ${loan_amount:,}\n"
            f"Employment length: {employment_length}\n"
            f"Location:          {city or 'Not specified'}",
            language="text",
        )
        st.markdown('</div>', unsafe_allow_html=True)

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
