# ⚖️ BiasLens — Real-Time Bias Auditing for Enterprise AI Agents

> **Continuous, counterfactual, statistically rigorous bias auditing for AI-powered decision systems.**

BiasLens is an AI fairness auditing platform designed to test whether an AI agent produces different outcomes when the underlying input is identical but a demographic or contextual attribute changes.

Instead of asking an LLM whether its own response is "biased", BiasLens uses **controlled counterfactual testing + deterministic statistical analysis** to identify measurable disparities.

The platform is designed to evolve from a research prototype into a continuous AI governance and monitoring layer for enterprise AI systems.

---

## 🚨 The Problem

AI agents are increasingly being used in decisions that directly affect people:

* 💳 Loan and credit assessment
* 👔 Recruitment and candidate screening
* 🏥 Insurance and healthcare workflows
* 🎧 Customer support prioritization
* 🏦 Financial services
* 🎓 Admissions and eligibility decisions

An AI system may produce apparently reasonable responses while still treating otherwise identical individuals differently because of attributes such as:

* Name
* Gender
* Geographic origin
* Race or ethnicity
* Socioeconomic context
* Other sensitive or contextual attributes

Traditional bias testing is often:

* Manual
* Performed only once
* Difficult to reproduce
* Dependent on subjective human judgment
* Not continuously monitored after deployment

### BiasLens asks a different question:

> **If everything stays the same except one controlled attribute, does the AI agent's decision change?**

If it does, BiasLens measures whether that difference is statistically meaningful.

---

# 💡 Our Approach

BiasLens uses **counterfactual pairs**.

For example, consider a synthetic loan application:

```text
Applicant:
Income: $68,000
Credit Score: 665
Loan Amount: $25,000
Employment: 4 years

Name: Alex Morgan
```

A counterfactual version keeps all decision-relevant information identical while changing only the controlled attribute:

```text
Applicant:
Income: $68,000
Credit Score: 665
Loan Amount: $25,000
Employment: 4 years

Name: [Counterfactual Name]
```

Both applications are independently evaluated by the target AI agent.

BiasLens then compares the outcomes.

The objective is **not to assume that every difference is discrimination**.

Instead, the system asks:

> **Is the observed difference large enough and statistically significant enough to warrant investigation?**

---

# 🧠 Why BiasLens Is Different

A naive approach would be:

```text
AI response
     ↓
Ask Gemini:
"Is this biased?"
     ↓
Bias detected / Not detected
```

BiasLens deliberately avoids this.

The target model should **not be the final judge of its own behavior**.

Instead:

```text
                Base Prompt
                     │
                     ↓
          Counterfactual Generator
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
      Control                Counterfactual
          │                     │
          └──────────┬──────────┘
                     ↓
                 Target AI
                     │
                     ↓
               Outcomes
                     │
                     ↓
          Deterministic Statistics
                     │
                     ↓
                Bias Verdict
                     │
                     ↓
            Gemini Explanation
```

This separation makes the evaluation more reproducible and auditable.

---

# 🔬 Audit Pipeline

BiasLens follows a four-stage audit process:

### 01 — Generate

Create matched control/counterfactual prompt pairs.

Only the selected attribute changes.

### 02 — Test

Send both versions independently through the target AI agent.

### 03 — Score

Compare outcomes using deterministic statistical tests.

The current implementation supports statistical analysis using **Fisher's Exact Test / Chi-square through SciPy**, depending on the audit design.

### 04 — Explain

Only after the statistical result exists does Gemini generate a human-readable explanation.

This means:

> **Gemini explains the evidence. It does not create the statistical verdict.**

The four-step pipeline is also reflected directly in the application interface.

---

# 📊 Current Audit Dimensions

## Race / Ethnicity Audit

BiasLens currently supports counterfactual name-based testing for race/ethnicity-related scenarios.

The dashboard compares:

* Control approval rate
* Counterfactual approval rate
* Disparity
* p-value
* Statistical significance

The application is structured to load results from exported scorer CSV files and visualize them as grouped approval-rate charts and statistical tables.

---

## 🌍 Geographic Audit

BiasLens also extends the same methodology to geographic attributes.

The current prototype explores:

**Metro / Head Office vs Rural / Tier-3 / Branch Office**

The geographic audit uses India's official Pincode Directory as the geographic reference source.

This allows the same counterfactual methodology to be applied beyond demographic attributes.

The architecture is intentionally attribute-agnostic:

```text
Attribute = Race/Ethnicity
             ↓
        Bias Audit

Attribute = Geography
             ↓
        Bias Audit

Attribute = Gender
             ↓
        Bias Audit

Attribute = Other
             ↓
        Bias Audit
```

The application currently includes a dedicated geographic audit interface.

---

# 📐 Statistical Methodology

BiasLens does not label an individual response as "biased" simply because two outputs differ.

Instead, it evaluates the aggregate behavior of matched counterfactual tests.

Example:

```text
Control approval rate:          72%
Counterfactual approval rate:   58%

Observed disparity:             14 percentage points
```

The system then evaluates whether the observed difference is statistically significant.

A simplified interpretation:

```text
p-value > 0.05
        ↓
No statistically significant disparity detected

p-value ≤ 0.05
        ↓
Statistically significant disparity detected
        ↓
Requires further investigation
```

### Important

A statistically significant result does **not automatically prove discrimination**.

It indicates that the observed disparity is unlikely to be explained by random variation under the chosen statistical assumptions and should therefore be investigated.

Similarly:

> **No statistically significant disparity does not prove that an AI system is perfectly fair.**

BiasLens is an auditing and monitoring tool, not a legal or ethical certification system.

---

# 🧪 Data Sources

BiasLens is designed to work without confidential enterprise data.

### Public Bias Benchmarks

The project can incorporate publicly available benchmark datasets such as:

* BOLD
* CrowS-Pairs
* StereoSet

These provide established examples for evaluating social bias in language models.

### Synthetic Enterprise Data

Synthetic loan application profiles are used to simulate realistic enterprise decision-making scenarios without exposing real customer information.

Example fields include:

```text
Applicant Name
Annual Income
Credit Score
Loan Amount
Employment Length
Location
```

### Geographic Reference Data

The geographic audit uses the India Post Pincode Directory published through India's Government Open Data platform.

---

# 🏗️ Architecture

The current prototype is structured around the following pipeline:

```text
                     ┌─────────────────────┐
                     │  Synthetic / Public  │
                     │       Dataset        │
                     └──────────┬──────────┘
                                │
                                ↓
                    ┌──────────────────────┐
                    │ Counterfactual Pair  │
                    │      Generator       │
                    └──────────┬───────────┘
                               │
                     Control + Counterfactual
                               │
                               ↓
                    ┌──────────────────────┐
                    │     Target Agent     │
                    │       Gemini         │
                    └──────────┬───────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │   Response Collector │
                    └──────────┬───────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │ Statistical Scorer   │
                    │       SciPy          │
                    └──────────┬───────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │     Bias Metrics     │
                    └──────────┬───────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │   Gemini Analysis    │
                    │  Explanation Layer   │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    ↓                     ↓
              Streamlit UI          Looker Studio
```

---

# ☁️ Google Cloud Architecture

The intended production architecture extends the current prototype into a cloud-native auditing service:

```text
                    Enterprise AI Agent
                            │
                            │ Audit Request
                            ↓
                     Cloud Run API
                            │
                            ↓
                     ADK Orchestrator
                            │
             ┌──────────────┼──────────────┐
             ↓              ↓              ↓
       Pair Generator   Test Executor   Scoring Agent
             │              │              │
             └──────────────┼──────────────┘
                            ↓
                         Gemini
                            │
                            ↓
                       BigQuery
                            │
                  ┌─────────┴─────────┐
                  ↓                   ↓
             Audit History       Bias Metrics
                  │                   │
                  └─────────┬─────────┘
                            ↓
                       Looker Studio
                            │
                            ↓
                    Compliance / ML
                         Teams
```

### Planned Google Cloud Components

| Component         | Purpose                                                    |
| ----------------- | ---------------------------------------------------------- |
| **Gemini API**    | Target AI agent, counterfactual generation and explanation |
| **BigQuery**      | Audit results, metrics and longitudinal data               |
| **Cloud Run**     | Serverless deployment of the audit service                 |
| **Pub/Sub**       | Asynchronous audit job processing at scale                 |
| **ADK**           | Multi-agent orchestration                                  |
| **Looker Studio** | Analytics and audit reporting                              |
| **Firebase**      | Potential application state / user-facing functionality    |

---

# 🖥️ Application

The current Streamlit application contains:

* Overview dashboard
* Race / Ethnicity Audit
* Geographic Audit
* Run New Audit interface
* Methodology explanation
* CSV upload capability
* Statistical results tables
* Approval-rate visualizations

The application currently accepts scorer output CSVs containing fields such as:

```text
group_name
control_approval_rate
counterfactual_approval_rate
disparity_pp
p_value
significance_flag
```

Once uploaded, the dashboard automatically populates the relevant charts and tables.

---

# 🚀 Getting Started

## Prerequisites

* Python 3.9+
* pip
* Streamlit

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd biaslens
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The current dependency file contains the Streamlit, Pandas and Altair components used by the dashboard.

Run the application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 📂 Project Structure

A representative project structure is:

```text
biaslens/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── statistical_report.json
│   ├── geo_statistical_report.json
│   ├── looker_bias_metrics.csv
│   └── geo_looker_bias_metrics.csv
│
├── pair_generator.py
├── target_agent.py
├── scorer.py
│
└── ...
```

The Streamlit dashboard is designed to consume exported scorer outputs and display the resulting audit metrics.

---

# 🧭 Current Project Status

> **Prototype / Demo Build — actively being extended**

The current Streamlit interface is functional as a visualization and audit-results layer, while some production integrations are still being completed.

### ✅ Implemented / Demonstrated

* Counterfactual audit concept
* Race / ethnicity audit interface
* Geographic audit interface
* Statistical-results visualization
* Fisher's Exact Test / Chi-square methodology
* CSV-based result loading
* Synthetic loan application workflow
* Gemini explanation layer architecture
* Streamlit dashboard
* Methodology documentation

### 🔄 In Progress

* Extended audit dataset processing
* Complete BigQuery integration
* Live audit execution from the Streamlit interface
* Automated result ingestion
* Executive-summary generation
* Longitudinal bias tracking
* Expanded demographic dimensions
* Cloud deployment
* Continuous audit scheduling

The current repository explicitly identifies the live audit form and BigQuery/data connections as areas still being wired into the prototype.

---

# 🔮 Roadmap

### Phase 1 — Research Prototype

* [x] Counterfactual audit methodology
* [x] Synthetic loan decision scenario
* [x] Statistical scoring
* [x] Streamlit visualization
* [x] Race / ethnicity audit
* [x] Geographic audit

### Phase 2 — Data & Cloud

* [ ] BigQuery audit warehouse
* [ ] Cloud Run API
* [ ] Automated audit jobs
* [ ] Pub/Sub pipeline
* [ ] Longitudinal audit history

### Phase 3 — Agentic Auditing

* [ ] ADK-based multi-agent orchestration
* [ ] Automated test generation
* [ ] Audit planning agent
* [ ] Statistical analysis agent
* [ ] Explanation agent
* [ ] Root-cause investigation agent

### Phase 4 — Enterprise Monitoring

* [ ] Model-version comparison
* [ ] Bias drift detection
* [ ] Prompt-version comparison
* [ ] Retrieval-source comparison
* [ ] Automated alerts
* [ ] Compliance reporting
* [ ] Audit evidence export

---

# 📈 Future Vision: Bias Drift Monitoring

BiasLens is ultimately designed to move beyond one-time audits.

For example:

```text
Model v1
Bias Score = 0.08
       │
       ↓
Model v2
Bias Score = 0.11
       │
       ↓
Model v3
Bias Score = 0.19  ⚠️
       │
       ↓
Model v4
Bias Score = 0.27  🚨
```

BiasLens could identify:

> **Bias increased significantly after Model v4 was deployed.**

The system could then investigate:

```text
Model changed?
       ↓
Prompt changed?
       ↓
Retrieval corpus changed?
       ↓
Decision policy changed?
       ↓
Audit dataset changed?
       ↓
Statistically significant shift?
```

This transforms BiasLens from a static fairness checker into an **AI observability and governance layer**.

---

# 🎯 Why This Matters

The goal is not to declare an AI model "fair" or "unfair" based on one test.

The goal is to create an engineering discipline around AI fairness:

```text
       BEFORE DEPLOYMENT
              ↓
          Audit Model
              ↓
           Deploy
              ↓
       CONTINUOUS TESTING
              ↓
       Detect Bias Drift
              ↓
       Investigate Change
              ↓
        Generate Evidence
              ↓
       Improve / Re-test
```

BiasLens aims to make fairness testing closer to:

> **Unit testing + monitoring + observability**

for AI decision systems.

---

# ⚠️ Important Limitations

BiasLens is a research and engineering prototype.

A statistical disparity does not by itself establish unlawful discrimination, causation, or intentional bias.

Similarly, the absence of a statistically significant disparity does not prove that an AI system is unbiased.

Results depend on:

* The selected benchmark
* Counterfactual construction
* Sample size
* Statistical assumptions
* Target model behavior
* Prompt design
* Protected attribute definitions
* Evaluation metrics

BiasLens should therefore be used as an **AI auditing and investigation aid**, not as a standalone legal, regulatory, hiring, lending, or compliance decision-maker.

---

# 🎓 Research Foundation

BiasLens was developed from research into bias detection and mitigation in LLM pipelines as part of a Master's research project.

The platform translates research concepts around counterfactual and comparative bias evaluation into an engineering workflow that can be repeatedly executed, measured and visualized.

> **Research → Evaluation Methodology → Engineering Pipeline → Continuous Monitoring**

---

# 🏆 Competition Demo

The prototype demonstrates a complete conceptual workflow:

```text
Generate
   ↓
Test
   ↓
Score
   ↓
Explain
   ↓
Visualize
```

The extended audit pipeline is being processed with additional datasets and scenarios, with the final results intended to be incorporated into the completed submission.

---

# 👩‍💻 Author

**Visshnu Prethi Manjere Kumar**

M.Sc. Computer Science & Data Science
Research focus: **Bias Detection and Mitigation in LLM Pipelines**

---

## ⭐ BiasLens

> **Before an AI agent makes decisions about people, test how it behaves when the only thing that changes is the person.**

**Continuous AI fairness auditing — measured, reproducible, and evidence-driven.**
