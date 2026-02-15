# The Actuarial Orchestrator — Requirements & Specifications

> **Platform:** Konecta Actuarial Orchestrator
> **Module:** [Actuarial Pricing Dashboard](https://github.com/eltaweelactuary/actuarial-pricing-dashboard)
> **Author:** Ahmed Eltaweel | Presales Engineer
> **Date:** February 2026

---

## 1. Platform Overview

An AI-powered platform for enhanced End-to-End (E2E) Actuarial Management, designed to boost:

- **Operational Efficiency** — streamlined workflows with automated orchestration
- **Real-Time Visibility** — live task execution tracking and seamless cross-system coordination
- **Collection Rate & Time** — accelerated debt recovery and reduced cycle times
- **Acquisition Quality** — improved risk segmentation and reduced exposure

```mermaid
graph TB
    subgraph ORCHESTRATOR["The Actuarial Orchestrator"]
        direction TB
        OE["Operational Efficiency"]
        RV["Real-time Visibility"]
        CR["Collection Rate & Time"]
        AQ["Acquisition Quality & Risk Reduction"]
    end

    subgraph CAPABILITIES["Key Capabilities"]
        WO["Workflow Orchestrator\nE2E Management"]
        AR["Actuarial Rules Config\nParameters & Strategies"]
        CG["Centralized Governance\n& Monitoring"]
        AI["AI Engine\nInsights, Recommendations,\nAutomation, CIM"]
    end

    ORCHESTRATOR --> CAPABILITIES

    subgraph KPI["Performance"]
        K1["&gt;85%\nDebt Recovery Rate"]
        K2["+94K\nTransactions / Day"]
    end

    CAPABILITIES --> KPI

    style ORCHESTRATOR fill:#0f172a,stroke:#f59e0b,color:#f1f5f9
    style CAPABILITIES fill:#0f172a,stroke:#3b82f6,color:#f1f5f9
    style KPI fill:#0f172a,stroke:#10b981,color:#f1f5f9
```

---

## 2. Governance & Actuarial Independence

> [!IMPORTANT]
> **The Actuarial Orchestrator does not override or modify any approved actuarial procedures.** The platform operates as a transparent, decision-support layer — not a decision-making layer.

### Our Approach

| Principle | Description |
|-----------|-------------|
| **Non-Interference** | All established actuarial methodologies, approved models, and regulatory procedures remain fully intact. The platform does not alter, bypass, or replace any approved departmental workflows. |
| **Assumption Extraction** | The engine extracts actuarial assumptions (loss distributions, mortality tables, expense ratios) directly from historical data and existing departmental records. No external assumptions are imposed. |
| **Expert-Driven Integration** | All extracted assumptions are validated and calibrated by the actuarial team before integration. The department's experience and professional judgment remain the ultimate authority. |
| **Full Transparency** | Every model output includes feature importance rankings, confidence intervals, and audit trails — ensuring complete visibility into how premiums are derived. |
| **Configurable Parameters** | All pricing loadings (expense, profit, contingency, reinsurance, commission) are manually adjustable by authorized actuaries. The engine applies them — it does not set them. |

```mermaid
flowchart LR
    subgraph DEPT["Actuarial Department (Authority)"]
        A1["Approved Procedures"]
        A2["Actuarial Assumptions"]
        A3["Professional Judgment"]
    end

    subgraph ENGINE["Pricing Engine (Support Tool)"]
        B1["Extract Assumptions\nfrom Historical Data"]
        B2["Train ML Models"]
        B3["Generate Premiums"]
    end

    subgraph OUTPUT["Validated Output"]
        C1["Actuary Reviews\n& Approves"]
        C2["Final Pricing\nDecision"]
    end

    A1 -->|"Guidelines"| ENGINE
    A2 -->|"Parameters"| B1
    B1 --> B2 --> B3
    B3 -->|"Draft Output"| C1
    A3 -->|"Override Authority"| C1
    C1 --> C2

    style DEPT fill:#0f172a,stroke:#f59e0b,color:#f1f5f9
    style ENGINE fill:#1e293b,stroke:#3b82f6,color:#f1f5f9
    style OUTPUT fill:#0f172a,stroke:#10b981,color:#f1f5f9
```

---

## 3. Pricing Engine Position in the Ecosystem

```mermaid
graph LR
    subgraph BEFORE["Pre-Actuarial Departments"]
        CL["Claims Management"]
        UW["Underwriting"]
    end

    subgraph ACTUARIAL["Actuarial Department"]
        direction TB
        DA["Debtor Analysis"]
        PE["Pricing Engine"]
        SS["Strategy & Simulation"]
    end

    subgraph AFTER["Post-Actuarial Departments"]
        COL["Collection"]
        RET["Retention"]
        FRA["FRA Reporting"]
    end

    CL -->|"Claims Data"| ACTUARIAL
    UW -->|"Policy Data"| ACTUARIAL
    ACTUARIAL -->|"Pricing Output"| COL
    ACTUARIAL -->|"Risk Scores"| RET
    ACTUARIAL -->|"Submission CSV"| FRA

    style BEFORE fill:#1e293b,stroke:#ef4444,color:#f1f5f9
    style ACTUARIAL fill:#0f172a,stroke:#f59e0b,color:#f1f5f9
    style AFTER fill:#1e293b,stroke:#10b981,color:#f1f5f9
    style PE fill:#f59e0b,stroke:#f59e0b,color:#000
```

---

## 4. Customer Journey Flow

```mermaid
flowchart LR
    subgraph S1["1. Activation"]
        A1["Contract"]
        A1A["CVP + Scoring\n+ Clustering"]
    end

    subgraph S2["2. Prediction & Prevention"]
        A2["Education Campaign"]
        A2A["Direct Debit Push"]
    end

    subgraph S3["3. Onboarding"]
        A3["First Invoice"]
        A3A["Reminder & Support"]
    end

    subgraph S4["4. Assistance"]
        A4["Criticalities & Complaints"]
        A4A["Auto-matching"]
    end

    subgraph S5["5. Active Collection"]
        A5["Phone Collection"]
        A5A["Dedicated Team"]
    end

    subgraph S6["6. Retention"]
        A6["Aimed Outbound"]
        A6A["Plans & Suspension"]
    end

    subgraph S7["7. Closure"]
        A7["Automated Registration"]
    end

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7

    style S1 fill:#1e40af,stroke:#3b82f6,color:#fff
    style S2 fill:#6d28d9,stroke:#8b5cf6,color:#fff
    style S3 fill:#047857,stroke:#10b981,color:#fff
    style S4 fill:#b45309,stroke:#f59e0b,color:#fff
    style S5 fill:#9f1239,stroke:#f43f5e,color:#fff
    style S6 fill:#0e7490,stroke:#06b6d4,color:#fff
    style S7 fill:#374151,stroke:#6b7280,color:#fff
```

---

## 5. Internal Process Management Flow

```mermaid
flowchart TD
    subgraph P1["1. Debtor Analysis & Clustering"]
        P1A["Behavioral Analysis"]
        P1B["Scoring & Clustering"]
        P1A --> P1B
    end

    subgraph P2["2. Rules & Parameters Setting"]
        P2A["Automatic Rule Adjustment"]
        P2B["Engine Language Translation"]
        P2A --> P2B
    end

    subgraph P3["3. Strategy Definition & Simulation"]
        P3A["Strategy per Cluster"]
        P3B["Recovery Path Simulation"]
        P3A --> P3B
    end

    subgraph P4["4. Prediction"]
        P4A["Action Volume Calculation"]
        P4B["Liquidity Planning"]
        P4A --> P4B
    end

    subgraph P5["5. Contact Management"]
        P5A["Inbound & Outbound"]
        P5B["AI Dynamic Contact Model"]
        P5A --> P5B
    end

    subgraph P6["6. Collection"]
        P6A["Auto-matching & Registration"]
        P6B["Plans Update & Suspension"]
        P6A --> P6B
    end

    P1 --> P2 --> P3 --> P4 --> P5 --> P6

    GOV["Centralized Governance & Continuous Improvement"]

    P1 -.-> GOV
    P3 -.-> GOV
    P6 -.-> GOV

    style P1 fill:#0f172a,stroke:#3b82f6,color:#f1f5f9
    style P2 fill:#0f172a,stroke:#8b5cf6,color:#f1f5f9
    style P3 fill:#0f172a,stroke:#f59e0b,color:#f1f5f9
    style P4 fill:#0f172a,stroke:#06b6d4,color:#f1f5f9
    style P5 fill:#0f172a,stroke:#10b981,color:#f1f5f9
    style P6 fill:#0f172a,stroke:#f43f5e,color:#f1f5f9
    style GOV fill:#1e293b,stroke:#f59e0b,color:#f59e0b
```

---

## 6. Document Flow Across Departments

```mermaid
flowchart LR
    subgraph CLAIMS["Claims Department"]
        D1["Claims Register"]
        D2["Loss Reports"]
        D3["Settlement Records"]
    end

    subgraph ACTUARIAL["Actuarial Department"]
        D4["Train CSV\n(Historical Data)"]
        D5["Test CSV\n(New Policies)"]
        D6["Pricing Engine"]
        D7["Model Report"]
        D8["Sensitivity Chart"]
        D9["Submission CSV"]
    end

    subgraph COLLECTION["Collection Department"]
        D10["Premium Schedule"]
        D11["Recovery Plan"]
    end

    subgraph FINANCE["Finance Department"]
        D12["P&L Report"]
        D13["Reserve Report"]
    end

    subgraph REGULATORY["Regulatory Bodies (FRA / NOSI)"]
        D14["Regulatory Filing"]
        D15["UHI Review"]
    end

    D1 -->|"Claims Data"| D4
    D2 -->|"Loss History"| D5
    D3 -->|"Settlement Info"| D6
    D4 --> D6
    D5 --> D6
    D6 --> D7
    D6 --> D8
    D6 --> D9
    D9 -->|"Premiums"| D10
    D7 -->|"Risk Scores"| D11
    D9 -->|"Filing"| D14
    D8 -->|"Review"| D15
    D7 -->|"Reserves"| D13
    D9 -->|"Revenue Forecast"| D12

    style CLAIMS fill:#1e293b,stroke:#ef4444,color:#f1f5f9
    style ACTUARIAL fill:#0f172a,stroke:#f59e0b,color:#f1f5f9
    style COLLECTION fill:#1e293b,stroke:#06b6d4,color:#f1f5f9
    style FINANCE fill:#1e293b,stroke:#10b981,color:#f1f5f9
    style REGULATORY fill:#1e293b,stroke:#8b5cf6,color:#f1f5f9
    style D6 fill:#f59e0b,stroke:#f59e0b,color:#000
```

### Document Registry

| # | Document | Format | Source | Destination | Purpose |
|---|----------|--------|--------|-------------|---------|
| 1 | Claims Register | Internal DB | Claims Dept. | Actuarial Dept. | Historical claims record |
| 2 | Loss Reports | PDF / CSV | Claims Dept. | Actuarial Dept. | Detailed loss reports |
| 3 | Train CSV | `.csv` | Actuarial Dept. | Pricing Engine | Model training data |
| 4 | Test CSV | `.csv` | Actuarial Dept. | Pricing Engine | New policies for pricing |
| 5 | Model Performance Report | On-screen | Pricing Engine | Actuary | RMSE, MAE, R² per model |
| 6 | Sensitivity Chart | On-screen | Pricing Engine | CFO / Board | Parameter impact analysis |
| 7 | Submission CSV | `.csv` | Pricing Engine | FRA / Collection | Final premium per policy |
| 8 | Premium Schedule | CSV / PDF | Actuarial Dept. | Collection Dept. | Due premium schedule |
| 9 | Reserve Report | PDF | Actuarial Dept. | Finance Dept. | Reserves report |
| 10 | Regulatory Filing | CSV | Actuarial Dept. | FRA / NOSI | Mandatory regulatory reports |

---

## 7. User Experience Flow

```mermaid
flowchart TD
    START(("User Opens\nDashboard")) --> T1

    subgraph T1["Tab 1: Data Upload"]
        U1["Upload Train CSV"] --> U2["Upload Test CSV"]
        U2 --> U3{"Files Valid?"}
        U3 -->|"Yes"| U4["Data Preview\n& Quality Report"]
        U3 -->|"No"| U5["Error Message\n+ Format Guide"]
        U5 --> U1
    end

    U4 --> T2

    subgraph T2["Tab 2: Feature Engineering"]
        F1["Auto-detect Date Columns"]
        F2["Extract: AgeAtAccident\nReportingLagDays\nAccidentMonth / Quarter"]
        F3["Outlier Capping\nAge: 0-100, Lag: 0-3650"]
        F1 --> F2 --> F3
    end

    F3 --> T3

    subgraph T3["Tab 3: Model Training"]
        M1["Configure: N Folds\nLearning Rate\nMax Depth"]
        M2["Start Training"]
        M3["Progress: XGBoost\nLightGBM, CatBoost"]
        M4["Per-Fold Metrics\n+ Feature Importance"]
        M1 --> M2 --> M3 --> M4
    end

    M4 --> T4

    subgraph T4["Tab 4: Results & Pricing"]
        R1["Adjust Sliders:\nExpense 15%\nProfit 10%\nContingency 5%\nReinsurance 3%\nCommission 7%"]
        R2["Real-time Premium\nRecalculation"]
        R3["Premium Distribution\n+ Summary Statistics"]
        R1 --> R2 --> R3
    end

    R3 --> T5

    subgraph T5["Tab 5: Sensitivity Analysis"]
        S1["Select Parameter\nto Analyze"]
        S2["Auto-generate\nValue Range"]
        S3["Impact Chart:\nParameter vs Mean Premium"]
        S1 --> S2 --> S3
    end

    T5 --> EXPORT

    subgraph EXPORT["Export"]
        E1["Download Submission CSV"]
        E2["Send to Collection Dept."]
        E3["Send to FRA"]
        E1 --> E2
        E1 --> E3
    end

    style START fill:#f59e0b,stroke:#d97706,color:#000
    style T1 fill:#0f172a,stroke:#3b82f6,color:#f1f5f9
    style T2 fill:#0f172a,stroke:#8b5cf6,color:#f1f5f9
    style T3 fill:#0f172a,stroke:#047857,color:#f1f5f9
    style T4 fill:#0f172a,stroke:#f59e0b,color:#f1f5f9
    style T5 fill:#0f172a,stroke:#06b6d4,color:#f1f5f9
    style EXPORT fill:#0f172a,stroke:#10b981,color:#f1f5f9
```

---

## 8. Stakeholder Interaction Map

```mermaid
graph TB
    subgraph DEPT_CLAIMS["Claims Department"]
        CM["Claims Manager\n- Prepares claims data\n- Sends CSV to Actuarial"]
    end

    subgraph DEPT_ACT["Actuarial Department"]
        ACT["Actuary\n- Uploads data\n- Trains models\n- Configures margins"]
        CACT["Chief Actuary\n- Reviews sensitivity\n- Approves pricing"]
    end

    subgraph DEPT_COL["Collection Department"]
        COL["Collection Officer\n- Receives premium schedule\n- Executes recovery plan"]
    end

    subgraph DEPT_FIN["Finance / Senior Management"]
        CFO["CFO / Board\n- Reviews sensitivity analysis\n- Approves profit margins"]
    end

    subgraph DEPT_REG["Regulatory Bodies"]
        REG["FRA / NOSI\n- Receives reports\n- Reviews compliance"]
    end

    subgraph ENGINE["Pricing Engine Dashboard"]
        DASH["Streamlit Dashboard"]
    end

    CM -->|"Claims CSV"| DASH
    ACT -->|"Upload + Train + Price"| DASH
    CACT -->|"Review + Approve"| DASH
    DASH -->|"Premium Schedule"| COL
    DASH -->|"Sensitivity Report"| CFO
    DASH -->|"Submission CSV"| REG

    style DEPT_CLAIMS fill:#1e293b,stroke:#ef4444,color:#f1f5f9
    style DEPT_ACT fill:#0f172a,stroke:#f59e0b,color:#f1f5f9
    style DEPT_COL fill:#1e293b,stroke:#06b6d4,color:#f1f5f9
    style DEPT_FIN fill:#1e293b,stroke:#10b981,color:#f1f5f9
    style DEPT_REG fill:#1e293b,stroke:#8b5cf6,color:#f1f5f9
    style ENGINE fill:#f59e0b,stroke:#d97706,color:#000
```

---

## 9. Pricing Logic Flow

```mermaid
graph TD
    INPUT["Claims Data\n(Train + Test CSV)"] --> FE["Feature Engineering\nDate extraction, outlier capping"]
    FE --> TRAIN["3-Model Ensemble Training"]

    subgraph TRAIN["Model Training"]
        XG["XGBoost"]
        LG["LightGBM"]
        CB["CatBoost"]
    end

    TRAIN --> PRED["Expected Loss\nexp(prediction) - 1"]
    PRED --> LOAD["Apply Loadings"]

    subgraph LOAD["Pricing Loadings"]
        L1["Expense: 15%"]
        L2["Profit: 10%"]
        L3["Contingency: 5%"]
        L4["Reinsurance: 3%"]
        L5["Commission: 7%"]
    end

    LOAD --> TOTAL["Total Loading = 40%"]
    TOTAL --> FORMULA["Premium = Loss x 1.40"]
    FORMULA --> CLIP["Clip to Range\n(100 - 1,000,000)"]
    CLIP --> OUT["Final Premium per Policy"]

    style INPUT fill:#1e40af,stroke:#3b82f6,color:#fff
    style PRED fill:#6d28d9,stroke:#8b5cf6,color:#fff
    style TOTAL fill:#b45309,stroke:#f59e0b,color:#fff
    style FORMULA fill:#047857,stroke:#10b981,color:#fff
    style OUT fill:#9f1239,stroke:#f43f5e,color:#fff
```

---

## 10. Technical Requirements

### 9.1 Software Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.8+ | Runtime |
| Streamlit | Latest | Web Dashboard UI |
| XGBoost | Latest | Gradient Boosting |
| LightGBM | Latest | Light Gradient Boosting |
| CatBoost | Latest | Categorical Boosting |
| Pandas / NumPy | Latest | Data Processing |
| Scikit-learn | Latest | Cross-validation & Metrics |
| Matplotlib / Plotly | Latest | Visualization |

### 9.2 Data Requirements

| Requirement | Specification |
|------------|---------------|
| File Format | CSV (UTF-8) |
| Minimum Rows | 1,000+ claims |
| Target Column | `UltimateIncurredClaimCost` |
| Date Columns | `DateOfBirth`, `DateOfAccident`, `DateReported` |
| Missing Values | Native handling (-999 sentinel) |

### 9.3 Infrastructure

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Storage | 500 MB | 2 GB |
| GPU | Not required | Not required |

### 9.4 Team Requirements

| Role | Count | Responsibility |
|------|-------|---------------|
| Data Scientist | 1 | Model tuning & data preparation |
| Actuary | 1 | Pricing calibration & validation |
| Claims Analyst | 1 | Data extraction from claims system |
| IT Support | 1 | Deployment & maintenance |

---

## 11. Implementation Timeline

```mermaid
gantt
    title Implementation Timeline (8-14 Weeks)
    dateFormat YYYY-MM-DD
    axisFormat %b %d

    section Phase 1 - Data Preparation
        Collect Claims Data from NOSI/Insurer  :a1, 2026-03-01, 7d
        Clean and Format CSV Files             :a2, after a1, 5d
        Define Target and Feature Mapping      :a3, after a2, 3d
        Environment Setup                      :a4, after a2, 2d

    section Phase 2 - Model Calibration
        Train Models on Egyptian Data          :b1, after a4, 7d
        Tune Hyperparameters                   :b2, after b1, 7d
        Calibrate Pricing Margins              :b3, after b2, 7d
        Validate vs Known Loss Ratios          :b4, after b3, 5d

    section Phase 3 - Deployment
        Deploy on Streamlit Cloud              :c1, after b4, 5d
        Train End-Users                        :c2, after c1, 5d
        Integrate into FRA Workflow            :c3, after c2, 7d
        Handover and Documentation             :c4, after c3, 5d
```

---

## 12. Risk Register

```mermaid
quadrantChart
    title Risk Assessment Matrix
    x-axis "Low Impact" --> "High Impact"
    y-axis "Low Likelihood" --> "High Likelihood"

    "Data Availability": [0.8, 0.7]
    "Regulatory Resistance": [0.6, 0.5]
    "Team Training": [0.4, 0.6]
    "Infra Failure": [0.3, 0.2]
    "Model Drift": [0.7, 0.4]
    "Data Privacy": [0.5, 0.3]
```

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Data Availability | High | High | Partner with NOSI for digitized claims extract |
| Regulatory Resistance | Medium | Medium | FRA Sandbox + transparent ML outputs |
| Team Training | Medium | Medium | 2-day workshop + video documentation |
| Infrastructure | Low | Low | Free Streamlit Cloud deployment |
| Model Drift | High | Medium | Quarterly retrain + monitoring alerts |
| Data Privacy | Medium | Low | On-premise processing, no PII in exports |

---

## 13. Regulatory Compliance Checklist

- [x] Supports Egyptian UHI **Law 2/2018** mandatory actuarial review
- [x] Produces downloadable CSV for FRA submission
- [x] Transparent ML model — feature importance visible
- [x] Configurable pricing margins — auditable loadings
- [x] Handles Egyptian date formats and local data standards
- [ ] Data encryption at rest (requires deployment configuration)
- [ ] Role-based access control (future enhancement)
- [ ] Audit trail logging (future enhancement)
