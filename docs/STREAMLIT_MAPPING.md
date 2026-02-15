# Dashboard Component Mapping (UHI Valuation)

This document maps the visual components of the **UHI Actuarial Dashboard** to the underlying code blocks in `app.py` and `pricing_engine.py`.

## 📍 Navigation & Tabs

| Dashboard Tab | UI Section | Code Logic |
| :--- | :--- | :--- |
| **📊 Solvency Projection** | Multi-year comparison chart. | `ActuarialValuationEngine().project_solvency()` |
| **💸 Revenue vs Cost** | Annual profit/loss bar chart. | `df_proj['Total_Revenue'] - df_proj['Total_Expenditure']` |
| **🗄️ Reserve Accumulation** | Cumulative fund growth (Area). | `accumulated_reserve += net_cash_flow + investment_income` |

## ⚙️ Sidebar Configuration

| Logic Variable | Sidebar Slider | Impact |
| :--- | :--- | :--- |
| `medical_inflation` | **Medical Inflation (%)** | Exponential increase in annual medical expenditure. |
| `wage_inflation` | **Wage Inflation (%)** | Growth of contribution revenues from employees. |
| `investment_return_rate` | **Investment Return (%)** | Compounded growth of the Reserve Fund. |
| `projection_years` | **Projection Horizon** | Number of years (rows) in the simulation. |

## 🏛️ Strategic KPI Cards

- **Total Population**: Count of records in the uploaded/generated CSV.
- **Reserve Fund (Final Year)**: The end-state technical reserves (millions).
- **Solvency Status**: Binary flag (Solvent/Deficit) based on reserve polarity.
- **Required State Subsidy**: Triggered if reserves < 0 (Article 48 requirement).

---

## Data Schema (`population_structure.csv`)

The model expects the following input features for valuation:
1.  **EmploymentStatus**: Determines contribution logic (Employee vs. Non-capable).
2.  **MonthlyWage**: Basis for the percentage-based contributions.
3.  **SpouseInSystem / ChildrenCount**: Triggers additional head-of-family loadings.
4.  **EstimatedAnnualCost**: The "today" baseline for per-person medical expenditure.

> [!NOTE]
> The dashboard is designed for **High-Level Stakeholders** (e.g., Ministers or UHI Board Members) to visualize long-term financial stability.
