# UHI Valuation Assumptions Update Guide

This guide explains how to update the core actuarial assumptions in the codebase to adjust long-term solvency projections for the Universal Health Insurance system.

## 1. Dynamic Updates (Dashboard Interface)
The most efficient way to test scenarios is via the **Sidebar Configuration** (<span style="color:#FFA500">■</span>).

- **Medical Inflation**: Rate at which healthcare service costs rise annually (Economic risk).
- **Wage Inflation**: Rate at which salary-based contributions grow (Revenue growth).
- **Investment Return**: Profit generated from the accumulated reserve fund.
- **Non-capable %**: The portion of the population supported by the state treasury.

## 2. Permanent Default Updates (Code)
To change the "base case" scenario for the authority, modify the `UHISystemConfig` data class.

### Location: `pricing_engine.py`
```python
@dataclass
class UHISystemConfig:
    """Social Health Insurance Actuarial Assumptions (Law 2/2018)"""
    wage_inflation: float = 0.07        # Default 7%
    medical_inflation: float = 0.12     # Default 12%
    investment_return_rate: float = 0.10 # Default 10%
    admin_expense_pct: float = 0.04     # Capped at 5%
```

## 3. Pre-Valuation Audit Checklist
Before presenting a 10-year outlook to the Board, the Actuary must verify:

- [ ] **Data Quality**: Is the `population_structure.csv` representative of the current governorates in the phase?
- [ ] **Gap Analysis**: In what year does the "Medical Expenditure" line cross the "Total Revenue" line?
- [ ] **Treasury Exposure**: Is the `Required_State_Subsidy` within the national budget limits?
- [ ] **Inflation Delta**: Is the gap between medical and wage inflation realistic given current CPI?

---

## Technical Update Checklist
| Assumption Type | Location in Code |
| :--- | :--- |
| **Legal Contribution Rates** | `UHISystemConfig` constants in `pricing_engine.py`. |
| **Admin Expense Caps** | `admin_expense_pct` logic in `project_solvency`. |
| **New Revenue Sources** | Add to `cigarette_tax_lump` or `highway_tolls_lump` in `pricing_engine.py`. |

> [!WARNING]
> Drastic changes in inflation assumptions can lead to massive reserve depletion. Always verify changes using the **Scenario Analysis** mindset.
