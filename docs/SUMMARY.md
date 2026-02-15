# 🏛️ Actuarial Valuation Documentation (Law 2/2018)

Welcome to the documentation for the **UHI Actuarial Valuation Model**. This repository contains a comprehensive suite of technical, legal, and strategic documents designed to support the implementation of the Universal Health Insurance system in Egypt.

---

## 🧭 Documentation Navigation

````carousel
### 🛡️ UHI Compliance (Article 3 & 40)
[UHI_COMPLIANCE.md](UHI_COMPLIANCE.md) | [النسخة المبسطة بالعربية](الامتثال_لقانون_التأمين.md)
Formal declaration of how the model meets the funding and solvency requirements of Law 2/2018.
<!-- slide -->
### 🏗️ Valuation Architecture
[CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | [النسخة المبسطة بالعربية](هيكلة_النظام_ودورة_العمل.md)
Technical breakdown of the `ActuarialValuationEngine` and the multi-year projection loop.
<!-- slide -->
### ⚙️ Actuarial Assumptions
[ASSUMPTIONS_UPDATE_GUIDE.md](ASSUMPTIONS_UPDATE_GUIDE.md) | [النسخة المبسطة بالعربية](دليل_تحديث_الافتراضات.md)
Guide for actuaries to update inflation, investment returns, and contribution defaults.
<!-- slide -->
### 🗺️ Dashboard Mapping
[STREAMLIT_MAPPING.md](STREAMLIT_MAPPING.md) | [النسخة المبسطة بالعربية](خريطة_لوحة_التحكم.md)
Mapping visual charts (Solvency, Reserves) to the underlying logic.
<!-- slide -->
### 📊 Strategic Analysis
[COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md) | [التحليل التنافسي](التحليل_التنافسي_للسوق.md)
How this UHI-specific solution outperforms general enterprise systems like FIS RiskSuite.
<!-- slide -->
### 💎 Buy vs. Build (Value)
[BUY_VS_BUILD_ANALYSIS.md](BUY_VS_BUILD_ANALYSIS.md) | [لماذا نحن؟ المبرمج الداخلي](لماذا_نحن؟_الاستثمار_مقابل_التطوير_الداخلي.md)
Strategic argument for purchasing this specialized valuation suite over internal development.
````

## 🚀 Key Improvements (Refactor v2.0)
- **Social Solidarity Model**: Shifted from individual pricing to aggregate system solvency.
- **Law 2/2018 Logic**: Automated contribution calculations (1%, 3%, 5% rules).
- **Long-term Foresight**: 20-year Solvency and Reserve Fund projections.
- **Decision Support**: Real-time "Required State Subsidy" trigger.

---
> [!TIP]
> **Getting Started**: Start by reviewing the [UHI_COMPLIANCE.md](UHI_COMPLIANCE.md) document to understand the legal backbone of the mathematical model.
