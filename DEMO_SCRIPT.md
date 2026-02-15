# Presales Demo Script: Actuarial Pricing Engine
**Target Audience:** C-Level Executives & Actuarial Directors
**Duration:** 10 Minutes
**Goal:** Demonstrate the "MoveDiscount" philosophy through a live, technical engine.

---

## 1. Introduction (1 Minute)
*   **Context:** "We've discussed the theory of Dynamic Pricing. Now, I want to show you the *engine* that powers it."
*   **Action:** Open the Streamlit App (`streamlit run app.py`).
*   **Narrative:** "This is a cloud-native, Python-based pricing engine capable of handling millions of records dynamically, unlike static Excel models."

## 2. Data Ingestion & "The Data Desert" (2 Minutes)
*   **Tab:** `📁 Data Upload`
*   **Action:** Upload `train.csv` and `test.csv` (use sample data).
*   **Feature Highlight:** Point to the **Data Quality Report**.
*   **Key Message:** "Notice how we instantly audit the data quality. In the Egyptian market, data carries 'noise'. Our engine automatically flags missing values and outliers before they corrupt the risk model."

## 3. Dynamic Feature Engineering (2 Minutes)
*   **Tab:** `🔧 Feature Engineering`
*   **Action:** Click `🚀 Extract Date Features`.
*   **Visual:** Show the 'AgeAtAccident' histogram.
*   **Key Message:** "Traditional systems ignore unstructured dates. We extract 'Reporting Lag' and 'Seasonality' automatically. This is how we find hidden risk patterns that others miss."

## 4. The "Black Box" Revealed (Model Training) (2 Minutes)
*   **Tab:** `🎯 Model Training`
*   **Action:** Select `UltimateIncurredClaimCost` as target. Click `🚀 Train Models`.
*   **Narrative:** "We are training an ensemble of XGBoost, LightGBM, and CatBoost live. This competes with the world's best Kaggle Grandmasters."
*   **Visual:** Show the standard deviation dropping across folds. "This low variance proves our pricing is stable and defensible to regulators."

## 5. The "CFO Dashboard" (Sensitivity Analysis) (3 Minutes) - **WOW MOMENT**
*   **Tab:** `📈 Sensitivity Analysis`
*   **Action:** Select `profit_margin`. Set range `5%` to `25%`. Click `Run`.
*   **Visual:** The Plotly line chart showing Premium vs. Profit.
*   **Key Message:** "Mr. CFO, you can now dial in your desired profit margin and instantly see the impact on market premiums. We move from 'guessing' to 'engineering' your profitability."

## 6. Closing
*   **Action:** Download `submission.csv`.
*   **Closing Line:** "In 10 minutes, we went from raw data to a regulatory-compliant pricing structure. This is the speed of 'MoveDiscount'."
