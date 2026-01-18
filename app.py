# Actuarial Pricing Dashboard
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pricing_engine import (
    PricingConfig, ModelConfig,
    extract_date_features, clean_data, prepare_features,
    train_models, calculate_premiums, sensitivity_analysis
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Actuarial Pricing Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'test_df' not in st.session_state:
    st.session_state.test_df = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<h1 class="main-header">🧮 Actuarial Pricing Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">End-to-End Insurance Claims Cost Prediction with Decision Intervention Points</p>', unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - CONFIGURATION
# =============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Settings
    st.subheader("🤖 Model Settings")
    
    use_xgboost = st.checkbox("XGBoost", value=True)
    use_lightgbm = st.checkbox("LightGBM", value=True)
    use_catboost = st.checkbox("CatBoost", value=True)
    
    n_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    n_estimators = st.slider("Number of Trees", 100, 1000, 500, step=100)
    learning_rate = st.slider("Learning Rate", 0.01, 0.2, 0.05, step=0.01)
    max_depth = st.slider("Max Depth", 3, 10, 6)
    
    st.markdown("---")
    
    # Pricing Settings
    st.subheader("💰 Pricing Parameters")
    
    expense_loading = st.slider("Expense Loading (%)", 10, 40, 25) / 100
    profit_margin = st.slider("Profit Margin (%)", 5, 20, 10) / 100
    contingency_margin = st.slider("Contingency Margin (%)", 2, 15, 5) / 100
    reinsurance_cost = st.slider("Reinsurance Cost (%)", 1, 10, 3) / 100
    commission_rate = st.slider("Commission Rate (%)", 5, 25, 15) / 100
    
    total_loading = expense_loading + profit_margin + contingency_margin + reinsurance_cost + commission_rate
    st.metric("Total Loading", f"{total_loading*100:.1f}%")

# =============================================================================
# MAIN CONTENT - TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Data Upload", 
    "🔧 Feature Engineering", 
    "🎯 Model Training",
    "📊 Results & Pricing",
    "📈 Sensitivity Analysis"
])

# =============================================================================
# TAB 1: DATA UPLOAD
# =============================================================================

with tab1:
    st.header("📁 Data Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Data")
        train_file = st.file_uploader("Upload train.csv", type=['csv'], key='train')
        
        if train_file is not None:
            st.session_state.train_df = pd.read_csv(train_file)
            st.success(f"✅ Loaded {len(st.session_state.train_df):,} training records")
    
    with col2:
        st.subheader("Test Data")
        test_file = st.file_uploader("Upload test.csv", type=['csv'], key='test')
        
        if test_file is not None:
            st.session_state.test_df = pd.read_csv(test_file)
            st.success(f"✅ Loaded {len(st.session_state.test_df):,} test records")
    
    if st.session_state.train_df is not None:
        st.markdown("---")
        st.subheader("📋 Data Preview")
        st.dataframe(st.session_state.train_df.head(10), use_container_width=True)
        
        # Data Quality Report
        st.subheader("📊 Data Quality Report")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{len(st.session_state.train_df):,}")
        with col2:
            st.metric("Total Columns", len(st.session_state.train_df.columns))
        with col3:
            missing_pct = (st.session_state.train_df.isnull().sum().sum() / 
                          (len(st.session_state.train_df) * len(st.session_state.train_df.columns)) * 100)
            st.metric("Missing Values", f"{missing_pct:.2f}%")
        
        # Show columns with missing values
        missing = st.session_state.train_df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            st.write("**Columns with Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Missing %': (missing.values / len(st.session_state.train_df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)

# =============================================================================
# TAB 2: FEATURE ENGINEERING
# =============================================================================

with tab2:
    st.header("🔧 Feature Engineering")
    
    if st.session_state.train_df is None:
        st.warning("⚠️ Please upload data first in the Data Upload tab.")
    else:
        st.info("📌 **Improvement:** Extracting date-based features instead of ignoring them.")
        
        if st.button("🚀 Extract Date Features", type="primary"):
            with st.spinner("Extracting features..."):
                # Apply feature engineering
                st.session_state.train_df = extract_date_features(st.session_state.train_df)
                if st.session_state.test_df is not None:
                    st.session_state.test_df = extract_date_features(st.session_state.test_df)
                
                st.success("✅ Date features extracted!")
        
        # Show new features
        new_features = ['AgeAtAccident', 'ReportingLagDays', 'AccidentMonth', 
                       'AccidentQuarter', 'AccidentDayOfWeek']
        existing_new = [f for f in new_features if f in st.session_state.train_df.columns]
        
        if existing_new:
            st.subheader("📊 Extracted Features")
            st.dataframe(
                st.session_state.train_df[existing_new].describe().round(2),
                use_container_width=True
            )
            
            # Visualize Age distribution
            if 'AgeAtAccident' in st.session_state.train_df.columns:
                fig = px.histogram(
                    st.session_state.train_df, 
                    x='AgeAtAccident',
                    nbins=40,
                    title="Age Distribution at Accident",
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 3: MODEL TRAINING
# =============================================================================

with tab3:
    st.header("🎯 Model Training")
    
    if st.session_state.train_df is None or st.session_state.test_df is None:
        st.warning("⚠️ Please upload both train and test data first.")
    else:
        # Auto-detect target column
        target_candidates = ['UltimateIncurredClaimCost', 'Premium Amount', 'target']
        target_col = None
        for tc in target_candidates:
            if tc in st.session_state.train_df.columns:
                target_col = tc
                break
        
        if target_col is None:
            numeric_cols = st.session_state.train_df.select_dtypes(include=[np.number]).columns.tolist()
            target_col = st.selectbox("Select Target Column", numeric_cols)
        else:
            st.info(f"🎯 Target Column: **{target_col}**")
        
        # Target Distribution
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                st.session_state.train_df, 
                x=target_col,
                nbins=50,
                title="Original Distribution",
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                x=np.log1p(st.session_state.train_df[target_col]),
                nbins=50,
                title="Log-Transformed Distribution",
                color_discrete_sequence=['#764ba2']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        if st.button("🚀 Train Models", type="primary"):
            # Clean data
            train_clean = clean_data(st.session_state.train_df)
            test_clean = clean_data(st.session_state.test_df)
            
            # Prepare features
            X, X_test, y, feature_cols = prepare_features(
                train_clean, test_clean, target_col
            )
            st.session_state.feature_cols = feature_cols
            
            st.write(f"📊 Prepared {len(feature_cols)} features")
            
            # Create model config from sidebar
            model_config = ModelConfig(
                use_xgboost=use_xgboost,
                use_lightgbm=use_lightgbm,
                use_catboost=use_catboost,
                n_folds=n_folds,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate
            )
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train models
            active_models = sum([use_xgboost, use_lightgbm, use_catboost])
            total_steps = active_models * n_folds
            step_counter = [0]  # Use mutable list to avoid nonlocal issue
            
            def update_progress(model_name, fold, total_folds, mae):
                step_counter[0] += 1
                progress_bar.progress(step_counter[0] / total_steps)
                status_text.text(f"Training {model_name} - Fold {fold}/{total_folds} - MAE: {mae:.4f}")
            
            results = train_models(X, y, X_test, model_config, update_progress)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Training Complete!")
            
            st.session_state.model_results = results
            
            # Ensemble predictions
            preds = [results[m]['pred'] for m in results]
            st.session_state.predictions = np.mean(preds, axis=0)
            
            # Display results
            st.subheader("📊 Model Performance")
            metrics_df = pd.DataFrame([
                {
                    'Model': model,
                    'OOF MAE': results[model]['oof_mae'],
                    'Fold Std': np.std(results[model]['fold_scores'])
                }
                for model in results
            ])
            st.dataframe(metrics_df.style.format({'OOF MAE': '{:.4f}', 'Fold Std': '{:.4f}'}),
                        use_container_width=True)

# =============================================================================
# TAB 4: RESULTS & PRICING
# =============================================================================

with tab4:
    st.header("📊 Results & Pricing")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ Please train models first.")
    else:
        # Create pricing config from sidebar
        pricing_config = PricingConfig(
            expense_loading=expense_loading,
            profit_margin=profit_margin,
            contingency_margin=contingency_margin,
            reinsurance_cost=reinsurance_cost,
            commission_rate=commission_rate
        )
        
        # Calculate premiums
        premium_result = calculate_premiums(st.session_state.predictions, pricing_config)
        
        # Premium Metrics
        st.subheader("💰 Premium Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Premium", f"${premium_result['mean_premium']:,.0f}")
        with col2:
            st.metric("Median Premium", f"${premium_result['median_premium']:,.0f}")
        with col3:
            st.metric("Min Premium", f"${premium_result['min_premium']:,.0f}")
        with col4:
            st.metric("Max Premium", f"${premium_result['max_premium']:,.0f}")
        
        st.markdown("---")
        
        # Feature Importance
        st.subheader("📊 Feature Importance (Top 15)")
        if st.session_state.model_results and st.session_state.feature_cols:
            # Get last trained model's importance
            last_model = list(st.session_state.model_results.keys())[-1]
            importance = pd.DataFrame({
                'Feature': st.session_state.feature_cols,
                'Importance': st.session_state.model_results[last_model]['feature_importance']
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(
                importance.sort_values('Importance'), 
                x='Importance', 
                y='Feature',
                orientation='h',
                title=f"Feature Importance ({last_model})",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Premium Distribution
        st.subheader("📈 Premium Distribution")
        fig = px.histogram(
            x=premium_result['final_premium'],
            nbins=50,
            title="Final Premium Distribution",
            labels={'x': 'Premium ($)'},
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Download Submission
        st.subheader("📥 Download Submission")
        
        # Create submission DataFrame
        id_col = 'ClaimNumber' if 'ClaimNumber' in st.session_state.test_df.columns else 'id'
        submission = pd.DataFrame({
            id_col: st.session_state.test_df[id_col],
            'UltimateIncurredClaimCost': premium_result['expected_loss']
        })
        
        csv_buffer = BytesIO()
        submission.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        st.download_button(
            label="📥 Download submission.csv",
            data=csv_buffer,
            file_name="submission.csv",
            mime="text/csv",
            type="primary"
        )
        
        st.dataframe(submission.head(10), use_container_width=True)

# =============================================================================
# TAB 5: SENSITIVITY ANALYSIS
# =============================================================================

with tab5:
    st.header("📈 Sensitivity Analysis")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ Please train models first.")
    else:
        st.info("📌 See how premium changes when varying pricing parameters.")
        
        # Parameter to vary
        param_to_analyze = st.selectbox(
            "Select Parameter to Analyze",
            ['expense_loading', 'profit_margin', 'contingency_margin', 
             'reinsurance_cost', 'commission_rate']
        )
        
        # Range of values
        col1, col2 = st.columns(2)
        with col1:
            min_val = st.number_input("Min Value (%)", value=5, min_value=0, max_value=50)
        with col2:
            max_val = st.number_input("Max Value (%)", value=30, min_value=0, max_value=50)
        
        if st.button("🔍 Run Sensitivity Analysis", type="primary"):
            # Create base config
            base_config = PricingConfig(
                expense_loading=expense_loading,
                profit_margin=profit_margin,
                contingency_margin=contingency_margin,
                reinsurance_cost=reinsurance_cost,
                commission_rate=commission_rate
            )
            
            # Run analysis
            values = np.linspace(min_val/100, max_val/100, 10)
            results_df = sensitivity_analysis(
                st.session_state.predictions, 
                base_config,
                param_to_analyze,
                values.tolist()
            )
            
            # Plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=results_df['value'] * 100,
                y=results_df['mean_premium'],
                mode='lines+markers',
                name='Mean Premium',
                line=dict(color='#667eea', width=3),
                marker=dict(size=10)
            ))
            
            # Mark current value
            current_val = getattr(base_config, param_to_analyze) * 100
            current_premium = calculate_premiums(st.session_state.predictions, base_config)['mean_premium']
            
            fig.add_trace(go.Scatter(
                x=[current_val],
                y=[current_premium],
                mode='markers',
                name='Current Setting',
                marker=dict(color='red', size=15, symbol='star')
            ))
            
            fig.update_layout(
                title=f"Sensitivity Analysis: {param_to_analyze.replace('_', ' ').title()}",
                xaxis_title=f"{param_to_analyze.replace('_', ' ').title()} (%)",
                yaxis_title="Mean Premium ($)",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.dataframe(
                results_df.style.format({
                    'value': '{:.1%}',
                    'mean_premium': '${:,.0f}',
                    'total_loading': '{:.1f}%'
                }),
                use_container_width=True
            )

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🧮 Actuarial Pricing Engine | Built with Streamlit</p>
    <p>Egyptian UHI Context - Law 2/2018 Compliant</p>
</div>
""", unsafe_allow_html=True)
