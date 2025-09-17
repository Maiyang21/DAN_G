"""
Main Streamlit application for the Refinery Forecast App
"""
import io
import json
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add the app directory to the path
sys.path.append(os.path.dirname(__file__))

from core.aws_utils import upload_df_csv, invoke_endpoint, check_aws_credentials
from etl.etl_pipeline import process_uploaded_files, unify_csvs, build_static_features, split_columns, prepare_model_inputs
from config import S3_PREFIX, TIME_INDEX_COL, TARGET_COLS, APP_TITLE, APP_ICON, ALLOWED_FILE_TYPES, MAX_FILE_SIZE_MB

# Page configuration
st.set_page_config(
    page_title=APP_TITLE, 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title(f"{APP_ICON} {APP_TITLE}")

# Initialize session state
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "targets_df" not in st.session_state:
    st.session_state.targets_df = None
if "statics_df" not in st.session_state:
    st.session_state.statics_df = None
if "futures_df" not in st.session_state:
    st.session_state.futures_df = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "etl_status" not in st.session_state:
    st.session_state.etl_status = "Not Started"

# Sidebar
with st.sidebar:
    st.header("📁 1) Upload Input Files")
    st.markdown("Upload Excel files containing refinery data:")
    st.markdown("- **TS Monitoring Tags** sheet")
    st.markdown("- **Blends** sheet") 
    st.markdown("- **Lab** sheet")
    
    files = st.file_uploader(
        "Choose files",
        type=ALLOWED_FILE_TYPES,
        accept_multiple_files=True,
        help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
    )
    
    st.markdown("---")
    st.header("⚙️ 2) ETL & Processing")
    
    # ETL Status
    st.markdown(f"**ETL Status:** {st.session_state.etl_status}")
    
    do_etl = st.button("🔄 Run ETL Pipeline", type="primary")
    
    st.markdown("---")
    st.header("🔮 3) Forecasting")
    
    # Check AWS credentials
    aws_available = check_aws_credentials()
    if aws_available:
        st.success("✅ AWS credentials configured")
    else:
        st.error("❌ AWS credentials not configured")
    
    do_predict = st.button("🎯 Request 7-day Forecast", disabled=not aws_available)
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    This app processes refinery data through an ETL pipeline and generates 7-day forecasts using Autoformer + EBM models deployed on AWS SageMaker.
    
    **Features:**
    - Excel file processing
    - Data transformation and feature engineering
    - S3 integration for data storage
    - SageMaker endpoint integration
    - Interactive visualizations
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Data Processing & Visualization")
    
    # ETL Phase
    if do_etl:
        if not files:
            st.error("Please upload at least one file.")
        else:
            st.session_state.etl_status = "Running..."
            
            with st.spinner("Processing files through ETL pipeline..."):
                # Process uploaded files
                targets_df, statics_df, futures_df = process_uploaded_files(files)
                
                if targets_df is not None and statics_df is not None and futures_df is not None:
                    st.session_state.targets_df = targets_df
                    st.session_state.statics_df = statics_df
                    st.session_state.futures_df = futures_df
                    st.session_state.etl_status = "Completed Successfully"
                    
                    st.success("✅ ETL pipeline completed successfully!")
                    
                    # Display data summaries
                    st.subheader("📈 Processed Data Summary")
                    
                    col_t1, col_t2, col_t3 = st.columns(3)
                    with col_t1:
                        st.metric("Targets", f"{targets_df.shape[0]} rows × {targets_df.shape[1]} cols")
                    with col_t2:
                        st.metric("Static Features", f"{statics_df.shape[0]} rows × {statics_df.shape[1]} cols")
                    with col_t3:
                        st.metric("Future Covariates", f"{futures_df.shape[0]} rows × {futures_df.shape[1]} cols")
                    
                    # Show sample data
                    st.subheader("🎯 Target Variables (Sample)")
                    st.dataframe(targets_df.head(10), use_container_width=True)
                    
                    st.subheader("📋 Static Features")
                    st.dataframe(statics_df, use_container_width=True)
                    
                else:
                    st.session_state.etl_status = "Failed"
                    st.error("❌ ETL pipeline failed. Please check your input files.")
    
    # Display current data if available
    if st.session_state.targets_df is not None:
        st.subheader("📊 Data Visualization")
        
        # Time series plot
        if TIME_INDEX_COL in st.session_state.targets_df.columns:
            # Select target columns for plotting
            target_cols_available = [c for c in TARGET_COLS if c in st.session_state.targets_df.columns]
            if not target_cols_available:
                target_cols_available = [c for c in st.session_state.targets_df.columns if c != TIME_INDEX_COL][:5]
            
            selected_targets = st.multiselect(
                "Select targets to visualize:",
                target_cols_available,
                default=target_cols_available[:3] if len(target_cols_available) >= 3 else target_cols_available
            )
            
            if selected_targets:
                fig = go.Figure()
                
                for target in selected_targets:
                    fig.add_trace(go.Scatter(
                        x=st.session_state.targets_df[TIME_INDEX_COL],
                        y=st.session_state.targets_df[target],
                        mode='lines',
                        name=target,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title="Time Series Data",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("🔧 Configuration")
    
    # Show current configuration
    st.subheader("AWS Settings")
    st.code(f"""
Region: {os.getenv('AWS_REGION', 'us-east-1')}
S3 Bucket: {os.getenv('S3_BUCKET', 'Not set')}
S3 Access Point: {os.getenv('S3_ACCESS_POINT_ARN', 'Not set')}
SageMaker Endpoint: {os.getenv('SM_ENDPOINT', 'autoformer-ebm-endpoint')}
    """)
    
    # Data status
    st.subheader("Data Status")
    if st.session_state.targets_df is not None:
        st.success("✅ Data loaded")
        st.metric("Records", len(st.session_state.targets_df))
    else:
        st.info("No data loaded")
    
    # Prediction status
    st.subheader("Prediction Status")
    if st.session_state.prediction is not None:
        st.success("✅ Prediction available")
    else:
        st.info("No prediction available")

# Prediction Phase
if do_predict:
    if st.session_state.targets_df is None or st.session_state.statics_df is None:
        st.error("Please run ETL pipeline first.")
    else:
        with st.spinner("Invoking SageMaker endpoint..."):
            try:
                # Prepare model inputs
                payload = prepare_model_inputs(
                    st.session_state.targets_df,
                    st.session_state.statics_df,
                    st.session_state.futures_df
                )
                
                # Upload data to S3
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                data_key = f"{S3_PREFIX}batch_{timestamp}/timeseries.csv"
                static_key = f"{S3_PREFIX}batch_{timestamp}/static.csv"
                
                data_uri = upload_df_csv(st.session_state.targets_df, data_key)
                static_uri = upload_df_csv(st.session_state.statics_df, static_key)
                
                st.success("📤 Data uploaded to S3:")
                st.code(f"Timeseries: {data_uri}")
                st.code(f"Static: {static_uri}")
                
                # Invoke SageMaker endpoint
                resp = invoke_endpoint(payload)
                st.session_state.prediction = resp
                
                st.success("🎯 Prediction received!")
                
                # Display prediction results
                st.subheader("📈 Forecast Results")
                
                if isinstance(resp, dict) and "forecast" in resp:
                    fdf = pd.DataFrame(resp["forecast"])
                    
                    if "date" in fdf.columns:
                        fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce")
                    
                    # Plot forecasts
                    for tgt in fdf["target"].unique():
                        sub = fdf[fdf["target"] == tgt]
                        y_col = "p50" if "p50" in sub.columns else ("value" if "value" in sub.columns else None)
                        
                        if y_col is not None:
                            fig = px.line(sub, x="date", y=y_col, title=f"7-day Forecast — {tgt}")
                            st.plotly_chart(fig, use_container_width=True)
                
                # Show EBM explanations
                if isinstance(resp, dict) and "explanations" in resp:
                    st.subheader("🔍 Feature Explanations (EBM)")
                    try:
                        imp = pd.DataFrame([
                            {"feature": k, "importance": v} for k, v in resp["explanations"].items()
                        ]).sort_values("importance", ascending=False)
                        
                        st.dataframe(imp, use_container_width=True)
                        
                        fig = px.bar(imp.head(20), x="feature", y="importance", 
                                   title="Top Static Drivers (EBM)")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.write("Raw explanations:")
                        st.json(resp["explanations"])
                
                # Show raw response if no standard format
                if not (isinstance(resp, dict) and ("forecast" in resp or "explanations" in resp)):
                    st.subheader("📄 Raw Response")
                    st.json(resp if len(json.dumps(resp)) < 5000 else {"keys": list(resp.keys())})
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Refinery Forecast App — Powered by Autoformer + EBM on AWS</p>
</div>
""", unsafe_allow_html=True)

