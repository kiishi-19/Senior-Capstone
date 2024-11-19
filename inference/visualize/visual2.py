import streamlit as st
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def load_scores(scores_directory):
    """
    Load all *_scores.npy files from the specified directory.
    """
    score_files = sorted(glob.glob(os.path.join(scores_directory, '*_scores.npy')))
    
    if not score_files:
        st.error("No .npy files found in the specified directory.")
        return {}
    
    scores_dict = {}
    for score_file in score_files:
        file_base = os.path.basename(score_file).replace('_scores.npy', '')
        try:
            scores = np.load(score_file, allow_pickle=True)
            scores_dict[file_base] = scores
        except Exception as e:
            st.warning(f"Failed to load {score_file}: {e}")
    
    return scores_dict

def aggregate_scores(scores_dict, threshold=90):
    """
    Aggregate scores from all files.
    """
    aggregated_data = []
    
    for file_idx, (file_name, scores) in enumerate(sorted(scores_dict.items()), 1):
        mse_scores = [entry[0] for entry in scores]
        confidence_scores = [entry[1] for entry in scores]
        anomalies = [score > threshold for score in confidence_scores]
        num_anomalies = sum(anomalies)
        anomaly_percentage = (num_anomalies / len(confidence_scores)) * 100 if confidence_scores else 0
        
        aggregated_data.append({
            'File_Index': file_idx,
            'File_Name': file_name,
            'MSE_Scores': mse_scores,
            'Confidence_Scores': confidence_scores,
            'Anomalies': anomalies,
            'Num_Anomalies': num_anomalies,
            'Anomaly_Percentage': anomaly_percentage
        })
    
    df = pd.DataFrame(aggregated_data)
    return df

def plot_alert_evolution(df, threshold):
    """
    Plot confidence scores over time as a line chart.
    """
    st.subheader("Alert Levels Over Time")
    confidence_records = []
    for _, row in df.iterrows():
        file_index = row['File_Index']
        for score in row['Confidence_Scores']:
            confidence_records.append({'File_Index': file_index, 'Confidence_Score': score})
    df_confidence = pd.DataFrame(confidence_records)
    
    fig = px.line(
        df_confidence, 
        x="File_Index", 
        y="Confidence_Score", 
        title="Alert Levels Over Time", 
        labels={'File_Index': 'File Sequence', 'Confidence_Score': 'Confidence Score'},
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    st.plotly_chart(fig, use_container_width=True)

def plot_top_agents():
    """
    Plot top agents as a donut chart.
    """
    st.subheader("Top Agents")
    labels = ['macOS', 'CentOS', 'RHEL7', 'Windows', 'Debian']
    values = [3000, 5000, 4000, 4500, 2500]
    fig = px.pie(values=values, names=labels, hole=0.4, title="Top 5 Agents")
    st.plotly_chart(fig, use_container_width=True)

def plot_mitre_attack():
    """
    Plot MITRE ATT&CK Techniques as a donut chart.
    """
    st.subheader("MITRE ATT&CK Techniques")
    labels = ['Password Guessing', 'SSH', 'Brute Force', 'Valid Accounts', 'System Binary Proxy']
    values = [10, 20, 30, 25, 15]
    fig = px.pie(values=values, names=labels, hole=0.4, title="MITRE ATT&CK Techniques")
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(layout="wide")
    st.title("🔍 Enhanced Anomaly Detection Dashboard")
    
    # Sidebar inputs
    st.sidebar.header("Settings")
    scores_directory = st.sidebar.text_input(
        "Scores Directory",
        value="/home/ubuntu/GHIDS2/Senior-Capstone/inference/data/scaled_features"
    )
    threshold = st.sidebar.slider("Confidence Threshold (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0)
    
    # Main dashboard
    if st.sidebar.button("Load and Visualize"):
        if not os.path.isdir(scores_directory):
            st.error("Directory does not exist. Please check the path.")
            return
        
        scores_dict = load_scores(scores_directory)
        if not scores_dict:
            st.error("No valid score files found.")
            return
        
        df_aggregated = aggregate_scores(scores_dict, threshold)
        
        # Top-level statistics
        st.markdown("### Overall Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Alerts", len(df_aggregated))
        col2.metric("Level 12 Alerts", len(df_aggregated[df_aggregated['Anomaly_Percentage'] > 12]))
        col3.metric("Auth Failures", "33624")
        col4.metric("Auth Success", "58")
        
        # Horizontal layout for charts
        st.markdown("### Detailed Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            plot_alert_evolution(df_aggregated, threshold)
        with col2:
            plot_top_agents()
        with col3:
            plot_mitre_attack()
        
        # Security alerts table
        st.markdown("### Security Alerts")
        st.dataframe(df_aggregated[['File_Name', 'Num_Anomalies', 'Anomaly_Percentage']], use_container_width=True)
    
if __name__ == "__main__":
    main()
