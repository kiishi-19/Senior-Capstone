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

    Args:
        scores_directory (str): Path to the directory containing .npy score files.

    Returns:
        dict: Dictionary with file names as keys and lists of (mse, is_anomalous) tuples as values.
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
            scores_dict[file_base] = scores  # Each entry is a tuple (mse, is_anomalous)
        except Exception as e:
            st.warning(f"Failed to load {score_file}: {e}")
    
    if not scores_dict:
        st.error("No valid score data found. Please check your .npy files.")
    
    return scores_dict

def aggregate_scores(scores_dict, baseline_threshold=81.71176):
    """
    Aggregate scores from all files based on the baseline threshold.

    Args:
        scores_dict (dict): Dictionary with file names as keys and lists of (mse, is_anomalous) tuples as values.
        baseline_threshold (float): MSE threshold to flag anomalies.

    Returns:
        pd.DataFrame: DataFrame containing aggregated scores and anomaly information.
    """
    aggregated_data = []
    
    for file_idx, (file_name, scores) in enumerate(sorted(scores_dict.items()), 1):
        mse_scores = [entry[0] for entry in scores]
        anomalies = [entry[1] for entry in scores]
        num_anomalies = sum(anomalies)
        anomaly_percentage = (num_anomalies / len(mse_scores)) * 100 if mse_scores else 0
        
        aggregated_data.append({
            'File_Index': file_idx,
            'File_Name': file_name,
            'MSE_Scores': mse_scores,
            'Anomalies': anomalies,
            'Num_Anomalies': num_anomalies,
            'Anomaly_Percentage': anomaly_percentage
        })
    
    df = pd.DataFrame(aggregated_data)
    return df

def plot_mse_distribution(df, baseline_threshold=81.71176):
    """
    Plot the distribution of MSE scores.

    Args:
        df (pd.DataFrame): DataFrame containing aggregated scores and anomaly information.
        baseline_threshold (float): MSE threshold to flag anomalies.
    """
    st.subheader("Distribution of MSE Scores")
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    
    # Aggregate all MSE scores
    all_mse_scores = [score for sublist in df['MSE_Scores'] for score in sublist]
    
    sns.histplot(all_mse_scores, bins=50, kde=True, color='skyblue')
    plt.axvline(x=baseline_threshold, color='red', linestyle='--', label=f'Baseline Threshold ({baseline_threshold})')
    plt.title('Distribution of MSE Scores', fontsize=16)
    plt.xlabel('MSE', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def plot_anomalies_per_file(df):
    """
    Plot the number and percentage of anomalies per file.

    Args:
        df (pd.DataFrame): DataFrame containing aggregated scores and anomaly information.
    """
    st.subheader("Anomaly Percentage per File")
    plt.figure(figsize=(18, 6))
    sns.set(style="whitegrid")
    
    # Bar plot for anomaly percentage per file
    sns.barplot(x='File_Index', y='Anomaly_Percentage', data=df, palette='viridis')
    plt.title('Anomaly Percentage per File', fontsize=16)
    plt.xlabel('File Index', fontsize=14)
    plt.ylabel('Anomaly Percentage (%)', fontsize=14)
    plt.ylim(0, 100)  # Assuming percentage between 0 and 100
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def plot_mse_interactive(df, baseline_threshold=81.71176):
    """
    Create an interactive scatter plot of MSE scores over time using Plotly.

    Args:
        df (pd.DataFrame): DataFrame containing aggregated scores and anomaly information.
        baseline_threshold (float): MSE threshold to flag anomalies.
    """
    st.subheader("Interactive MSE Scores Over Time")
    
    # Prepare data
    mse_records = []
    for _, row in df.iterrows():
        file_index = row['File_Index']
        for mse, anomaly in zip(row['MSE_Scores'], row['Anomalies']):
            mse_records.append({'File_Index': file_index, 'MSE': mse, 'Anomaly': anomaly})
    
    df_mse = pd.DataFrame(mse_records)
    
    # Create interactive scatter plot
    fig = px.scatter(
        df_mse,
        x='File_Index',
        y='MSE',
        color='Anomaly',
        title='Interactive MSE Scores Over Time',
        labels={'File_Index': 'File Sequence', 'MSE': 'Mean Squared Error'},
        hover_data=['MSE']
    )
    
    # Add threshold line
    fig.add_shape(
        type='line',
        x0=df_mse['File_Index'].min(),
        y0=baseline_threshold,
        x1=df_mse['File_Index'].max(),
        y1=baseline_threshold,
        line=dict(color='Red', dash='dash'),
        name='Baseline Threshold'
    )
    
    st.plotly_chart(fig)

def main():
    st.title("🔍 MSE-Based Anomaly Detection Visualization Dashboard")
    
    st.markdown("""
    This dashboard visualizes the results of your anomaly detection model using MSE scores derived from an autoencoder.
    Adjust the parameters and explore the plots to gain insights into the anomalies detected across your data files.
    """)
    
    # Input: Directory containing .npy score files
    scores_directory = st.text_input(
        "Enter the path to the directory containing `_scores.npy` files:",
        value="/home/ubuntu/GHIDS2/Senior-Capstone/inference/data/scaled_features"
    )
    
    # Button to load and visualize data
    if st.button("Load and Visualize"):
        if not os.path.isdir(scores_directory):
            st.error("The specified directory does not exist. Please check the path and try again.")
            st.stop()
        
        with st.spinner("Loading score files..."):
            scores_dict = load_scores(scores_directory)
        
        if not scores_dict:
            st.warning("No valid score data found. Please ensure that your `.npy` files are correctly formatted.")
            st.stop()
        
        with st.spinner("Aggregating scores..."):
            df_aggregated = aggregate_scores(scores_dict, baseline_threshold=81.71176)
        
        st.success("Scores loaded and aggregated successfully!")
        
        # Display basic statistics
        st.subheader("Basic Statistics")
        st.write(df_aggregated[['File_Name', 'Num_Anomalies', 'Anomaly_Percentage']])
        
        # Generate visualizations
        plot_mse_distribution(df_aggregated, baseline_threshold=81.71176)
        plot_anomalies_per_file(df_aggregated)
        plot_mse_interactive(df_aggregated, baseline_threshold=81.71176)
        
        st.balloons()  # Fun animation to indicate completion

if __name__ == "__main__":
    main()
