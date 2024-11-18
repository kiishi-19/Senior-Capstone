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
        dict: Dictionary with file names as keys and lists of (mse, confidence_score) tuples as values.
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
            scores_dict[file_base] = scores  # Each entry is a tuple (mse, confidence_score)
        except Exception as e:
            st.warning(f"Failed to load {score_file}: {e}")
    
    if not scores_dict:
        st.error("No valid score data found. Please check your .npy files.")
    
    return scores_dict

def aggregate_scores(scores_dict, threshold=90):
    """
    Aggregate scores from all files.

    Args:
        scores_dict (dict): Dictionary with file names as keys and lists of (mse, confidence_score) tuples as values.
        threshold (float): Confidence score threshold to flag anomalies.

    Returns:
        pd.DataFrame: DataFrame containing aggregated scores and anomaly information.
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

def plot_confidence_over_time(df, threshold=90):
    """
    Plot confidence scores over time with anomalies highlighted.

    Args:
        df (pd.DataFrame): DataFrame containing aggregated scores and anomaly information.
        threshold (float): Confidence score threshold to flag anomalies.
    """
    st.subheader("Confidence Scores Over Time")
    plt.figure(figsize=(18, 6))
    sns.set(style="whitegrid")
    
    # Prepare data for plotting
    confidence_records = []
    for _, row in df.iterrows():
        file_index = row['File_Index']
        for score in row['Confidence_Scores']:
            confidence_records.append({'File_Index': file_index, 'Confidence_Score': score})
    
    df_confidence = pd.DataFrame(confidence_records)
    
    # Plotting
    sns.lineplot(data=df_confidence, x='File_Index', y='Confidence_Score', label='Confidence Score', color='blue', alpha=0.6)
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold}%)')
    
    # Highlight anomalies
    anomalies = df_confidence[df_confidence['Confidence_Score'] > threshold]
    sns.scatterplot(data=anomalies, x='File_Index', y='Confidence_Score', color='red', label='Anomalies', s=20, marker='X')
    
    plt.title('Confidence Scores Over Time', fontsize=16)
    plt.xlabel('File Sequence (1-99)', fontsize=14)
    plt.ylabel('Confidence Score (%)', fontsize=14)
    plt.legend()
    plt.ylim(0, 100)  # Assuming confidence scores are between 0 and 100
    st.pyplot(plt)
    plt.close()

def plot_confidence_distribution(df, plot_type='Histogram', threshold=90):
    """
    Plot the distribution of confidence scores.

    Args:
        df (pd.DataFrame): DataFrame containing aggregated scores and anomaly information.
        plot_type (str): Type of plot ('Histogram' or 'Box Plot').
        threshold (float): Confidence score threshold to flag anomalies.
    """
    st.subheader(f"{plot_type} of Confidence Scores")
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    
    # Aggregate all confidence scores
    all_confidences = [score for sublist in df['Confidence_Scores'] for score in sublist]
    
    if plot_type == 'Histogram':
        sns.histplot(all_confidences, bins=50, kde=True, color='skyblue')
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold}%)')
        plt.title('Distribution of Confidence Scores', fontsize=16)
        plt.xlabel('Confidence Score (%)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
    elif plot_type == 'Box Plot':
        sns.boxplot(x=all_confidences, color='lightblue')
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold}%)')
        plt.title('Box Plot of Confidence Scores', fontsize=16)
        plt.xlabel('Confidence Score (%)', fontsize=14)
    else:
        st.error("Invalid plot type selected.")
        return
    
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def plot_anomalies_per_file(df, threshold=90):
    """
    Plot the number and percentage of anomalies per file.

    Args:
        df (pd.DataFrame): DataFrame containing aggregated scores and anomaly information.
        threshold (float): Confidence score threshold to flag anomalies.
    """
    st.subheader("Anomaly Percentage per File")
    plt.figure(figsize=(18, 6))
    sns.set(style="whitegrid")
    
    # Bar plot for anomaly percentage per file
    sns.barplot(x='File_Index', y='Anomaly_Percentage', data=df, palette='viridis')
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold}%)')
    
    plt.title('Anomaly Percentage per File', fontsize=16)
    plt.xlabel('File Sequence (1-99)', fontsize=14)
    plt.ylabel('Anomaly Percentage (%)', fontsize=14)
    plt.legend()
    plt.ylim(0, 100)  # Assuming percentage between 0 and 100
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def plot_confidence_interactive(df, threshold=90):
    """
    Create an interactive scatter plot of confidence scores over time using Plotly.

    Args:
        df (pd.DataFrame): DataFrame containing aggregated scores and anomaly information.
        threshold (float): Confidence score threshold to flag anomalies.
    """
    st.subheader("Interactive Confidence Scores Over Time")
    
    # Prepare data
    confidence_records = []
    for _, row in df.iterrows():
        file_index = row['File_Index']
        for score in row['Confidence_Scores']:
            confidence_records.append({'File_Index': file_index, 'Confidence_Score': score})
    
    df_confidence = pd.DataFrame(confidence_records)
    df_confidence['Anomaly'] = df_confidence['Confidence_Score'] > threshold
    
    # Create interactive scatter plot
    fig = px.scatter(
        df_confidence,
        x='File_Index',
        y='Confidence_Score',
        color='Anomaly',
        title='Interactive Confidence Scores Over Time',
        labels={'File_Index': 'File Sequence', 'Confidence_Score': 'Confidence Score (%)'},
        hover_data=['Confidence_Score']
    )
    
    # Add threshold line
    fig.add_shape(
        type='line',
        x0=df_confidence['File_Index'].min(),
        y0=threshold,
        x1=df_confidence['File_Index'].max(),
        y1=threshold,
        line=dict(color='Red', dash='dash'),
        name='Threshold'
    )
    
    # Update layout for better visualization
    fig.update_layout(
        legend_title='Anomaly',
        xaxis=dict(title='File Sequence (1-99)'),
        yaxis=dict(title='Confidence Score (%)', range=[0, 100]),
        shapes=[dict(
            type='line',
            x0=df_confidence['File_Index'].min(),
            y0=threshold,
            x1=df_confidence['File_Index'].max(),
            y1=threshold,
            line=dict(color='Red', dash='dash')
        )]
    )
    
    st.plotly_chart(fig)

def main():
    st.title("🔍 Anomaly Detection Visualization Dashboard")
    
    st.markdown("""
    This dashboard visualizes the results of your anomaly detection model using confidence scores derived from an autoencoder.
    Adjust the parameters and explore the plots to gain insights into the anomalies detected across your data files.
    """)
    
    # Input: Directory containing .npy score files
    scores_directory = st.text_input(
        "Enter the path to the directory containing `_scores.npy` files:",
        value="/home/ubuntu/GHIDS2/Senior-Capstone/inference/data/scaled_features"
    )
    
    # Input: Confidence Threshold
    threshold = st.slider(
        "Select Confidence Threshold (%) for Anomaly Detection:",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=1.0
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
            df_aggregated = aggregate_scores(scores_dict, threshold=threshold)
        
        st.success("Scores loaded and aggregated successfully!")
        
        # Display basic statistics
        st.subheader("Basic Statistics")
        st.write(df_aggregated[['File_Name', 'Num_Anomalies', 'Anomaly_Percentage']])
        
        # Generate visualizations
        plot_confidence_over_time(df_aggregated, threshold=threshold)
        plot_confidence_distribution(df_aggregated, plot_type='Histogram', threshold=threshold)
        plot_anomalies_per_file(df_aggregated, threshold=threshold)
        plot_confidence_distribution(df_aggregated, plot_type='Box Plot', threshold=threshold)
        plot_confidence_interactive(df_aggregated, threshold=threshold)
        
        st.balloons()  # Fun animation to indicate completion

if __name__ == "__main__":
    main()
