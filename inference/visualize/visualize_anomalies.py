import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
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
        print("No .npy files found in the directory.")
        return {}
    
    scores_dict = {}
    for score_file in score_files:
        file_base = os.path.basename(score_file).replace('_scores.npy', '')
        scores = np.load(score_file, allow_pickle=True)
        scores_dict[file_base] = scores  # Each entry is a tuple (mse, confidence_score)
    
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

def plot_confidence_over_time(df, threshold=90, scores_directory='.'):
    """
    Plot confidence scores over time with anomalies highlighted.

    Args:
        df (pd.DataFrame): DataFrame containing aggregated scores and anomaly information.
        threshold (float): Confidence score threshold to flag anomalies.
        scores_directory (str): Directory to save the plot.
    """
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
    plt.tight_layout()
    plot_path = os.path.join(scores_directory, 'confidence_scores_over_time.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Confidence scores over time plot saved as '{plot_path}'.")

def plot_confidence_distribution(df, scores_directory='.', plot_type='hist', threshold=90):
    """
    Plot the distribution of confidence scores.

    Args:
        df (pd.DataFrame): DataFrame containing aggregated scores and anomaly information.
        scores_directory (str): Directory to save the plot.
        plot_type (str): Type of plot ('hist' for histogram, 'box' for boxplot).
        threshold (float): Confidence score threshold to flag anomalies.
    """
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    
    # Aggregate all confidence scores
    all_confidences = [score for sublist in df['Confidence_Scores'] for score in sublist]
    
    if plot_type == 'hist':
        sns.histplot(all_confidences, bins=50, kde=True, color='skyblue')
    elif plot_type == 'box':
        sns.boxplot(x=all_confidences, color='lightblue')
    else:
        print("Invalid plot type specified. Choose 'hist' or 'box'.")
        return
    
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold}%)')
    
    if plot_type == 'hist':
        plt.title('Distribution of Confidence Scores', fontsize=16)
        plt.xlabel('Confidence Score (%)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
    else:
        plt.title('Box Plot of Confidence Scores', fontsize=16)
        plt.xlabel('Confidence Score (%)', fontsize=14)
    
    plt.legend()
    plt.tight_layout()
    plot_filename = 'confidence_scores_distribution.png' if plot_type == 'hist' else 'confidence_scores_boxplot.png'
    plot_path = os.path.join(scores_directory, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Confidence scores distribution plot saved as '{plot_path}'.")

def plot_anomalies_per_file(df, threshold=90, scores_directory='.'):
    """
    Plot the number and percentage of anomalies per file.

    Args:
        df (pd.DataFrame): DataFrame containing aggregated scores and anomaly information.
        threshold (float): Confidence score threshold to flag anomalies.
        scores_directory (str): Directory to save the plot.
    """
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
    plot_path = os.path.join(scores_directory, 'anomaly_percentage_per_file.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Anomaly percentage per file plot saved as '{plot_path}'.")

def plot_confidence_interactive(df, threshold=90, scores_directory='.'):
    """
    Create an interactive scatter plot of confidence scores over time using Plotly.

    Args:
        df (pd.DataFrame): DataFrame containing aggregated scores and anomaly information.
        threshold (float): Confidence score threshold to flag anomalies.
        scores_directory (str): Directory to save the plot (optional).
    """
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
        yaxis=dict(title='Confidence Score (%)', range=[0, 100])
    )
    
    # Save the interactive plot as HTML
    plot_path = os.path.join(scores_directory, 'confidence_scores_interactive.html')
    fig.write_html(plot_path)
    print(f"Interactive confidence scores plot saved as '{plot_path}'.")
    
    # Optionally display the plot in a browser
    # fig.show()

def main(scores_directory, threshold=90):
    """
    Main function to load scores, aggregate them, and generate visualizations.

    Args:
        scores_directory (str): Path to the directory containing .npy score files.
        threshold (float): Confidence score threshold to flag anomalies.
    """
    # Load scores
    print("Loading score files...")
    scores_dict = load_scores(scores_directory)
    
    if not scores_dict:
        print("No scores to process. Exiting.")
        return
    
    # Aggregate scores
    print("Aggregating scores...")
    df_aggregated = aggregate_scores(scores_dict, threshold=threshold)
    
    # Visualizations
    print("Generating confidence scores over time plot...")
    plot_confidence_over_time(df_aggregated, threshold=threshold, scores_directory=scores_directory)
    
    print("Generating confidence scores distribution plot...")
    plot_confidence_distribution(df_aggregated, scores_directory=scores_directory, plot_type='hist', threshold=threshold)
    
    print("Generating anomaly percentage per file plot...")
    plot_anomalies_per_file(df_aggregated, threshold=threshold, scores_directory=scores_directory)
    
    # Optional: Box plot
    print("Generating confidence scores boxplot...")
    plot_confidence_distribution(df_aggregated, scores_directory=scores_directory, plot_type='box', threshold=threshold)
    
    # Optional: Interactive plot
    print("Generating interactive confidence scores plot...")
    plot_confidence_interactive(df_aggregated, threshold=threshold, scores_directory=scores_directory)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Anomaly Detection Results from .npy Score Files.")
    parser.add_argument("scores_directory", help="Path to the directory containing *_scores.npy files.")
    parser.add_argument("--threshold", type=float, default=90.0, help="Confidence score threshold for anomalies.")
    
    args = parser.parse_args()
    
    main(args.scores_directory, threshold=args.threshold)
