#!/usr/bin/env python3

import argparse
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import orjson
import pandas as pd
import numpy as np
import os
import pickle
from itertools import islice
import time

# Argument Parsing for --json_dir and optional paths for pkl files
parser = argparse.ArgumentParser(description="HIDS Dashboard")
parser.add_argument("--json_dir", type=str, required=True, help="Directory containing JSON files")
parser.add_argument("--batch_size", type=int, default=100, help="Number of JSON files to process in a batch")
parser.add_argument("--known_syscalls", type=str, default="/home/ubuntu/GHIDS2/Senior-Capstone/output/features/known_syscalls.pkl",
                    help="Path to known syscalls pickle file")
parser.add_argument("--known_arguments", type=str, default="/home/ubuntu/GHIDS2/Senior-Capstone/output/features/known_arguments.pkl",
                    help="Path to known arguments pickle file")
args = parser.parse_args()

# Streamlit Configuration
st.set_page_config(layout="wide")  # Landscape layout
st.sidebar.title("HIDS Dashboard")

json_dir = args.json_dir
batch_size = args.batch_size
known_syscalls_path = args.known_syscalls
known_arguments_path = args.known_arguments

if json_dir:
    # Validate Directory
    if not os.path.isdir(json_dir):
        st.error(f"The provided path '{json_dir}' is not a valid directory!")
    else:
        # Load all JSON files in the directory
        json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
        total_files = len(json_files)
        if not json_files:
            st.error("No JSON files found in the directory!")
        else:
            st.sidebar.success(f"{total_files} JSON files found.")

            # Initialize Structures
            thresholds = [0.1486, 1.1148, 1.4864]
            global_time = 0
            model_anomaly_counts = {}
            model_reconstruction_errors = {}
            anomalies_time_records = []  # To store timestamped anomaly counts

            # Load Known Syscalls and Arguments
            try:
                with open(known_syscalls_path, 'rb') as f:
                    known_syscalls = pickle.load(f)
                with open(known_arguments_path, 'rb') as f:
                    known_arguments = pickle.load(f)
                st.info(f"Loaded {len(known_syscalls)} known syscalls and {len(known_arguments)} known arguments. "
                        f"{known_arguments_path} {known_syscalls_path}")
            except Exception as e:
                st.error(f"Failed to load known syscalls or arguments: {e}")
                known_syscalls = []
                known_arguments = []

            def process_files(files):
                """Process a list of JSON files."""
                global global_time
                for json_file in files:
                    file_path = os.path.join(json_dir, json_file)
                    with open(file_path, 'r') as f:
                        data = orjson.loads(f.read())

                    for model_idx, model_data in enumerate(data):
                        if model_idx not in model_anomaly_counts:
                            model_anomaly_counts[model_idx] = {threshold: 0 for threshold in thresholds}
                            model_reconstruction_errors[model_idx] = []

                        for row_idx, row in enumerate(model_data):
                            reconstruction_error, classification_dict = row[0], row[1]
                            global_time += 1
                            current_time = pd.Timestamp.now()

                            # Add reconstruction errors with timestamp
                            for loss in reconstruction_error:
                                model_reconstruction_errors[model_idx].append({
                                    "Time": current_time,
                                    "Loss": loss,
                                    "File": json_file
                                })

                            # Update anomaly counts with timestamp
                            for threshold in thresholds:
                                anomalies = sum(classification_dict.get(str(threshold), []))
                                model_anomaly_counts[model_idx][threshold] += anomalies
                                anomalies_time_records.append({
                                    "Time": current_time,
                                    "Model": f"Model {model_idx}",
                                    "Threshold": threshold,
                                    "Count": anomalies
                                })

            def batched(iterable, n):
                """Batch an iterable into chunks of size n."""
                it = iter(iterable)
                while batch := list(islice(it, n)):
                    yield batch

            # Initialize performance metrics
            start_time = time.time()

            # Process files in batches
            for batch_idx, batch_files in enumerate(batched(json_files, batch_size)):
                st.sidebar.info(f"Processing batch {batch_idx + 1} of {len(json_files) // batch_size + 1}...")
                process_files(batch_files)

            processing_time = time.time() - start_time

            # Calculate total anomalies
            total_anomalies = sum(
                count for counts in model_anomaly_counts.values() for count in counts.values()
            )

            # Convert to DataFrame for visualization
            loss_trends = [
                {"Time": entry["Time"], "Model": f"Model {model_idx}", "Loss": entry["Loss"]}
                for model_idx, errors in model_reconstruction_errors.items()
                for entry in errors
            ]
            loss_trends_df = pd.DataFrame(loss_trends)

            # Anomalies Over Time DataFrame
            anomalies_time_df = pd.DataFrame(anomalies_time_records)

            # Compile detailed anomalies data
            detailed_anomalies = []
            for model_idx, errors in model_reconstruction_errors.items():
                for entry in errors:
                    breached_thresholds = [threshold for threshold in thresholds if entry["Loss"] > threshold]
                    if breached_thresholds:
                        detailed_anomalies.append({
                            "Model": f"Model {model_idx}",
                            "Time": entry["Time"],
                            "File": entry["File"],
                            "Loss": entry["Loss"],
                            "Thresholds Breached": ", ".join(map(str, breached_thresholds))
                        })

            detailed_anomalies_df = pd.DataFrame(detailed_anomalies)

            # Calculate anomalies per file
            anomalies_per_file = detailed_anomalies_df['File'].value_counts().reset_index()
            anomalies_per_file.columns = ['File', 'Anomaly Count']

            # Dashboard Layout
            with st.container():
                st.title("Host-Based Intrusion Detection System (HIDS) Dashboard")
                
                # Display Summary Metrics
                st.subheader("Summary Metrics")
                col_total, col_avg, col_time = st.columns(3)
                
                with col_total:
                    st.metric("Total Anomalies", total_anomalies)
                
                average_anomalies = total_anomalies / len(model_anomaly_counts) if model_anomaly_counts else 0
                with col_avg:
                    st.metric("Average Anomalies per Model", f"{average_anomalies:.2f}")
                
                with col_time:
                    st.metric("Processing Time (s)", f"{processing_time:.2f}")
                
                st.markdown("---")
                
                # Filters
                with st.sidebar:
                    st.markdown("### Filters")
                    selected_models = st.multiselect(
                        "Select Models",
                        options=[f"Model {m}" for m in model_anomaly_counts.keys()],
                        default=[f"Model {m}" for m in model_anomaly_counts.keys()]
                    )
                    
                    selected_thresholds = st.multiselect(
                        "Select Thresholds",
                        options=thresholds,
                        default=thresholds
                    )
                    
                    # Date range filter
                    if not anomalies_time_df.empty:
                        min_time = anomalies_time_df["Time"].min()
                        max_time = anomalies_time_df["Time"].max()
                        selected_time = st.date_input(
                            "Select Date Range",
                            value=(min_time.date(), max_time.date()),
                            min_value=min_time.date(),
                            max_value=max_time.date()
                        )
                        anomalies_time_filtered = anomalies_time_df[
                            (anomalies_time_df["Model"].isin(selected_models)) &
                            (anomalies_time_df["Threshold"].isin(selected_thresholds)) &
                            (anomalies_time_df["Time"].dt.date >= selected_time[0]) &
                            (anomalies_time_df["Time"].dt.date <= selected_time[1])
                        ]
                    else:
                        anomalies_time_filtered = anomalies_time_df

                # Row 1: Anomalies Over Time and Model Comparison
                col1, col2 = st.columns(2)

                # Visualization: Anomaly Trends Over Time
                with col1:
                    st.subheader("Anomaly Trends Over Time")
                    if not anomalies_time_filtered.empty:
                        anomaly_trends_fig = px.line(
                            anomalies_time_filtered,
                            x="Time",
                            y="Count",
                            color="Model",
                            title="Anomaly Counts Over Time",
                            markers=True
                        )
                        st.plotly_chart(anomaly_trends_fig, use_container_width=True)
                    else:
                        st.write("No anomaly data available for the selected filters.")

                # Visualization: Anomalies by Model (Pie Chart)
                with col2:
                    st.subheader("Anomalies by Model")
                    total_anomalies_by_model = {
                        model: sum(counts.values()) for model, counts in model_anomaly_counts.items()
                    }

                    pie_chart_fig = px.pie(
                        names=[f"Model {m}" for m in total_anomalies_by_model.keys()],
                        values=total_anomalies_by_model.values(),
                        title="Distribution of Anomalies by Model",
                    )
                    st.plotly_chart(pie_chart_fig, use_container_width=True)

                # Row 2: Reconstruction Error Distribution and Heatmap
                st.subheader("Reconstruction Error Analysis")
                col3, col4 = st.columns(2)

                with col3:
                    st.subheader("Reconstruction Error Distribution by Model")
                    if not loss_trends_df.empty:
                        reconstruction_error_data = [
                            {"Model": f"Model {model_idx}", "Error": entry["Loss"]}
                            for model_idx, errors in model_reconstruction_errors.items()
                            for entry in errors
                        ]

                        histogram_fig = px.histogram(
                            pd.DataFrame(reconstruction_error_data),
                            x="Error",
                            color="Model",
                            title="Reconstruction Error Distribution",
                            nbins=50
                        )
                        st.plotly_chart(histogram_fig, use_container_width=True)
                    else:
                        st.write("No reconstruction error data available.")

                with col4:
                    st.subheader("Heatmap of Anomalies")
                    if not anomalies_time_filtered.empty:
                        heatmap_data = anomalies_time_filtered.groupby(['Model', 'Threshold']).agg({'Count': 'sum'}).reset_index()
                        heatmap_fig = px.density_heatmap(
                            heatmap_data,
                            x="Threshold",
                            y="Model",
                            z="Count",
                            color_continuous_scale="Viridis",
                            title="Anomalies Heatmap by Model and Threshold",
                        )
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                    else:
                        st.write("No heatmap data available for the selected filters.")

                # Row 3: Model Loss Over Time
                st.subheader("Model Loss Over Time")
                show_thresholds = st.checkbox("Show Threshold Lines", value=True)
                loss_time_fig = go.Figure()

                for model in loss_trends_df["Model"].unique():
                    model_data = loss_trends_df[loss_trends_df["Model"] == model]
                    loss_time_fig.add_trace(
                        go.Scatter(
                            x=model_data["Time"],
                            y=model_data["Loss"],
                            mode="lines+markers",
                            name=model,
                        )
                    )

                if show_thresholds:
                    for threshold in thresholds:
                        loss_time_fig.add_shape(
                            type="line",
                            x0=loss_trends_df["Time"].min(),
                            x1=loss_trends_df["Time"].max(),
                            y0=threshold,
                            y1=threshold,
                            line=dict(color="red", width=2, dash="dash"),
                        )
                        loss_time_fig.add_annotation(
                            x=loss_trends_df["Time"].max(),
                            y=threshold,
                            text=f"Threshold: {threshold}",
                            showarrow=False,
                            font=dict(color="red"),
                        )

                loss_time_fig.update_layout(
                    title="Reconstruction Loss Over Time",
                    xaxis=dict(title="Time"),
                    yaxis=dict(title="Loss"),
                    legend_title="Model",
                    template="plotly_dark",
                )
                st.plotly_chart(loss_time_fig, use_container_width=True)

                st.markdown("---")

                # Row 4: Detailed Anomalies and Top Files
                col5, col6 = st.columns(2)

                with col5:
                    st.subheader("Detailed Anomalies")
                    st.dataframe(detailed_anomalies_df.sort_values(by="Time", ascending=False))

                with col6:
                    st.subheader("Top Anomalous Files")
                    top_files = anomalies_per_file.head(10)  # Top 10 files
                    top_files_fig = px.bar(
                        top_files,
                        x='File',
                        y='Anomaly Count',
                        title='Top 10 Files with Most Anomalies',
                        labels={'Anomaly Count': 'Number of Anomalies'},
                        text='Anomaly Count'
                    )
                    top_files_fig.update_traces(textposition='outside')
                    top_files_fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(top_files_fig, use_container_width=True)

                # Row 5: Known Syscalls and Arguments
                st.subheader("Known Syscalls and Arguments")
                col7, col8 = st.columns(2)

                with col7:
                    with st.expander("View Known Syscalls"):
                        if known_syscalls:
                            st.write(f"**Total Known Syscalls:** {len(known_syscalls)}")
                            st.dataframe(pd.DataFrame(known_syscalls, columns=["Syscall"]))
                        else:
                            st.write("No known syscalls data available.")

                with col8:
                    with st.expander("View Known Arguments"):
                        if known_arguments:
                            st.write(f"**Total Known Arguments:** {len(known_arguments)}")
                            # Displaying arguments in a table, adjust as needed
                            st.dataframe(pd.DataFrame(known_arguments, columns=["Argument"]))
                        else:
                            st.write("No known arguments data available.")

                # Row 6: Export and Alerts
                st.subheader("Export and Alerts")
                col9, col10 = st.columns(2)

                with col9:
                    st.download_button(
                        label="Download Detailed Anomalies as CSV",
                        data=detailed_anomalies_df.to_csv(index=False).encode('utf-8'),
                        file_name='detailed_anomalies.csv',
                        mime='text/csv',
                    )

                with col10:
                    # Example Alert
                    alert_threshold = 1000  # Define your threshold
                    if total_anomalies > alert_threshold:
                        st.warning(f"Total anomalies have exceeded the threshold of {alert_threshold}!")

