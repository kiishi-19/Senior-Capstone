import sys
import os
import glob
import pickle
import pandas as pd

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

def load_and_combine_results(directory):
    # Find all .pkl files ending with _features_inference_anomaly_scores.pkl
    pkl_files = glob.glob(os.path.join(directory, '*_features_inference_anomaly_scores.pkl'))
    if not pkl_files:
        raise FileNotFoundError("No .pkl files found matching '*_features_inference_anomaly_scores.pkl' in the provided directory.")

    data = []
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            all_results = pickle.load(f)
            # all_results structure:
            # all_results = [
            #   [ (loss_array_batch1, classifications_batch1), (loss_array_batch2, classifications_batch2), ... ],
            #   [ ... results for model 2 ... ],
            #   ...
            # ]
            file_name = os.path.basename(pkl_file)
            for model_idx, model_result in enumerate(all_results):
                for batch_idx, (losses, class_dict) in enumerate(model_result):
                    for i, loss in enumerate(losses):
                        row = {
                            "file_name": file_name,
                            "model_idx": model_idx,
                            "batch_idx": batch_idx,
                            "point_idx": i,
                            "reconstruction_loss": float(loss)
                        }
                        # Add anomaly columns
                        for threshold, cls_list in class_dict.items():
                            row[f"anomaly_{threshold}"] = cls_list[i]
                        data.append(row)

    df = pd.DataFrame(data)
    return df

def create_app(df):
    app = dash.Dash(__name__)

    # Unique files
    file_options = [{"label": f, "value": f} for f in df["file_name"].unique()]

    app.layout = html.Div([
        html.H1("HIDS Anomaly Visualization Dashboard"),
        
        html.Div([
            html.Label("Select File:"),
            dcc.Dropdown(
                id='file-dropdown',
                options=file_options,
                value=file_options[0]['value'] if file_options else None
            ),
        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top"}),
        
        html.Div([
            html.Label("Select Model:"),
            dcc.Dropdown(
                id='model-dropdown',
                options=[],
                value=None
            )
        ], style={"width": "30%", "display": "inline-block", "marginLeft": "20px", "verticalAlign": "top"}),
        
        html.Div([
            html.Label("Select Batch:"),
            dcc.Dropdown(
                id='batch-dropdown',
                options=[],
                value=None
            )
        ], style={"width": "30%", "display": "inline-block", "marginLeft": "20px", "verticalAlign": "top"}),

        html.Hr(),
        
        html.Div([
            dcc.Graph(id='loss-histogram'),
            dcc.Graph(id='loss-scatter')
        ])
    ])

    @app.callback(
        Output('model-dropdown', 'options'),
        Output('model-dropdown', 'value'),
        Input('file-dropdown', 'value')
    )
    def update_models(selected_file):
        if selected_file is None:
            return [], None
        filtered = df[df['file_name'] == selected_file]
        models = filtered['model_idx'].unique()
        opts = [{'label': f'Model {m}', 'value': m} for m in models]
        return opts, (models[0] if len(models) > 0 else None)

    @app.callback(
        Output('batch-dropdown', 'options'),
        Output('batch-dropdown', 'value'),
        Input('file-dropdown', 'value'),
        Input('model-dropdown', 'value')
    )
    def update_batches(selected_file, selected_model):
        if selected_file is None or selected_model is None:
            return [], None
        filtered = df[(df['file_name'] == selected_file) & (df['model_idx'] == selected_model)]
        batches = filtered['batch_idx'].unique()
        opts = [{'label': f'Batch {b}', 'value': b} for b in batches]
        return opts, (batches[0] if len(batches) > 0 else None)

    @app.callback(
        Output('loss-histogram', 'figure'),
        Output('loss-scatter', 'figure'),
        Input('file-dropdown', 'value'),
        Input('model-dropdown', 'value'),
        Input('batch-dropdown', 'value')
    )
    def update_figures(selected_file, selected_model, selected_batch):
        if selected_file is None or selected_model is None or selected_batch is None:
            return px.scatter(), px.scatter()
        
        filtered = df[
            (df['file_name'] == selected_file) &
            (df['model_idx'] == selected_model) &
            (df['batch_idx'] == selected_batch)
        ]
        
        fig_hist = px.histogram(filtered, x='reconstruction_loss',
                                title='Reconstruction Loss Distribution')
        
        anomaly_cols = [c for c in filtered.columns if c.startswith("anomaly_")]
        
        if anomaly_cols:
            color_col = anomaly_cols[0]  # pick the first threshold column for coloring
            fig_scatter = px.scatter(filtered, x='point_idx', y='reconstruction_loss',
                                     color=color_col,
                                     title=f'Reconstruction Loss by Data Point (Color = {color_col})')
        else:
            fig_scatter = px.scatter(filtered, x='point_idx', y='reconstruction_loss',
                                     title='Reconstruction Loss by Data Point')

        return fig_hist, fig_scatter

    return app

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python dashboard.py <directory_of_pkl_files>")
        sys.exit(1)
    
    directory = sys.argv[1]
    df = load_and_combine_results(directory)
    app = create_app(df)
    app.run_server(debug=True)
