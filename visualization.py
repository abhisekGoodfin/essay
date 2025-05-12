"""Visualization functions for model comparison results."""

import os
import logging
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

def plot_scatter(df, x_col, y_col, title, filename, charts_dir, color_col=None):
    """Generates and saves a scatter plot using Plotly."""
    hover_cols = [col for col in df.columns if col != 'Model']
    fig = px.scatter(df, x=x_col, y=y_col, text='Model', title=title,
                     labels={x_col: x_col.replace('_', ' ').title(),
                             y_col: y_col.replace('_', ' ').title()},
                     hover_data=hover_cols,
                     color=color_col,
                     color_continuous_scale=px.colors.sequential.Viridis if color_col else None)
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title=x_col.replace('_', ' ').title(),
                      yaxis_title=y_col.replace('_', ' ').title(),
                      title_font_size=18, height=600, width=800)
    try:
        Path(charts_dir).mkdir(parents=True, exist_ok=True)
        img_path = os.path.join(charts_dir, f"{filename}.png")
        html_path = os.path.join(charts_dir, f"{filename}.html")
        fig.write_image(img_path)
        fig.write_html(html_path)
        logging.info(f"Saved scatter plot: {filename}.png / .html")
    except Exception as e:
        logging.error(f"Error saving scatter plot {filename}: {e}. Ensure kaleido is installed.")

def plot_box(df_melted, y_col, title, filename, charts_dir):
    """Generates and saves a box plot using Plotly."""
    logging.warning("Box plots require loading detailed per-item results, which is currently not implemented. Skipping box plot.")

def plot_radar(df, title, filename, charts_dir):
    """Generates and saves a radar chart using Plotly."""
    df_normalized = df.copy()
    numeric_cols = [col for col in df_normalized.columns if col != 'Model' and pd.api.types.is_numeric_dtype(df_normalized[col])]

    if not numeric_cols:
        logging.warning(f"No numeric columns found for radar chart {filename}. Skipping.")
        return

    for col in numeric_cols:
         min_val = df_normalized[col].min()
         max_val = df_normalized[col].max()
         if max_val > min_val:
             df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
         else:
             df_normalized[col] = 0.5

    categories = numeric_cols
    fig = go.Figure()

    for _, row in df_normalized.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[categories].values.tolist(),
            theta=[cat.replace('_', ' ').title() for cat in categories],
            fill='toself',
            name=row['Model']
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=title,
        title_font_size=18,
        height=700, width=900
    )
    try:
        Path(charts_dir).mkdir(parents=True, exist_ok=True)
        img_path = os.path.join(charts_dir, f"{filename}.png")
        html_path = os.path.join(charts_dir, f"{filename}.html")
        fig.write_image(img_path)
        fig.write_html(html_path)
        logging.info(f"Saved radar chart: {filename}.png / .html")
    except Exception as e:
         logging.error(f"Error saving radar chart {filename}: {e}. Ensure kaleido is installed.")

def generate_comparison_charts(summary_df: pd.DataFrame, charts_dir: str) -> None:
    """
    Generates various comparison charts based on the summary evaluation data.
    Uses Plotly for interactive charts. Handles multiple prompt sets.

    Args:
        summary_df: DataFrame containing the aggregated comparison stats
                    (must include 'Model' and 'PromptSet' columns).
        charts_dir: Directory to save the generated charts.
    """
    logging.info("\n--- Generating Comparison Charts (Original vs JPM Prompts) --- ")

    if summary_df.empty or 'Model' not in summary_df.columns or 'PromptSet' not in summary_df.columns:
        logging.warning("Summary DataFrame is empty or missing required 'Model'/'PromptSet' columns. Skipping chart generation.")
        return

    try:
        df = summary_df.copy()

        numeric_cols = [
            'average_similarity', 'std_dev_similarity',
            'average_time_per_question', 'total_time_seconds',
            'estimated_cost_per_question', 'estimated_total_cost',
            'answer_success_rate', 'grade_success_rate',
            'average_tokens_per_question', 'total_input_tokens', 'total_completion_tokens',
            'total_questions', 'successful_answers', 'successful_grades'
        ]

        if "average_marks" in df.columns:
             df["average_marks"] = pd.to_numeric(df["average_marks"], errors='coerce')
             if pd.api.types.is_numeric_dtype(df["average_marks"]):
                 numeric_cols.append("average_marks")
             else:
                 logging.warning("'average_marks' column contains non-numeric data and cannot be plotted numerically.")

        for col in numeric_cols:
            if col in df.columns and col != "average_marks":
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.fillna(0, inplace=True)

        rename_map = {
            "answer_success_rate": "Answer Success Rate",
            "grade_success_rate": "Grade Success Rate",
            "average_similarity": "Avg Similarity",
            "average_time_per_question": "Avg Time (s)",
            "average_tokens_per_question": "Avg Tokens",
            "estimated_cost_per_question": "Avg Cost ($)",
            "average_marks": "Avg Marks"
        }
        df.rename(columns=rename_map, inplace=True)

        metrics_to_plot_renamed = [rename_map.get(col, col) for col in numeric_cols if col in rename_map]

    except Exception as e:
        logging.error(f"Error processing DataFrame for plotting: {e}")
        return

    logging.info("Generating grouped bar charts...")
    for metric in metrics_to_plot_renamed:
        if metric in df.columns:
            title = f"Model Comparison: {metric} (Original vs JPM Prompt)"
            filename = f"grouped_bar_{metric.lower().replace(' ', '_').replace('($)', 'usd').replace('(s)', 'sec')}"
            try:
                fig = px.bar(df, x='Model', y=metric, color='PromptSet',
                             title=title, text_auto='.3f',
                             labels={'Model': 'Model', metric: metric, 'PromptSet': 'Prompt Set'},
                             barmode='group')

                fig.update_layout(xaxis_tickangle=-45, title_font_size=18, height=600, width=max(800, len(df['Model'].unique())*80))

                img_path = os.path.join(charts_dir, f"{filename}.png")
                html_path = os.path.join(charts_dir, f"{filename}.html")
                fig.write_image(img_path)
                fig.write_html(html_path)
                logging.info(f"Saved grouped bar chart: {filename}.png / .html")
            except Exception as e:
                logging.error(f"Error saving grouped bar chart {filename}: {e}. Ensure kaleido is installed.")

    logging.info("Generating scatter plots...")
    scatter_pairs = [
        ("Avg Cost ($)", "Avg Similarity"),
        ("Avg Time (s)", "Avg Similarity"),
        ("Avg Cost ($)", "Avg Time (s)")
    ]
    for x_col, y_col in scatter_pairs:
        if x_col in df.columns and y_col in df.columns:
            title = f"{x_col} vs. {y_col} by Prompt Set"
            filename = f"scatter_{x_col.lower().split()[1]}_vs_{y_col.lower().split()[1]}_by_promptset"
            try:
                fig = px.scatter(df, x=x_col, y=y_col, color='PromptSet', symbol='PromptSet',
                                 hover_name='Model', hover_data=df.columns,
                                 title=title,
                                 labels={'PromptSet': 'Prompt Set'})
                fig.update_layout(title_font_size=18, height=600, width=800)

                img_path = os.path.join(charts_dir, f"{filename}.png")
                html_path = os.path.join(charts_dir, f"{filename}.html")
                fig.write_image(img_path)
                fig.write_html(html_path)
                logging.info(f"Saved scatter plot: {filename}.png / .html")
            except Exception as e:
                logging.error(f"Error saving scatter plot {filename}: {e}. Ensure kaleido is installed.")

    logging.info("Generating radar chart...")
    radar_metrics_renamed = [
        "Avg Similarity", "Avg Marks",
        "Answer Success Rate", "Grade Success Rate",
    ]

    df_radar_base = df[['Model', 'PromptSet'] + [m for m in radar_metrics_renamed if m in df.columns]].copy()

    cost_col = "Avg Cost ($)"
    time_col = "Avg Time (s)"
    cost_inv_col = "Cost Efficiency (Inv)"
    time_inv_col = "Speed (Inv Time)"

    if cost_col in df.columns:
        max_cost = df[cost_col].max()
        df_radar_base[cost_inv_col] = max_cost - df[cost_col] if max_cost > 0 else 1.0
        radar_metrics_renamed.append(cost_inv_col)
    if time_col in df.columns:
        max_time = df[time_col].max()
        df_radar_base[time_inv_col] = max_time - df[time_col] if max_time > 0 else 1.0
        radar_metrics_renamed.append(time_inv_col)

    radar_numeric_cols = [m for m in radar_metrics_renamed if m in df_radar_base.columns and m not in [cost_col, time_col]]

    if not radar_numeric_cols:
        logging.warning("No suitable numeric columns found for radar chart. Skipping.")
    else:
        df_normalized = df_radar_base.copy()
        for col in radar_numeric_cols:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val > min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
            else:
                df_normalized[col] = 0.5

        fig = go.Figure()
        categories = [col.replace('(', '').replace(')', '') for col in radar_numeric_cols]

        for _, row in df_normalized.iterrows():
            trace_name = f"{row['Model']} ({row['PromptSet']})"
            fig.add_trace(go.Scatterpolar(
                r=row[radar_numeric_cols].values.tolist(),
                theta=categories,
                fill='toself',
                name=trace_name
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            title="Overall Model Performance Profile by Prompt Set (Normalized)",
            title_font_size=18,
            legend=dict(font=dict(size=10)),
            height=700, width=900
        )
        try:
            title="Overall Model Performance Profile (Normalized)"
            filename="radar_overall_performance_by_promptset"
            img_path = os.path.join(charts_dir, f"{filename}.png")
            html_path = os.path.join(charts_dir, f"{filename}.html")
            fig.write_image(img_path)
            fig.write_html(html_path)
            logging.info(f"Saved radar chart: {filename}.png / .html")
        except Exception as e:
            logging.error(f"Error saving radar chart {filename}: {e}. Ensure kaleido is installed.") 