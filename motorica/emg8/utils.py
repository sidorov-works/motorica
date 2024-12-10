# Работа с табличными данными
import pandas as pd


# --------------------------------------------------------------------------------------------
# ВИЗУАЛИЗАЦИЯ ДАННЫХ

import plotly.express as px


def fig_montage(
        fig_data: pd.DataFrame,
        title: str = '', 
        width: int = 1200, 
        height: int = 700,
        mult_labels: int = 1_000_000,
        **extra_labels
        ):

    fig_data = fig_data.copy()

    for extra_label in extra_labels:
        fig_data[extra_label] = extra_labels[extra_label] * mult_labels
        
    fig = px.line(fig_data, width=width, height=height, title=title)
    fig.update_traces(line=dict(width=1))
    return fig