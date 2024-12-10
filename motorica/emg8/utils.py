# Работа с табличными данными
import pandas as pd


# --------------------------------------------------------------------------------------------
# ВИЗУАЛИЗАЦИЯ ДАННЫХ

import plotly.express as px
import plotly.io as pio
pio.templates.default = 'plotly_dark'

from pipeline import TS_COL, SYNC_COL, CMD_COL, TARGET

def fig_montage(
        data: pd.DataFrame,
        title: str = '', 
        width: int = 1200, 
        height: int = 700,
        mult_labels: int = 1_000_000,
        **extra_labels
        ):
    if TS_COL in data:
        data = data.drop(TS_COL, axis=1)
    if SYNC_COL in data:
        data = data.drop(SYNC_COL , axis=1)
    if CMD_COL in data:
        data[CMD_COL] *= mult_labels
    if TARGET in data:
        data[TARGET] *= mult_labels

    for extra_label in extra_labels:
        data[extra_label] = extra_labels[extra_label] * mult_labels
        
    fig = px.line(data, width=width, height=height, title=title)
    fig.update_traces(line=dict(width=1))
    return fig