import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# Визуализация
import plotly.express as px
import plotly.io as pio
pio.templates.default = 'plotly_dark'

from typing import List


#---------------------------------------------------------------------------------------------
N_OMG_SENSORS = 50
N_GYR_SENSORS = 3
N_ACC_SENSORS = 3

COLS_OMG = list(map(str, range(N_OMG_SENSORS)))
COLS_GYR = [f'GYR{i}' for i in range(N_GYR_SENSORS)]
COLS_ACC = [f'ACC{i}' for i in range(N_ACC_SENSORS)]

pilots = [1, 2, 3, 4]

NOGO = 0
THUMB = 1
GRAB = 2
OPEN = 3
OK = 4
PISTOL = 5


#---------------------------------------------------------------------------------------------
def read_meta_info(
    filepath: str, 
    cols_with_lists: List[str] = ['mark_sensors', 'hi_val_sensors']
) -> pd.DataFrame:
    '''
    Читает в датафрейм метаданные из файла, путь к которому передан в аргументе **filepath**.
    Столбцы, названия которых переданы в списке **cols_with_lists**, преобразуются к типу *list*
    '''
    meta_info = pd.read_csv(filepath, index_col=0)
    # После чтения файла заново - столбцы со списками, стали обычными строками. 
    # Нужно их преобразовать обратно в списки
    for col in cols_with_lists:
        if col in meta_info:
            meta_info[col] = meta_info[col].apply(
                lambda x: x.strip('[]').replace("'", ' ').replace(' ', '').split(',')
            )
    return meta_info



#---------------------------------------------------------------------------------------------
def read_train_and_test(
    montage: str,
    features: List[str],
    target_col: str = 'act_label',
    subdir: str = 'marked/',
    extra_train_features: List[str] = []

) -> List:
    
    data_train = pd.read_csv(subdir + montage + ".train", index_col=0)
    data_test = pd.read_csv(subdir + montage + ".test", index_col=0)
    X_train = data_train.drop(target_col, axis=1)[features + extra_train_features]
    y_train = data_train[target_col]
    X_test = data_test.drop(target_col, axis=1)[features]
    y_test = data_test[target_col]
    
    return X_train, X_test, y_train, y_test


#---------------------------------------------------------------------------------------------
def visualize_predictions(
    X, 
    y_true, 
    y_pred=None,
    y_proba=None,
    title="",
    width=1000, height=700  
):
    fig_data = X.copy()
    fig_data['true'] = y_true * 100

    if not y_pred is None:
        fig_data['pred'] = y_pred * 100

    if not y_proba is None:
        n_classes = y_proba[0].shape[0]
        y_proba = pd.DataFrame(
            y_proba,
            columns=['proba_' + str(c) for c in range(n_classes)],
            index=fig_data.index
        )
        fig_data = pd.concat([fig_data, y_proba * 100], axis=1)

    fig = px.line(fig_data, width=width, height=height, title=title)
    fig.update_traces(line=dict(width=1))
    return fig