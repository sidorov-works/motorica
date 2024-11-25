# Работа с табличными данными
import pandas as pd
import numpy as np

# Преобразование признаков
from sklearn.preprocessing import MinMaxScaler

# Кодирование признаков
from sklearn.preprocessing import OneHotEncoder

# Модели
from sklearn.linear_model import LogisticRegression

# Пайплайн
from sklearn.base import BaseEstimator, TransformerMixin

from motorica.utils import *

# Аннотирование
from typing import Any


#----------------------------------------------------------------------------------------
class PronationPredictor(BaseEstimator, TransformerMixin):

    '''
    Класс-преобразователь для добавления в данные признака `"act_pronation_pred"` – метки пронации.
    В размеченных нами данных метка пронации уже присутствует и называется `"act_pronation"`. 
    Метка принимает одно из трех возможных значений: 
    - `0` - ладонь вверх
    - `1` - ладонь вбок
    - `2` - ладонь вниз

    Метка сформирована на основании файлов описания протокола и в тестовой выборке не может быть 
    использована в качестве признака (поскольку в реальных данных никаких описаний протоколов 
    уже не будет).

    Применеие же данного класса поволяет ввести признак пронации в тестовые данных 
    и использовать его для предсказания жеста (после ***one-hot* кодирования**)

    ### Параметры объекта класса
    
    **features**: *List[str], default=cols_acc + cols_gyr*<br>cписок признаков для предсказания пронации
    (по умолчанию берутся каналы ACC и GYR)
    
    **pron_col**: *str, default='act_pronation'*<br>название столбца с истиным значением метки пронации
    
    **predicted_pron_col**: *str, default='act_pronation_pred'*<br>название столбца для предсказанной метки<br>
    (в обучающем датафрейме новый столбец будет полностью аналогичен исходному столбцу с разметкой)
        
    **model**: *default=LogisticRegression()*<br>модель (необученная), используемая для предсказания метки
        
    **scaler**: *default=MinMaxScaler()*<br>объект-преобразователь (необученный) для шкалирования признаков

    ### Методы
    Данный класс реализует стандартые методы классов-преобразователей *scikit-learn*:

    `fit()`, `fit_transform()` и `transform()` и может быть использован как элемент пайплайна.
    '''

    def __init__(
        self,
        features: List[str] = cols_acc + cols_gyr,
        pron_col: str = 'act_pronation',
        predicted_pron_col: str = 'act_pronation_pred',
        model = LogisticRegression(),
        scaler = MinMaxScaler(),
        encode = False,
        drop_pron_col = False
    ):
        self.features = features.copy()
        self.pron_col = pron_col
        self.model = model
        self.scaler = scaler
        self.predicted_pron_col = predicted_pron_col
        self.encode = encode
        self.drop_pron_col = drop_pron_col

    def fit(self, X, y=None):
        self.model = LogisticRegression()
        X_copy = X[self.features]
        y = X[self.pron_col]
        X_copy_scaled = self.scaler.fit_transform(X_copy)
        self.model.fit(X_copy_scaled, y)
        return self
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        X_copy = X.copy()
        # при вызове fit_transform нам не нужно предсказывать пронацию, 
        # мы просто используем имеющийся столбец с размеченной пронацией
        X_copy[self.predicted_pron_col] = X_copy[self.pron_col]
        if self.encode:
            X_copy = pd.get_dummies(X_copy, columns=[self.predicted_pron_col])
        if self.drop_pron_col:
            X_copy.drop(self.pron_col, axis=1, inplace=True)

        return X_copy
    
    def transform(self, X):
        X_copy_full = X.copy()
        X_copy_scaled = self.scaler.transform(X_copy_full[self.features])
        X_copy_full[self.predicted_pron_col] = self.model.predict(X_copy_scaled)
        if self.encode:
            X_copy_full = pd.get_dummies(X_copy_full, columns=[self.predicted_pron_col])
        if self.drop_pron_col and self.pron_col in X_copy_full:
            X_copy_full.drop(self.pron_col, axis=1, inplace=True)

        return X_copy_full
    
