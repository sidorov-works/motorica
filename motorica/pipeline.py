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
    
    **features**: *List[str], default=COLS_ACC + COLS_GYR*<br>cписок признаков для предсказания пронации
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
        features: List[str] = COLS_ACC + COLS_GYR,
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
    

#----------------------------------------------------------------------------------------
class LagWrapper(BaseEstimator, TransformerMixin):

    def __init__(
        self, 
        estimator,         # объект модели (необученный)
        w: int = 3,        # ширина лага
        fill_val: Any = 0  # метка класса для первых w - 1 предсказаний
    ):
        self.estimator = estimator
        self.w = w
        self.fill_val = fill_val

    # Внутренний метод для формирования набора с лаговыми признаками
    def _make_lag(self, X: pd.DataFrame | np.ndarray):
        X_copy = np.array(X)
        w = self.w          # w - ширина лага
        n, m = X_copy.shape # n - кол-во строк, m - кол-во столбцов
            
        return np.vstack(
            [X_copy[i: i + w, :].reshape(m * w) for i in range(n - w + 1)]
        )
    
    # Внутренний метод для формирования набора с лаговыми признаками
    def _make_lag_diff(self, X: pd.DataFrame | np.ndarray):
        X_copy = np.array(X)
        w = self.w          # w - ширина лага
        n, m = X_copy.shape # n - кол-во строк, m - кол-во столбцов

        return np.vstack(
            [X_copy[i: i + w, :].reshape(m * w) for i in range(n - w + 1)]
        )

    def fit(self, X, y):
        # Обучаем модель на лаговых признаках. 
        # При этом первые w - 1 примеров исходных данных пропускаются 
        # и в обучении не участвуют 
        self.estimator.fit(self._make_lag(X), np.array(y)[self.w - 1:])
        # Скопируем атрибут модели .classes_ в соответсвутующий атрибут обёртки
        self.classes_ = self.estimator.classes_
        return self
    
    def predict(self, X):
        
        return np.hstack([
            # заполняем первые w - 1 "предсказанных" меток значением по умолчанию
            [self.fill_val] * (self.w - 1),
            # делаем предсказание на лаговых признаках        
            self.estimator.predict(self._make_lag(X))
        ])
    
    def predict_proba(self, X):
        classes = self.estimator.classes_
        n_classes = classes.shape[0]
        return np.vstack([
            # заполняем первые w - 1 "предсказанных" меток равновероятными значениями
            [[1 / n_classes] * n_classes] * (self.w - 1),
            # делаем предсказание на лаговых признаках        
            self.estimator.predict_proba(self._make_lag(X))
        ])
    
    def set_params(self, **params):
        # Все именованные арументы кроме перечисленных
        wrapper_params = ['w']
        for param in wrapper_params:
            if param in params:
                self.w = params.pop(param)
        # предназначены для оборачиваемой модели
        self.estimator.set_params(**params)


class LagFeatures(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        n_lags: int = 3
    ):
        self.n_lags = n_lags
        self.X_flt = np.empty((0,))

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray | pd.DataFrame, y=None):
         
        X = np.array(X)

        if self.n_lags < 2:
            return X

        m = X.shape[1] if X.ndim == 2 else X.shape[0] # кол-во исходных признаков
        w = m * self.n_lags                           # кол-во лаговых признаков
        
        # Если наш объект получает данные впервые,
        if self.X_flt.shape[0] == 0:
            # накопируем первый пример 
            self.X_flt = np.tile(X[0], self.n_lags - 1)

        self.X_flt = np.hstack((self.X_flt, X.flatten()))

        X_lag = np.vstack(
            [self.X_flt[i: i + w] for i in range(0, self.X_flt.shape[0] - w + 1, m)]
        )

        # Запомним в нашем объекте лаги только для будущего нового примера
        self.X_flt = self.X_flt[- (self.n_lags - 1) * m: ]

        return X_lag


class TLPEMarker(BasePeakMarker):
    '''
    TLPE -- two largest peaks in epoch
    '''
    # Функция для непосредственной разметки  
    def _mark(
        self,
        X: pd.DataFrame
    ) -> np.ndarray[int]:
        
        # Сглаживание
        X_omg = pd.DataFrame(X[self.mark_sensors]).rolling(self.window, center=True).median()
        # Приведение к единому масштабу
        X_omg = MinMaxScaler((1, 1000)).fit_transform(X_omg)

        peaks_std1, peaks_grad2 = self._find_peaks(
            X_omg,
            window=self.window,
            spacing=self.spacing
        )

        peaks = peaks_grad2 if self.use_grad2 else peaks_std1

        sync = X[self.sync_col].copy()
        # Сдвигаем синхронизацию
        if self.sync_shift > 0:
            sync.iloc[self.sync_shift: ] = sync.iloc[: -self.sync_shift]
            sync.iloc[: self.sync_shift] = 0
        

        # Искать максимальные пики будем внутри отрезков, 
        # определяемых по признаку синхронизации
        sync_mask = sync != sync.shift(-1)
        sync_index = X[sync_mask].index

        labels = [int(X.loc[idx + 1, self.label_col]) for idx in sync_index[:-1]]

        bounds = np.array([])

        for l, r in zip(sync_index[::2], sync_index[2::2]):
            # Берем 2 самых сильных пика
            bounds = np.append(bounds, np.sort(np.argpartition(peaks[l: r], -2)[-2:]) + l)

        X_mrk = X.copy()

        # Теперь разметим каждое измерение в наборе фактическими метками
        X_mrk[TARGET] = 0

        for i, lr in enumerate(zip(bounds, bounds[1:] + [X_mrk.index[-1] + 1])):
            l, r = lr
            # l, r - индексы начала текущего и следующего жестов соответственно
            X_mrk.loc[l: r, TARGET] = labels[i]      

        return X_mrk, bounds + self.bounds_shift