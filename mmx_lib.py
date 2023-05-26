import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import date
import re
import os
import math
import functools 

from tqdm.notebook import tqdm

from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_predict, TimeSeriesSplit, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics.scorer import make_scorer

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def series_norm(x, from_zero=True):
    if from_zero:
        return (x - x.min())/(x.max()-x.min())
    else:
        return (x - x.mean())/(x.max()-x.min())

def long_term_correlation(s1, s2, w1=7, w2=7, diff_w=7):
    """Вычисляет корреляцию на длительных промежутках времени. Показывает в целом, позожи ли тренды."""
    s1 = series_norm(s1)
    s2 = series_norm(s2)
 
    #Не так важно, центрированное ли усреднение, главное, чтобы одинаковое.
    s1_diff = s1.rolling(window=w1, center=True).mean().diff().rolling(window=diff_w).mean().dropna()
    s2_diff = s2.rolling(window=w2, center=True).mean().diff().rolling(window=diff_w).mean().dropna()
    common_index = set(s1_diff.index).intersection(set(s2_diff.index))
    #corrcoef очень чувствителен к длине строк. Он не смотрит на индексы
    s1_diff = s1_diff[s1_diff.index.isin(common_index)]
    s2_diff = s2_diff[s2_diff.index.isin(common_index)]
    corr = np.corrcoef(s1_diff, s2_diff)[0][1]
    return corr

delay = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

def correlations_with_shift(x, delay):
    """Считает корреляции между трендами с разными сдвигами"""
    keys = list(x.keys())
    pairs = [(k1, k2) for i, k1 in enumerate(keys[:-1]) for k2 in keys[i+1:]]
    output_dict = {}
    for pair in pairs:
        k1, k2 = pair
        x1, w1 = x[k1]
        x2, w2 = x[k2]
        delay_corr = {}
        for d in delay:
            x1_ = x1.shift(d)
            delay_corr[d] = long_term_correlation(x1_, x2, w1, w2, 5)
        max_corr_delay = max(delay_corr, key=delay_corr.get)
        output_dict[f'{k1} & {k2}']  = (max_corr_delay, round(delay_corr[max_corr_delay], 2))
    return output_dict


def sigmoid(x):
    return int(((1 / (1 + math.exp(-x*10)))-0.5)>0)

def mean_and_norm(df_tuple, reg_w):
    """Усредняет и нормирует список таблиц"""
    df_list = []
    for df in df_tuple:
        df = df.rolling(window=reg_w, center=True).mean()
        df = series_norm(df)
        df_list.append(df)
    return tuple(df_list)

def find_holes(x, width):
    x_hole = x.rolling(width).sum()==0
    x_walls = x.shift(-1).rolling(width+2).sum()==2
    x_result = x_hole&x_walls
    x_results = [x_result.shift(-i).fillna(0) for i in range(width)]
    x_result = functools.reduce((lambda a, b: a|b), x_results)
    return x_result

def find_range_of_holes(x, hole_range):
    s, f = hole_range
    if s!=f:
        list_of_masks = []
        for width in range(s, f):
            list_of_masks.append(find_holes(x, width))
        mask = functools.reduce((lambda a, b: a|b), list_of_masks)
    else:
        mask = find_holes(x, s)
    return mask

def select_periods(x, flag, border=True):
    """
    сравнивает два списка, оставляя в первом только те периоды, 
    на которые выпадают единички во втором списке
    """
    x = x.astype(int)
    flag = flag.astype(int)
    x_period = []
    prev_x = 0
    n_period = 0
    if (len(x)==len(flag)):
        for el in x:
            if el-prev_x==1:
                n_period = n_period + 1
                x_period.append(n_period)
            elif el-prev_x==-1:
                x_period.append(0)
            elif (el+prev_x)==0:
                x_period.append(0)
            else:
                x_period.append(n_period)
            prev_x = el        
        x_period = pd.Series(x_period, index=x.index)
        for n in range(1, n_period+1):
            if border:
                if flag[(x_period==n).to_list()].sum()>0:
                    x[x_period==n] = 1
                else:
                    x[x_period==n] = 0
            else:
                if flag[(x_period==n).to_list()].sum()==0:
                    x[x_period==n] = 1
                else:
                    x[x_period==n] = 0
        return x
    else:
        print('разные длины!')
        return ''
    

def delay_func(y_pred, y_true):
    """Метрика качества, возвращающая величину несовпадения границ периодов"""
    y_pred = pd.Series(list(y_pred)).astype(int)
    y_true = pd.Series(list(y_true)).astype(int)
    start_late = select_periods(((y_true-y_pred)==1)[1:], (y_pred.diff()==1)[1:])
    start_early = select_periods(((y_true-y_pred)==-1)[1:], (y_pred.diff()==1)[1:])
    end_late = select_periods(((y_true-y_pred)==-1)[1:], (y_pred.diff()==-1)[1:])
    end_early = select_periods(((y_true-y_pred)==1)[1:], (y_pred.diff()==-1)[1:])
    start = ((start_early+start_late)>0).astype(int)
    end = ((end_early+end_late)>0).astype(int)
    #налисие лишнего пика
    false_alarm = select_periods(y_true-y_pred, y_true, border=False)
    
    periods = []
    for i, timeseries in enumerate([start, end, false_alarm]):
        period = [0]
        for el in timeseries:
            if el:
                period[-1] = period[-1] + 1
            else:
                if period[-1]!=0:
                    period.append(0)
        periods.append(np.mean(period))
        

    return np.mean(periods)

#создадим метрику качества
delay_score = make_scorer(delay_func, greater_is_better=False)