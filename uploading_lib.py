import pandas as pd
import numpy as np
import re
import os
import math
import json
import functools

from matplotlib import pyplot as plt

from datetime import datetime
from datetime import timedelta
from datetime import date
import time
 
from tqdm.notebook import tqdm

pd.set_option('mode.chained_assignment', None)
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

import urllib
import requests
import codecs
from pandas.io.json import json_normalize
import requests
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US', tz=180)

AUTH_TOKEN = "b1b0c10a222647c8" # токен из личного кабинета
#196680 - триггеры

topicId = 196680 # id топика

def youscan_upload(media):
    """Позволяет подгружать новые упоминания через youscan api"""
    last_day = str(media['published'].max()).split('T')[0]
    today = (datetime.now() + timedelta(hours=3)).date().strftime('%Y-%m-%d')
    print(last_day, today)
    if (last_day!=today):
        req_num = requests.get("https://api.youscan.io/api/external/topics/%s/mentions/" % (topicId), params={
                'apiKey': AUTH_TOKEN,
                'from': f'{last_day}T00:00',
                'to': f'{today}T00:00',      
                'size': 1,
                'fullText': True,
                'orderBy' : "seqAsc",
                'sinceSeq': 0
            })
        print('Новых комментариев будет загружено: ', req_num.json()['total'])
        df = pd.DataFrame()
        firstSeq = min([m['seq'] for m in req_num.json()['mentions']])
        lastSeq = req_num.json()['lastSeq']
        total_mentions = lastSeq - firstSeq
        sinceSeq = 0
        
        for i in tqdm(range(int(total_mentions//1000+1))): # т.к. выгружать можем олько по 1к то проходимся по всему
    
            sinceSeq = i*1000
            req = requests.get("https://api.youscan.io/api/external/topics/%s/mentions/" % (topicId), params={
                    'apiKey': AUTH_TOKEN,
                    'from': f'{last_day}T00:00',
                    'to': f'{today}T00:00',             
                    'size': 1000,
                    'fullText': True,
                    'orderBy' : "seqAsc",
                    'sinceSeq': firstSeq + sinceSeq 
                    })
            try:
                df = df.append(json_normalize(req.json()['mentions'])) # добавляем в df каждую новую тысяч
            except:
                pass
        df = df.drop_duplicates(subset=['id'])
        
#         for i in range(int(req_num.json()['total'])//1000+1): # т.к. выгружать можем олько по 1к то проходимся по всему
#             sinceSeq = i*1000
#             req = requests.get("https://api.youscan.io/api/external/topics/%s/mentions/" % (topicId), params={
#                     'apiKey': AUTH_TOKEN,
#                     'from': f'{last_day}T00:00',
#                     'to': f'{today}T00:00',          
#                     'size': 1000,
#                     'fullText': True,
#                     'orderBy' : "seqAsc",
#                     'sinceSeq': sinceSeq 
#                     })
#             df = df.append(json_normalize(req.json()['mentions'])) # добавляем в df каждую новую тысячу
        print('uploaded!')
        media = pd.concat([media, df], sort=True)
        media = media.drop_duplicates(subset=['id'])
    
    return media

def gt_upload(kws, regions=[], period='today 3-m'): 
    
    if isinstance(kws, list)&(all(isinstance(kw, str) for kw in kws)):
        kws = ['+'.join(kws)]          
    elif isinstance(kws, str):
        pass
    else:
        print('Input must be str or list of str')
        return pd.DataFrame()
    kws = [kws]
    pytrends = TrendReq(hl='en-US', tz=180)
    #Так как это полуофициальное api, то пришлось руками создать список кодов регионов, которые использует GT
    regions_codes = pd.read_excel('DATA/GTRegions.xlsx', header=0)
    regions_codes_dict = dict(zip(regions_codes['Region'], regions_codes['Code']))
    regions_names_dict = dict(zip(regions_codes['GT name'], regions_codes['Region']))
    if len(regions)==0:
        regions = sorted(list(regions_codes_dict.keys()))
    gt_dict = {}
    
    for region in tqdm(regions):
        code = regions_codes_dict[region]
        pytrends.build_payload(kws, cat=0, timeframe=period, geo=code, gprop='')
        gt_dict[region] = pytrends.interest_over_time()

    df = 'empty'
    for r in regions:
        if (r in list(gt_dict.keys()))&(len(gt_dict[r])>0):
            if isinstance(df, str):
                df = gt_dict[regions[0]][kws]
                df.columns = [regions[0]]
            else:
                df_reg = gt_dict[r][kws]
                df_reg.columns = [r]
                df = df.join(df_reg)
    df.index = df.index.to_series().astype(str)
    return df

def timeindex_week_to_days(df):
    """растягивает еженедельные данные до ежедневных"""
    from datetime import timedelta
    a = df.index[0]
    if isinstance(a, str):
        df.index = [datetime.strptime(date, "%Y-%m-%d") for date in df.index]
        a = df.index[0]
    df = pd.concat([df]*7).sort_index()     
    numdays = len(df)
    dateList = []
    for x in range (0, numdays):
        dateList.append(a + timedelta(x))
    df.index = dateList
    df.index = df.index.to_series().astype(str)
    df.index.names = ['date']
    return df

def gt_add_recent():
    """
    Позволяет подгрузить последние данные вплоть до вчерашнего дня.
    Делает это так: мы всегда берем за основу пятилетнюю выгрузку, 
    в ней точно нет скачков и она выверена самим гуглом. 
    Далее, мы проверяем, что последняя дата этой выгрузки такова, 
    что ее пересечение с трехмесячной составляет не менее двух месяцев. 
    Если это так, мы нормируем значения в трехмесяной выгрузке на пятилетние,
    опираясь на среднее значение пересекающегося периода. 
    Если пятлетняя выгрузка слишком старая, мы делаем новую и повторяем операцию.
    Все равно выгружать новые данные необходимо, 
    так как пятилетняя выгрузка отстает на одну-две недели. 
    """
    kws = 'температура+жар+сопли+насморк+кашель+горло'
    df_y = pd.read_csv('DATA/google-trends_Trigger-MMX_5years.csv', index_col=0)    
    df_m = gt_upload(kws, period='today 3-m')
    
    #Выберем общие даты у старой выгрузки и у новой.
    common_dates = df_y.index.to_series()[df_y.index.isin(
        df_m.index.to_series().sort_values().to_list())]
    
    #Это эмпирическое значение, полученное в результате построение графиков. 
    #среднее по двум месяцам обычно достаточно хорошо сбалансировано
    minimum_common_days = 60
    #Если оказывается, что общих данных меньше, чем за два месяца, 
    #то выгрузим пятилетнюю выгрузку еще раз.
    if len(common_dates)<minimum_common_days:
        time.sleep(60)
        df_y_old = df_y.copy()
        df_y = gt_upload(kws, period='today 5-y')
        df_y = timeindex_week_to_days(df_y)
        
        old_mean = df_y_old.median()
        new_mean = df_y.median()
        df_y_old = pd.concat([df_y_old[~df_y_old.index.isin(df_y.index)],
                             df_y/new_mean*old_mean])
        df_y_old.to_csv('DATA/google-trends_Trigger-MMX_5years-old.csv')
        df_y.to_csv('DATA/google-trends_Trigger-MMX_5years.csv')
        
        common_dates = df_y.index.to_series()[df_y.index.isin(df_m.index)]
    
    old_value = df_y[df_y.index.isin(common_dates)].mean()
    new_value = df_m[df_m.index.isin(common_dates)].mean()
    
    #отнормируем новые данные на среднее значение в старых
    df_m = df_m/new_value*old_value
    df_m.index = df_m.index.to_series().astype(str)
    df_m = df_m.rolling(7, min_periods=1).mean()
    result_df = pd.concat([df_y[~df_y.index.isin(common_dates)], df_m])
     
    result_df = result_df.sort_index()
    return result_df


secretKey = '492522da3cc9ef50618f1f30506a4925'
reqUrl =  'https://api.darksky.net/forecast/492522da3cc9ef50618f1f30506a4925/'
global_flags = '?exclude=currently,flags,hourly'

def api_call(lat, lon, flags):
    req = requests.get(reqUrl+str(lat)+','+str(lon)+flags)
    try:
        return json_normalize(req.json()['daily']['data'])
    except:
        print(req.json())
        return pd.DataFrame(columns=['city', 'time', 'summary', 'icon', 'sunriseTime', 'sunsetTime', 'moonPhase', 
                                     'precipIntensity', 'precipIntensityMax', 'precipIntensityMaxTime','precipProbability', 
                                     'precipType', 'temperatureHigh','temperatureHighTime', 'temperatureLow', 
                                     'temperatureLowTime','apparentTemperatureHigh', 'apparentTemperatureHighTime', 
                                     'apparentTemperatureLow', 'apparentTemperatureLowTime', 'dewPoint', 'humidity', 
                                     'pressure', 'windSpeed', 'windGust', 'windGustTime','windBearing', 'cloudCover', 
                                     'uvIndex', 'uvIndexTime', 'visibility','ozone', 'temperatureMin', 'temperatureMinTime',
                                     'temperatureMax','temperatureMaxTime', 'apparentTemperatureMin', 'apparentTemperatureMinTime', 
                                     'apparentTemperatureMax','apparentTemperatureMaxTime'])

def get_data_for_city(lat, lon, city, flags):
    df_city = pd.DataFrame(columns=['city', 'time', 'summary', 'icon', 'sunriseTime', 'sunsetTime', 'moonPhase','precipIntensity',
                                    'precipIntensityMax', 'precipIntensityMaxTime','precipProbability', 'precipType', 
                                    'temperatureHigh','temperatureHighTime', 'temperatureLow', 'temperatureLowTime',
                                    'apparentTemperatureHigh', 'apparentTemperatureHighTime','apparentTemperatureLow', 
                                    'apparentTemperatureLowTime', 'dewPoint', 'humidity', 'pressure', 'windSpeed', 
                                    'windGust', 'windGustTime','windBearing', 'cloudCover', 'uvIndex', 'uvIndexTime', 
                                    'visibility','ozone', 'temperatureMin', 'temperatureMinTime', 'temperatureMax',
                                    'temperatureMaxTime', 'apparentTemperatureMin', 'apparentTemperatureMinTime', 
                                    'apparentTemperatureMax','apparentTemperatureMaxTime'])
    df_date = api_call(lat, lon, flags)
    df_date['city'] = [city for x in range(len(df_date))]
    df_city = pd.concat([df_city,df_date], sort=False)
    return df_city

def weather_metrics_generator(w):
    """генерирует погодные фичи и вовращает только полезные"""
    target_cols = ['Week']
    
    MonthMeanTemperture = w.groupby('Month')[['apparentTemperatureHigh', 'apparentTemperatureLow']].mean().reset_index()
    w = w.merge(MonthMeanTemperture, on='Month', how='left', suffixes=('', '_mean'))
    for name in ['apparentTemperatureHigh', 'apparentTemperatureLow']:
        w[name+'_delta'] = w[name] - w[name+'_mean']
        target_cols = target_cols + [name+'_delta']
    
    w['apparentTempertureDelta24'] = w['apparentTemperatureHigh'] - w['apparentTemperatureLow']
    x = w.groupby('Week')['apparentTempertureDelta24'].max().reset_index()
    w = w.merge(x, on='Week', how='left', suffixes=('', '_max'))
    target_cols.append('apparentTempertureDelta24_max')

    w['apparentTempertureDelta24'] = w['apparentTemperatureHigh'] - w['apparentTemperatureLow']
    x = w.groupby('Week').agg({'apparentTempertureDelta24': (lambda x: x.abs().mean())}).reset_index()
    x.columns = ['Week', 'Total24Delta']
    if 'Total24Delta' in list(w.columns):
            w = w.drop(columns=['Total24Delta'])
    w = w.merge(x, on='Week', how='left')
    target_cols.append('Total24Delta')

    x = w.groupby('Week').agg({'apparentTempertureDelta24': (lambda x: x.abs().max())}).reset_index()
    x.columns = ['Week', 'Total24Max']
    if 'Total24Max' in list(w.columns):
            w = w.drop(columns=['Total24Max'])
    w = w.merge(x, on='Week', how='left')
    target_cols.append('Total24Max')
    
    w['apparentTempertureDayDeltas'] = w['apparentTemperatureHigh'].diff()
    x = w.groupby('Week').agg({'apparentTempertureDayDeltas': (lambda x: x.abs().mean())}).reset_index()
    x.columns = ['Week', 'MeanDayDiff']
    if 'MeanDayDiff' in list(w.columns):
            w = w.drop(columns=['MeanDayDiff'])
    w = w.merge(x, on='Week', how='left')
    target_cols.append('MeanDayDiff')

    w['apparentTempertureNightDeltas'] = w['apparentTemperatureLow'].diff()
    x = w.groupby('Week').agg({'apparentTempertureNightDeltas': (lambda x: x.abs().mean())}).reset_index()
    x.columns = ['Week', 'MeanNightDiff']
    if 'MeanNightDiff' in list(w.columns):
            w = w.drop(columns=['MeanNightDiff'])
    w = w.merge(x, on='Week', how='left')
    target_cols.append('MeanNightDiff')

    x = w.groupby('Week').agg({'apparentTemperatureHigh': (lambda x: x.max()-x.min())}).reset_index()
    x.columns = ['Week', 'WeekDayDisper']
    if 'WeekDayDisper' in list(w.columns):
            w = w.drop(columns=['WeekDayDisper'])
    w = w.merge(x, on='Week', how='left')
    target_cols.append('WeekDayDisper')

    x = w.groupby('Week').agg({'apparentTemperatureLow': (lambda x: x.max()-x.min())}).reset_index()
    x.columns = ['Week', 'WeekNightDisper']
    if 'WeekNightDisper' in list(w.columns):
            w = w.drop(columns=['WeekNightDisper'])
    w = w.merge(x, on='Week', how='left')
    target_cols.append('WeekNightDisper')
    
    w['Dampness'] = w['windSpeed']*w['humidity']
    x = w.groupby('Week')[['Dampness']].mean().reset_index()
    x.columns = ['Week', 'MeanDampness']
    if 'MeanDampness' in list(w.columns):
            w = w.drop(columns=['MeanDampness'])
    w = w.merge(x, on='Week', how='left')
    target_cols.append('MeanDampness')
    
    w['CloudyMonth'] = (w['cloudCover']>0.8).astype(int).rolling(window=30, min_periods=1).sum()
    target_cols.append('CloudyMonth')
    
    return w[target_cols].drop_duplicates(subset=['Week'])
    

def weather_upload(weather):    

    df = pd.read_csv('DATA/cities-100.csv', sep=';')    
    folder = 'DATA/weather_raw/'
    
    df_tmp = df[:]
 
    for index, row in df_tmp.iterrows():
        name = [f for f in os.listdir(folder) if row["city"] in f][0]
        last_day = name.split('_')[-1].split('.')[0]

        timestamps_to_upload = []
        for single_date in [date.fromisoformat(last_day)+timedelta(n) for n in range(1,
                        (date.today()-date.fromisoformat(last_day)).days)]:
            timestamps_to_upload.append(int(time.mktime(single_date.timetuple())))
        timestamps_to_upload = timestamps_to_upload[:5]
        if len(timestamps_to_upload)!=0:             
            for i, our_timestamp in enumerate(timestamps_to_upload):
                global_flags = f',{our_timestamp}?exclude=currently,flags'    
                output = get_data_for_city(row['geo_lat'], row['geo_lon'], row['city'], global_flags)
                if i==0:
                    df_row = output
                else:
                    df_row = pd.concat([output, df_row], sort=False)

            file = pd.read_csv(f'DATA/weather_raw/{row["city"]}_{last_day}.csv')
            file = pd.concat([df_row, file], sort=False)
            new_last_day = date.fromtimestamp(timestamps_to_upload[-1]).isoformat()

            file.to_csv(f'DATA/weather_raw/{row["city"]}_{new_last_day}.csv', index=False)
            os.remove(f'DATA/weather_raw/{row["city"]}_{last_day}.csv')

    weather_folder = 'DATA/'
    files = sorted([file for file in os.listdir(weather_folder+'weather_raw/') if '.csv' in file])
    history_files_date = files[0]
    history_files_date = history_files_date.split('_')[-1].split('.')[0]

    #Геоданные
    geo = pd.read_csv('DATA/geo-rus.csv')

    temp_cols = ['apparentTemperatureHigh', 'apparentTemperatureLow', 'apparentTemperatureMax', 
             'apparentTemperatureMin', 'temperatureHigh', 'temperatureLow', 'temperatureMax', 
             'temperatureMin']
    region_dict = {}
    for file in tqdm(files):
        city = file.split('_')[0]    
        if city not in list(geo['Город']):
            pass
        else:
            reg = geo[geo['Город']==city]['Регион'].values[0]
            if reg not in list(region_dict.keys()):
                region_dict[reg] = {} 
            try:
                #загрузим файлы с историческими данными и с прогнозом
                #w = pd.read_csv(f'{weather_folder}forecast/{file}')
                w = pd.read_csv(f'{weather_folder}weather_raw/{city}_{history_files_date}.csv')
            except:
                break
            #обработаем их вместе
            #w = pd.concat([w, w_history], sort=False)

            w[temp_cols] = w[temp_cols].apply(lambda Fahrenheit: (Fahrenheit - 32) * 5/9)
            w['Date'] = pd.to_datetime(w['time'], unit='s').dt.date
            w['Week'] = pd.to_datetime(w['time'], unit='s').dt.strftime('%Y-%W')        
            w['Month'] = pd.to_datetime(w['time'], unit='s').dt.month.astype(str)

            w_final = weather_metrics_generator(w)

            #запишем результат
            region_dict[reg][city] = w_final.set_index('Week')

    #Здесь производится нормировка погоды на плотность населения и заполнение пустых значений
    #Обрабатываются все данные, какие есть
    for reg in list(region_dict.keys()):
        region_dict[reg] = pd.concat(region_dict[reg].values(), keys=region_dict[reg].keys(), axis=1)
        total_pop = 0     
        cities = region_dict[reg].columns.get_level_values(0).unique().to_list()
        num_cities = len(cities)
        for i, city in enumerate(cities):
            if i==0:
                to_fill = region_dict[reg].xs(city, axis=1, level=0, drop_level=False).fillna(method='bfill')            
            else:
                to_fill = to_fill + region_dict[reg].xs(city, axis=1, level=0, drop_level=False).fillna(method='bfill')
        to_fill = to_fill/num_cities
        #посчитаем среднее по региону и для каждого пропущенного значения возьмем соседнее по дате
        #почему-то, если делать это в цикле, то последующее умножение на население копится
        #to_fill = region_dict[reg].mean(axis='items').fillna(method='bfill')
        for city in cities:        
            pop = int(geo[geo['Город']==city]['Население'].values[0])        
            region_dict[reg].loc[:, city] = region_dict[reg].xs(
                                                city, axis=1, level=0, drop_level=False).fillna(to_fill)*pop
            total_pop += pop
        #отнормируем на общее население в регионе
        region_dict[reg] = region_dict[reg].groupby(level=1, axis=1).sum()/total_pop
        region_dict[reg] = region_dict[reg][region_dict[reg].index.notna()]
        region_dict[reg]['Регион'] = reg

    region_weather = pd.concat(region_dict.values(), sort=True)
    return region_weather
            





