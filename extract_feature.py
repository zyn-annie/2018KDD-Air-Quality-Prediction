'''
Created 2018-11-5
XGBoost Algorithm working on missvalue
Author: ZHANG Yuning
'''
#!/usr/bin/python
# coding:utf8

import numpy as np
import pandas as pd
import sklearn.feature_selection
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import Imputer
from sklearn import metrics
from sklearn.decomposition import PCA 

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
#from __future__ import print_function
#计算分类的正确率
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.preprocessing import scale
from xgboost import plot_importance
from datetime import datetime
from datetime import timedelta

import random
import time

aotizhongxin = pd.read_csv('35 plot/aotizhongxin.csv')
badaling = pd.read_csv('35 plot/badaling.csv')
beibuxinqu = pd.read_csv('35 plot/beibuxinqu.csv')
daxing = pd.read_csv('35 plot/daxing.csv')
dingling = pd.read_csv('35 plot/dingling.csv')
donggaocun = pd.read_csv('35 plot/donggaocun.csv')
dongsi = pd.read_csv('35 plot/dongsi.csv')
dongsihuan = pd.read_csv('35 plot/dongsihuan.csv')
fangshan = pd.read_csv('35 plot/fangshan.csv')
fengtaihuayuan = pd.read_csv('35 plot/fengtaihuayuan.csv')
guanyuan = pd.read_csv('35 plot/guanyuan.csv')
gucheng = pd.read_csv('35 plot/gucheng.csv')
huairou = pd.read_csv('35 plot/huairou.csv')
liulihe = pd.read_csv('35 plot/liulihe.csv')
mentougou = pd.read_csv('35 plot/mentougou.csv')
miyun = pd.read_csv('35 plot/miyun.csv')
miyunshuiku = pd.read_csv('35 plot/miyunshuiku.csv')
nansanhuan = pd.read_csv('35 plot/nansanhuan.csv')
nongzhanguan = pd.read_csv('35 plot/nongzhanguan.csv')
pingchang = pd.read_csv('35 plot/pingchang.csv')
pinggu = pd.read_csv('35 plot/pinggu.csv')
qianmen = pd.read_csv('35 plot/qianmen.csv')
shunyi = pd.read_csv('35 plot/shunyi.csv')
tiantan = pd.read_csv('35 plot/tiantan.csv')
tongzhou = pd.read_csv('35 plot/tongzhou.csv')
wanliu = pd.read_csv('35 plot/wanliu.csv')
wanshouxigong = pd.read_csv('35 plot/wanshouxigong.csv')
xizhimenbei = pd.read_csv('35 plot/xizhimenbei.csv')
yanqin = pd.read_csv('35 plot/yanqin.csv')
yizhuang = pd.read_csv('35 plot/yizhuang.csv')
yongdingmennei = pd.read_csv('35 plot/yongdingmennei.csv')
yongledian = pd.read_csv('35 plot/yongledian.csv')
yufa = pd.read_csv('35 plot/yufa.csv')
yungang = pd.read_csv('35 plot/yungang.csv')
zhiwuyuan = pd.read_csv('35 plot/zhiwuyuan.csv')

aotizhongxin.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
badaling.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
beibuxinqu.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
daxing.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
dingling.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
donggaocun.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
dongsi.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
guanyuan.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
gucheng.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
liulihe.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
miyunshuiku.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
nansanhuan.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
qianmen.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
tiantan.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
wanshouxigong.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
xizhimenbei.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
yizhuang.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
yongdingmennei.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
yongledian.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
yufa.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
yungang.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
zhiwuyuan.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)
#print (dongsi.head())

#处理时间序列
def processTime(data):
    data['day'] = pd.to_datetime(data['utc_time']).dt.date
    #data['utc_time']= pd.to_datetime(data['utc_time'])
    #data = data.set_index('utc_time') # 将time设置为index
    #data = data['2017/02':'2017/05']
    #data=data.reset_index()
    #print(data.head())
    return data

def feature_vector(df, feature, N):
    rows = df.shape[0]
    #column_n = [None] * N + [df[feature][i - N] for i in range(N, rows)]
    column_n = [df[feature][i] for i in range(0, N)] + [df[feature][i - N] for i in range(N, rows)]
    column_name = "{}_{}".format(feature, N)
    df[column_name] = column_n

def feature_vector_test(df, feature):
    rows = df.shape[0]
    column_n = [df[feature][i] for i in range(0, 24)]+ [df[feature][i-24] for i in range(24, rows)]
    column_name = "{}_{}".format(feature, 1)
    df[column_name] = column_n
    
  

    
#对每一地点分别提取特征，特征包括当前和前一时刻天气数据，以及前3时刻需要训练的污染物数据
def extractFeature(data):
    data=data.drop_duplicates()
    data=processTime(data)
    #print(data.head(5))
    #data=data.drop(['longitude','latitude','stationId'], axis = 1)
    data=data.drop(['station_id'], axis = 1)
    grouped1=data.groupby('day')['PM2.5'].mean().reset_index(name='mean')
    data=pd.merge(data,grouped1,how='left',on='day')
    #data[data.columns[-1]].rename('mean')
    #print(data.head(5))
    grouped2=data.groupby('day')['PM2.5'].min().reset_index(name='min')
    data=pd.merge(data,grouped2,how='left',on='day')
    #data[data.columns[-1]].rename('min')
    
    grouped3=data.groupby('day')['PM2.5'].max().reset_index(name='max')
    data=pd.merge(data,grouped3,how='left',on='day')
    #data[data.columns[-1]].rename('max')
    """
    #grouped4=data.groupby('day').variance().reset_index(name='variance')
    grouped4=data.groupby('day').variance()
    data=pd.merge(data,grouped4,how='left',on='day')
    data[data.columns[-1]].rename('variance')
    
    grouped5=data.groupby('day').stdev()
    data=pd.merge(data,grouped5,how='left',on='day')
    data[data.columns[-1]].rename('stdev')
    """
    data['range']=data['max']-data['min']
    
    
    #temp=air_quality[['stationId','utc_time']]
    #grouped=pd.merge(temp, grouped, how='left', on='utc_time')
    data=data.drop(['day'], axis = 1)
    feature_vector_test(data, 'mean')
    feature_vector_test(data, 'max')
    feature_vector_test(data, 'min')
    feature_vector_test(data, 'range')
    for N in range(1,4):
        feature_vector(data,'PM2.5',N)
        feature_vector(data,'PM10',N)
        feature_vector(data,'O3',N)
        feature_vector(data,'temperature',N)
        feature_vector(data,'pressure',N)
        feature_vector(data,'humidity',N)
        feature_vector(data,'wind_direction',N)
        feature_vector(data,'wind_speed',N)
    data=data.drop(['mean','max','min','range'], axis = 1)
    
    return data
        
        
if __name__ == "__main__":
    aotizhongxin = extractFeature(aotizhongxin)
    badaling =extractFeature(badaling)
    beibuxinqu = extractFeature(beibuxinqu)
    daxing = extractFeature(daxing)
    dingling = extractFeature(dingling)
    donggaocun =extractFeature(donggaocun)
    dongsi = extractFeature(dongsi)
    dongsihuan = extractFeature(dongsihuan)
    fangshan = extractFeature(fangshan)
    fengtaihuayuan = extractFeature(fengtaihuayuan)
    guanyuan = extractFeature(guanyuan)
    gucheng = extractFeature(gucheng)
    huairou = extractFeature(huairou)
    liulihe = extractFeature(liulihe)
    mentougou = extractFeature(mentougou)
    miyun = extractFeature(miyun)
    miyunshuiku = extractFeature(miyunshuiku)
    nansanhuan = extractFeature(nansanhuan)
    nongzhanguan = extractFeature(nongzhanguan)
    pingchang = extractFeature(pingchang)
    pinggu = extractFeature(pinggu)
    qianmen = extractFeature(qianmen)
    shunyi = extractFeature(shunyi)
    tiantan = extractFeature(tiantan)
    tongzhou = extractFeature(tongzhou)
    wanliu = extractFeature(wanliu)
    wanshouxigong = extractFeature(wanshouxigong)
    xizhimenbei = extractFeature(xizhimenbei)
    yanqin = extractFeature(yanqin)
    yizhuang = extractFeature(yizhuang)
    yongdingmennei = extractFeature(yongdingmennei)
    yongledian = extractFeature(yongledian)
    yufa = extractFeature(yufa)
    yungang = extractFeature(yungang)
    zhiwuyuan = extractFeature(zhiwuyuan)
    
    aotizhongxin.to_csv('35plot_feature/aotizhongxin_feature.csv')
    badaling.to_csv('35plot_feature/badaling_feature.csv')
    beibuxinqu.to_csv('35plot_feature/beibuxinqu_feature.csv')
    daxing.to_csv('35plot_feature/daxing_feature.csv')
    dingling.to_csv('35plot_feature/dingling_feature.csv')
    donggaocun.to_csv('35plot_feature/donggaocun_feature.csv')
    dongsi.to_csv('35plot_feature/dongsi_feature.csv')
    dongsihuan.to_csv('35plot_feature/dongsihuan_feature.csv')
    fangshan.to_csv('35plot_feature/fangshan_feature.csv')
    fengtaihuayuan.to_csv('35plot_feature/fengtaihuayuan_feature.csv')
    guanyuan.to_csv('35plot_feature/guanyuan_feature.csv')
    gucheng.to_csv('35plot_feature/gucheng_feature.csv')
    huairou.to_csv('35plot_feature/huairou_feature.csv')
    liulihe.to_csv('35plot_feature/liulihe_feature.csv')
    mentougou.to_csv('35plot_feature/mentougou_feature.csv')
    miyun.to_csv('35plot_feature/miyun_feature.csv')
    miyunshuiku.to_csv('35plot_feature/miyunshuiku_feature.csv')
    nansanhuan.to_csv('35plot_feature/nansanhuan_feature.csv')
    nongzhanguan.to_csv('35plot_feature/nongzhanguan_feature.csv')
    pingchang.to_csv('35plot_feature/pingchang_feature.csv')
    pinggu.to_csv('35plot_feature/pinggu_feature.csv')
    qianmen.to_csv('35plot_feature/qianmen_feature.csv')
    shunyi.to_csv('35plot_feature/shunyi_feature.csv')
    tiantan.to_csv('35plot_feature/tiantan_feature.csv')
    tongzhou.to_csv('35plot_feature/tongzhou_feature.csv')
    wanliu.to_csv('35plot_feature/wanliu_feature.csv')
    wanshouxigong.to_csv('35plot_feature/wanshouxigong_feature.csv')
    xizhimenbei.to_csv('35plot_feature/xizhimenbei_feature.csv')
    yanqin.to_csv('35plot_feature/yanqin_feature.csv')
    yizhuang.to_csv('35plot_feature/yizhuang_feature.csv')
    yongdingmennei.to_csv('35plot_feature/yongdingmennei_feature.csv')
    yongledian.to_csv('35plot_feature/yongledian_feature.csv')
    yufa.to_csv('35plot_feature/yufa_feature.csv')
    yungang.to_csv('35plot_feature/yungang_feature.csv')
    zhiwuyuan.to_csv('35plot_feature/zhiwuyuan_feature.csv')