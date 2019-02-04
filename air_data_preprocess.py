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

def loadcsv():
	aq4 = pd.read_csv('airQuality_201805.csv')
	aq4=aq4.dropna(thresh=8)
    aq4.to_csv('airQuality_201805_new.csv', index=None)
    aq1 = pd.read_csv('aiqQuality_201804.csv')
    aq2 = pd.read_csv('airQuality_201701-201801.csv')
    aq3 = pd.read_csv('airQuality_201802-201803.csv')
    #aq4 = pd.read_csv('airQuality_201805_new.csv')
    aq2.rename(columns={'utc_time':'time'}, inplace=True)
    aq3.rename(columns={'utc_time':'time'}, inplace=True)
    aq1.rename(columns={'station_id':'stationId'}, inplace=True)
    aq1.rename(columns=lambda x:x.replace('_Concentration',''), inplace=True)
    aq1.rename(columns={'PM25':'PM2.5'}, inplace=True)
    aq1.drop(columns='id',inplace=True)
    aq4.rename(columns={'station_id':'stationId'}, inplace=True)
    aq4.rename(columns=lambda x:x.replace('_Concentration',''), inplace=True)
    aq4.rename(columns={'PM25':'PM2.5'}, inplace=True)
    aq4.drop(columns='id',inplace=True)
    #print(aq4.head(20))
    #aq1=aq1.append(aq4)
    #print(aq1.head())
    station=aq1['stationId'].unique()
    #print (aq1.describe())
    #print (aq2.describe())
    #print (aq3.describe())
    return station,aq1,aq2,aq3,aq4
	#return station,aq1,aq2,aq3

#查找完全重复的行并删除
def findDuplicated(air_quality):
    test=air_quality.duplicated()
    #print (test)
    print (len(air_quality))
    air_quality=air_quality.drop_duplicates()
    print (len(air_quality))
    return air_quality
    
    

#对2017.01-2018.03数据文件做处理，改为按时间进行排序
def changeSeq(air_quality):
    air_quality=air_quality.groupby('stationId')
    newdata=pd.DataFrame(index=None)
    for group_name,group_data in air_quality:
        newdata=newdata.append(group_data)
    newdata=pd.DataFrame(newdata)
    #print(newdata.head())
    return newdata

#删除空缺值多于三个的空行
def findLoss(air_quality):
    l1=len(air_quality)
    air_quality.isnull().any()
    air_quality = air_quality.dropna(thresh=3)
    l2=len(air_quality)
    #print (l1)
    #print (l2)
    #print (l1-l2)
    return air_quality

#填补空缺值，用同一时间其它监测点的均值填补
def processLoss(air_quality):
    grouped=air_quality.groupby('time')
    grouped=grouped.mean()
    grouped.isnull().any()
    temp=air_quality[['stationId','time']]
    grouped=pd.merge(temp, grouped, how='left', on='time')
    #print (grouped.isnull().any())
    #print (grouped.head(10))
    return grouped


#调用函数处理空缺值，划分数据集。选取201702-05,201802-04数据，按时间排序
def preprocess(aq1,aq2,aq3,aq4):
    #result=aq2.append(aq3)
    #result=result.append(aq1)
    #result.to_csv('totalAirQuality.csv',index=None)
    #aq1=findLoss(aq1)
    #aq2=findLoss(aq2)
    #aq3=findLoss(aq3)
    aq1=findDuplicated(aq1)
    aq2=findDuplicated(aq2)
    aq3=findDuplicated(aq3)
    #aq4=findDuplicated(aq4)
    print(aq4.head(20))
    group1=processLoss(aq1)
    group2=processLoss(aq2)
    group3=processLoss(aq3)
    group4=processLoss(aq4)
    aq1=aq1.combine_first(group1)
    aq1=aq1.dropna()
    aq2=aq2.combine_first(group2)
    aq2=aq2.dropna()
    aq3=aq3.combine_first(group3)
    aq3=aq3.dropna()
    aq4=aq4.combine_first(group4)
    print(aq4.head(20))
    aq4=aq4.dropna()
    print(aq4.head(20))
    aq1=changeSeq(aq1)
    aq4=changeSeq(aq4)
    print(aq4.head(20))
    #aq2=changeSeq(aq2)
    #aq3=changeSeq(aq3)
    #print (aq3.head())
    #total_air=aq2.append(aq3)
    total_air=aq3.append(aq1)
    total_air=total_air.append(aq4)
    aqTotal=changeSeq(total_air)
   
    
    aq1.to_csv('1804aq1_withoutloss.csv',index=None)
    aq2.to_csv('1701-1801aq2_withoutloss.csv',index=None)
    aq3.to_csv('180203aq3_withoutloss.csv',index=None)
    aq4.to_csv('1805aq4_withoutloss.csv',index=None)
    aqTotal.to_csv('180205aq_withoutloss.csv',index=None)
    #total_air.to_csv('total_air.csv',index=None)
    
if __name__ == "__main__":
    station,aq1,aq2,aq3,aq4=loadcsv()
    preprocess(aq1,aq2,aq3,aq4)
    #test=changeSeq(aq4)
    #test=processLoss(test)
    #print(test.head(24))
    
    
  
