# 2018KDD-Air-Quality-Prediction
Solution of Predicting Air Quality Data of Beijing
Group 9
ZHANG Yuning 20550099   WANG Zichun 20547042
1. Introduction
	Project task:
In this data mining project, the main task is to predict 48 hours of air pollution data (including PM2.5, PM10 and O3) in Beijing on May 1 and May 2, 2018.
	Individual task:
In this project, ZHANG Yuning is responsible for pre-processing of air quality data, and feature extraction for the model. Meanwhile, WANG Zichun is responsible for pre-processing weather data and model training. The rest of other trivial works, like data analysis visualization and write the report is completed by us together.
	Key points and difficulties of the project:
First, Air quality changes very rapidly and there are many points of change. Besides, the original air quality data has much losing values, and some stations even vacancy. In addition, the amount of weather data is large, and the dataset needs to be extracted based on the station coordinates. What’s more, contaminants have complex spatial dependencies. For example, if strong winds blow from a heavily polluted area to a surrounding area, the surrounding air quality will also deteriorate. But if the data from all stations is featured, it will result in a severe overfitting.
2. Data Preprocess and Split Train and Test Dataset
2.1 Data Preprocess
Data preprocess is divided into two parts, preprocess air quality data and weather data, after that, we can get entire Dataset to extract features.
2.1.1	Air Quality Data 
1. Remove duplicated data. Some of the hour data are duplicated, remove them.
2. We use groupby function in pandas to process missing value in air data. For each row of missing data (PM2.5, PM10, O3, SO2...), fill with the mean of the corresponding data of other station at the same time. Code is shown below. After this work, if there are still missing values in dataset, drop it directly.
Code 2.1 Processing air data missing values
def processLoss(air_quality):
    grouped=air_quality.groupby('time')
    grouped=grouped.mean()
    grouped.isnull().any()
    temp=air_quality[['stationId','time']]
    grouped=pd.merge(temp, grouped, how='left', on='time')
    #print (grouped.isnull().any())
    #print (grouped.head(10))
    return grouped
2.1.2	Weather Data
1. Because the observed weather data consists human errors or equipment faults occur from time to time, so we have to clean it first.
2. And also weather data contains some ‘dirty’ data, like “999”. This part of the data is replaced by data generated randomly within the valid range, which means, if the wind speed is less than 0.5m/s (nearly no wind), the value of the wind direction is 999017.Code is shown below
Code 2.2 Processing weather data 
# miss some data in wind_direction and wind_speed
weather_data.isnull().any()
weather_data.drop_duplicates()
weather_data = weather_data.fillna(999999)
def clean_direction(d):
    if d > 360:
        return random.randint(0, 360)
    else:
        return d
weather_data['wind_direction'] = weather_data.wind_direction.apply(clean_direction)
def clean_speed(s):
    if s > 16:
        return random.randint(0, 16)
    else:
        return s
weather_data['wind_speed'] = weather_data.wind_speed.apply(clean_speed)
def clean_temperature(t):
    if not (-20 <= t <= 40):
        return random.randint(-20, 40)
    else:
        return t
weather_data['temperature'] = weather_data.temperature.apply(clean_temperature)
def clean_pressure(p):
    if not (990 <= p <= 1040):
        return random.randint(990, 1040)
    else:
        return p
weather_data['pressure'] = weather_data.pressure.apply(clean_pressure)
def clean_humidity(h):
    if not (0 <= h <= 100):
        return random.randint(0, 100)
    else:
        return h
weather_data['humidity'] = weather_data.humidity.apply(clean_humidity)  
2.2	Merging data
1.The site that needs to be predicted does not match the location of the provided weather data, and we need to manually select the appropriate grid weather data or observed weather data to fill.
 
2.Code
Code 2.3 Merging Air Quality data with Weather data
# by checking the same name between air quality data and weather data
weather['station_id'] = weather['station_id'].map(lambda x : str(x)[:-4])
air_quality['station_id'] = air_quality['station_id'].map(lambda x : str(x)[:-3])
def merge_dataframe(weather, air_quality, station):
    w_s = weather[weather['station_id'] == station]
    aq_s = air_quality[air_quality['station_id'] == station].drop(['station_id'], axis=1)
    station_data = pd.merge(w_s, aq_s, on='utc_time')
    station_data.to_csv('path' + station + '.csv', index=False)
    print(station, len(station_data))
2.3	Split Train and Test Dataset
On the one hand, we statistic and visualize PM2.5/PM10/O3 mean changes monthly, as shown in Figure 2.1. Obviously, the amounts of pollutants fluctuate dramatically with the seasons. Therefore, we selected data from 35 sites from February to May 2017 and February to April 2018 as training data.
Figure 2.1 Mean air quality each month in dongsi and dongsihuan
On the other hand, the data in 35 stations also have much difference, thus we independent forecast for each station. And after the prediction, we corrections the consequence referring to the adjacent station.
At last, the test dataset is the values of each contaminates from the next hour.
3. Feature Engineering
3.1. Weather Feature
As what we mentioned before, we split the data based on the 35 stations, and extract features for each of them. When extracting weather features, we first judge the nearest weather data from both on observe weather data and grid weather data, and we use merge function add them to corresponding air data. Then we use the slide-window method to extract features from the previous moment of the current time. Let's take the Dongsi station as an example. If we want to predict the data of 0:00 on May 1st at Dongsi, we will add the wind speed, wind direction, temperature, humidity and pressure of 23rd, 22nd, and 21st on April 30 at Dongsi as features.
3.2. Air Quality Feature
The process of extracting air quality feature is similar with weather feature. In our model, the effects between individual pollutants were not considered, so we trained three models to predict PM2.5, PM10, and O3, respectively. For instance, we use PM2.5 to illustrate the extraction of pollutant characteristics. Through common sense, we know that PM2.5 at the current moment is highly correlated with PM2.5 at the previous moment. Therefore, we continue to use the previously mentioned windowing method to extract contaminant features. In addition, we added the average, minimum, maximum and range of PM2.5 of the site in previous day. The statistical feature of the data from the previous day was introduced to prevent persistent deviations in the prediction process.
Code 3.1 Extract feature method and total features for prediction PM2.5
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

features = ['PM2.5_1', 'PM2.5_2', 'PM2.5_3',
              'temperature_1', 'temperature_2', 'temperature_3',
              'pressure_1', 'pressure_2', 'pressure_3',
              'humidity_1','humidity_2', 'humidity_3',
              'wind_direction_1', 'wind_direction_2', 'wind_direction_3',
              'wind_speed_1', 'wind_speed_2', 'wind_speed_3']
4. Models
4.1 Model Selection
Here we use XGBoost model as our main model. Compared to AdaBoost, XGBoost uses a second-order Taylor expansion to make the algorithm converge to global optimality faster, and the speed and precision of the algorithm are greatly improved. Due to our feature conclude the contaminants index of previous moment t-1, t-2, t-3 (suppose the current time is t), we need to add the results to our feature constantly. 
Figure 4.1 XGBoost
4.2 Parameters for XGBoost
The detail for our parameters of xgboost model is shown as code 4.1, while the other parameters are system default. We take fangshan for instance and predict PM2.5.
 Code 4.1 XGBoost
import xgboost as xgb
xg_train = xgb.DMatrix(train_data, label = train_label)
# param define
 param={
         'booster':'gbtree',
        'objective': 'reg:linear', 
        'gamma':0, # The parameters used to control whether or not to pruning are larger and more conservative.
        'max_depth':4, # The depth of the tree is built, the bigger the easier the overfitting
        'lambda':1, # The larger the parameter, the less likely the model is to overfit.
        'subsample':1, # random get training sample
        'colsample_bytree':0.5, # Random sampling training sample
        'min_child_weight':1, 
        'silent':1 ,
        'eta': 0.1, # learning rate 
        'seed':1500,
        'nthread':8,# Cpu thread number
        'eval_metric': 'rmse'
            }
param['eval_metric'] = ['rmse', 'map']
watchlist = [ (xg_train,'train') ]
#train_model
num_round = 1000
print ('XGBoost Model is Training...')
bst = xgb.train(param, xg_train, num_round, watchlist,verbose_eval=500)
# make prediction
list_1=[]
for i in range((len(X_t)-1)):
        xg_test = xgb.DMatrix(X_t[i:i+2])
        result=bst.predict(xg_test)
        X_t[i+1][2]=X_t[i+1][1]
        X_t[i+1][1]=X_t[i+1][0]
        X_t[i+1][0]=result[0]
        list_1.append(result[0])
        #return result
        if i == 46:
            list_1.append(result[1])            
result_1=pd.DataFrame(list_1)
print(smape(y_test, result_1))
5. Conclusion and Result
After using XGBoost models with slide window method to predict our train dataset, our local test results are show below, and we present in figure 5.1 some visualizations on some stations, which is perform well. Orange color line is our prediction data, the other blue line is the true data. Average PM2.5 of local test consequence of different stations is 0.4235; average PM10 of local test consequence of different stations is 0.30496; average O3 of local test consequence of different stations is 0.4246.
Figure 5.1 Results visualization

