# 2018KDD-Air-Quality-Prediction

ZHANG Yuning 20550099   WANG Zichun 20547042
##1. Introduction
###Project task:
In this data mining project, the main task is to predict 48 hours of air pollution data (including PM2.5, PM10 and O3) in Beijing on May 1 and May 2, 2018.
###Individual task:
In this project, ZHANG Yuning is responsible for pre-processing of air quality data, and feature extraction for the model. Meanwhile, WANG Zichun is responsible for pre-processing weather data and model training. The rest of other trivial works, like data analysis visualization and write the report is completed by us together.
###Key points and difficulties of the project:
First, Air quality changes very rapidly and there are many points of change. Besides, the original air quality data has much losing values, and some stations even vacancy. In addition, the amount of weather data is large, and the dataset needs to be extracted based on the station coordinates. What’s more, contaminants have complex spatial dependencies. For example, if strong winds blow from a heavily polluted area to a surrounding area, the surrounding air quality will also deteriorate. But if the data from all stations is featured, it will result in a severe overfitting.
##2. Data Preprocess and Split Train and Test Dataset
###2.1 Data Preprocess
Data preprocess is divided into two parts, preprocess air quality data and weather data, after that, we can get entire Dataset to extract features.
###2.1.1	Air Quality Data 
    1. Remove duplicated data. Some of the hour data are duplicated, remove them.
    2. We use groupby function in pandas to process missing value in air data. For each row of missing data (PM2.5, PM10, O3, SO2...), fill with the mean of the corresponding data of other station at the same time. Code is shown below. After this work, if there are still missing values in dataset, drop it directly.

###2.1.2	Weather Data
1. Because the observed weather data consists human errors or equipment faults occur from time to time, so we have to clean it first.
2. And also weather data contains some ‘dirty’ data, like “999”. This part of the data is replaced by data generated randomly within the valid range, which means, if the wind speed is less than 0.5m/s (nearly no wind), the value of the wind direction is 999017.

###2.2	Merging data
The site that needs to be predicted does not match the location of the provided weather data, and we need to manually select the appropriate grid weather data or observed weather data to fill.
 
###2.3	Split Train and Test Dataset
On the one hand, we statistic and visualize PM2.5/PM10/O3 mean changes monthly, as shown in Figure 2.1. Obviously, the amounts of pollutants fluctuate dramatically with the seasons. Therefore, we selected data from 35 sites from February to May 2017 and February to April 2018 as training data.

On the other hand, the data in 35 stations also have much difference, thus we independent forecast for each station. And after the prediction, we corrections the consequence referring to the adjacent station.
At last, the test dataset is the values of each contaminates from the next hour.
##3. Feature Engineering
###3.1. Weather Feature
As what we mentioned before, we split the data based on the 35 stations, and extract features for each of them. When extracting weather features, we first judge the nearest weather data from both on observe weather data and grid weather data, and we use merge function add them to corresponding air data. Then we use the slide-window method to extract features from the previous moment of the current time. Let's take the Dongsi station as an example. If we want to predict the data of 0:00 on May 1st at Dongsi, we will add the wind speed, wind direction, temperature, humidity and pressure of 23rd, 22nd, and 21st on April 30 at Dongsi as features.
###3.2. Air Quality Feature
The process of extracting air quality feature is similar with weather feature. In our model, the effects between individual pollutants were not considered, so we trained three models to predict PM2.5, PM10, and O3, respectively. For instance, we use PM2.5 to illustrate the extraction of pollutant characteristics. Through common sense, we know that PM2.5 at the current moment is highly correlated with PM2.5 at the previous moment. Therefore, we continue to use the previously mentioned windowing method to extract contaminant features. In addition, we added the average, minimum, maximum and range of PM2.5 of the site in previous day. The statistical feature of the data from the previous day was introduced to prevent persistent deviations in the prediction process. 

##4. Models
###4.1 Model Selection
Here we use XGBoost model as our main model. Compared to AdaBoost, XGBoost uses a second-order Taylor expansion to make the algorithm converge to global optimality faster, and the speed and precision of the algorithm are greatly improved. Due to our feature conclude the contaminants index of previous moment t-1, t-2, t-3 (suppose the current time is t), we need to add the results to our feature constantly. 
Figure 4.1 XGBoost
###4.2 Parameters for XGBoost
The detail for our parameters of xgboost model is shown as code 4.1, while the other parameters are system default. We take fangshan for instance and predict PM2.5.
 
##5. Conclusion and Result
After using XGBoost models with slide window method to predict our train dataset, our local test results are show below, and we present in figure 5.1 some visualizations on some stations, which is perform well. Orange color line is our prediction data, the other blue line is the true data. Average PM2.5 of local test consequence of different stations is 0.4235; average PM10 of local test consequence of different stations is 0.30496; average O3 of local test consequence of different stations is 0.4246.
Figure 5.1 Results visualization

