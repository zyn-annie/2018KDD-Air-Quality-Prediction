Readme
The sequence of file be ordered:
air_data_process.py-------process original air quality data
weather_data_clean.py------process grid and observe weather data and merge with the processed air data
extract_feature.py-------process data that contain both weather and air and divided by 35 stations
main.py-------use xgboost to predict