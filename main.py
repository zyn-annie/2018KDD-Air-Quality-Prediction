import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm
# bj_index contains all the name of 35 plot
# base_path conatain 35 predicted plot.csv 
features = ['PM2.5_1', 'PM2.5_2', 'PM2.5_3',
              'temperature_1', 'temperature_2', 'temperature_3',
              'pressure_1', 'pressure_2', 'pressure_3',
              'humidity_1','humidity_2', 'humidity_3',
              'wind_direction_1', 'wind_direction_2', 'wind_direction_3',
              'wind_speed_1', 'wind_speed_2', 'wind_speed_3']
for each in tqdm(bj_index):
    t_df_aq = pd.read_csv(base_path + each + '_feature.csv')
    t_df_aq['utc_time'] = pd.to_datetime(t_df_aq['utc_time'])
    df_all_1 = t_df_aq.set_index('utc_time')
    df_month__4 = df_all_1['2017-02-01':'2017-04-30']
    df_5_1_5_2 =  df_all_1['2017-05-01':'2017-05-02']
    df_month__4 = df_month__4.dropna(axis=0)
    train_label = np.array(df_month__4['PM2.5'])
    train_data = np.array(df_month__4[features])
    X_t = np.array(df_5_1_5_2[features])
    y_test = np.array(df_5_1_5_2['PM2.5'])
    print ("TRAINING SIZE:",train_data.shape,',',train_label.shape[0])
    xg_train = xgb.DMatrix(train_data, label = train_label)
    # param define
    param={
         'booster':'gbtree',
        'objective': 'reg:linear', 
        'gamma':0,  # The parameters used to control whether or not to pruning are larger and more conservative.
        'max_depth':4, # The depth of the tree is built, the bigger the easier the overfitting
        'lambda':1,  # The larger the parameter, the less likely the model is to overfit.
        'subsample':1, # take random training sample
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
    print(each)
    print(smape(y_test, result))
    plt.plot(y_test)
    plt.plot(result)
    plt.title(each + " PM2.5 2017-05-01_05-02")
    plt.show()
    result=pd.DataFrame(result)
    result.to_csv(base_path + 'result/PM2.5/'+each + '.csv')

    