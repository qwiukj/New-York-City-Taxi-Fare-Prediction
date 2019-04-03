import pandas as pd
import numpy as np

pd.set_option('display.float_format',lambda  x: '%.3f' %x)
RSEED = 100
import  matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18

import seaborn as sns
palette = sns.color_palette('Paired',10)

#读取指定行数据，把pickup_datetime字段解析成日期格式
data = pd.read_csv('./data/train.csv',nrows=50000,).drop(columns="key")
#剔除缺失值
data = data.dropna()
#查看数据前五行
print(data.head())
#查看数据描述，可以看到各min和max列均有异常值
print(data.describe())

#画出价格分布图
plt.figure(figsize=(10,6))
sns.distplot(data['fare_amount'])
plt.title("Distribution of Fare")
plt.show()

print("价格小于0的异常值个数为%d"%len(data[data['fare_amount'] < 0]))
print("价格等于零的异常值个数为%d"%len(data[data['fare_amount'] == 0]))
print("价格大于100的异常值个数为%d"%len(data[data['fare_amount'] > 100]))
#剔除价格异常的数据
data = data[data['fare_amount'].between(left = 2.5,right = 100)]

#把价格数据连续值分为离散值
data['fare-bin'] = pd.cut(data['fare_amount'],bins = list(range(0,50,5))).astype(str)
#把超过45的数据列修改为[45+],其中loc传入的参数左边为行，右边为列
data.loc[data['fare-bin'] == 'nan','fare-bin'] = '[45+]'
data.loc[data['fare-bin'] == '(5,10]','fare-bin'] = '(05,10]'

#画出价格的区间分布图
data['fare-bin'].value_counts().sort_index().plot.bar(color = 'b',edgecolor = 'k')
plt.title("Fare Binned")
plt.show()
print(data.head())

def ecdf(x):
    x = np.sort(x)
    n = len(x)
    y = np.arange(1,n+1,1)/n
    return x , y

#以百分比的形式画图
x,y = ecdf(data['fare_amount'])
plt.figure(figsize=(8,8))
plt.plot(x,y,'.')
plt.ylabel("percentile")
plt.xlabel('Fare amount')
plt.title('ECDF of Fare Amount')
plt.show()

#画出顾客分布图
data['passenger_count'].value_counts().plot.bar(color = 'b',edgecolor = 'k')
plt.title('passenger counts')
plt.xlabel('number of passenger')
plt.ylabel('count')
plt.show()
#剔除乘客数异常的数值
data = data.loc[data['passenger_count'] < 6]

print('数据还有%d行'%data.shape[0])

#查看2.5%区间和97.5%区间的经纬度数值
for col in ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']:
    print(f'{col}:2.5%={round(np.percentile(data[col],2.5),3)} \t 97.5%={round(np.percentile(data[col],97.5),3)}')

#剔除经纬度异常的数据
data = data.loc[data['pickup_latitude'].between(39,42)]
data = data.loc[data['pickup_longitude'].between(-75,-72)]
data = data.loc[data['dropoff_latitude'].between(39,42)]
data = data.loc[data['dropoff_longitude'].between(-75,-72)]
print("现在的数据量为：%d"%data.shape[0])

fig,axes = plt.subplots(1,2,figsize = (20,8),sharex=True,sharey=True)
axes = axes.flatten()

sns.regplot('pickup_longitude','pickup_latitude',fit_reg = False,data=data,ax=axes[0])
sns.regplot('dropoff_longitude','dropoff_latitude',fit_reg = False,data=data,ax=axes[1])
axes[0].set_title('pickup locations')
axes[1].set_title('pickoff locations')
plt.show()

#计算经纬度相减的绝对值
data['abs_lat_diff'] = (data['dropoff_latitude']-data['pickup_latitude']).abs()
data['abs_lon_diff'] = (data['dropoff_longitude']-data['pickup_longitude']).abs()

sns.lmplot("abs_lat_diff",'abs_lon_diff',fit_reg=False,data = data)
plt.title('Absolute latitude difference vs Absolute longitude difference')
plt.show()
#剔除经纬度相减得到的异常值
no_diff = data[(data['abs_lat_diff']==0)&(data['abs_lon_diff']==0)]
print(no_diff.shape)
#以不同的颜色显示对费用的影响
sns.lmplot('abs_lat_diff','abs_lon_diff',hue = 'fare-bin',size=8,palette=palette,fit_reg=False,data=data)
plt.title('Absolute latitude difference vs Absolute longitude difference')
plt.show()
#更细致的经纬度查看
sns.lmplot('abs_lat_diff','abs_lon_diff',hue = 'fare-bin',size=8,palette=palette,fit_reg=False,data=data)
plt.title('Absolute latitude difference vs Absolute longitude difference')
plt.xlim(-0.01,0.25)
plt.ylim(-0.01,0.25)
plt.show()

def minkowski_distance(x1,x2,y1,y2,p):
    return ((abs(x2-x1)**p)+(abs(y2-y1)**p))**(1/p)
data['manhattan'] = minkowski_distance(data['pickup_longitude'],data['dropoff_longitude'],data['pickup_latitude'],data['dropoff_latitude'],1)
plt.figure(figsize=(12,6))

print(data.groupby('fare-bin'))
#groupby返回值为键值对

#查看各分组对价钱的影响，使用曼哈顿距离1
color_mapping = {fare_bin: palette[i] for i, fare_bin in enumerate(data['fare-bin'].unique())}
data['color'] = data['fare-bin'].map(color_mapping)

for f,group in data.groupby('fare-bin'):
    sns.kdeplot(group['manhattan'],label=f,color = list(group['color'])[0])

plt.xlabel('degrees')
plt.ylabel('density')
plt.title('Manhattan Distance by Fare Amount');
plt.show()

#查看各分组车费均值和距离的关系
data.groupby('fare-bin')['manhattan'].mean().plot.bar(color = 'b')
plt.title('Manhattan Distance by Fare Amount')
plt.show()

#查看各分组对价钱的影响，使用曼哈顿距离2
data['euclidean'] = minkowski_distance(data['pickup_longitude'], data['dropoff_longitude'],
                                       data['pickup_latitude'], data['dropoff_latitude'], 2)

# Calculate distribution by each fare bin
plt.figure(figsize = (12, 6))
for f, grouped in data.groupby('fare-bin'):
    sns.kdeplot(grouped['euclidean'], label = f'{f}', color = list(grouped['color'])[0]);
plt.xlabel('degrees'); plt.ylabel('density')
plt.title('Euclidean Distance by Fare Amount')
plt.show()

#查看各分组车费均值和距离的关系
print(data.groupby('fare-bin')['euclidean'].agg(['mean','count']))
data.groupby('fare-bin')['euclidean'].mean().plot.bar(color='b')
plt.title('Average Euclidean Distance by Fare Bin')
plt.show()

#费用与乘客数量的关系
plt.figure(figsize = (12, 6))
for f, grouped in data.groupby('passenger_count'):
    sns.kdeplot(grouped['fare_amount'], label = f'{f}', color = list(grouped['color'])[0]);
plt.xlabel('fare_amount'); plt.ylabel('density')
plt.title('Distribution of Fare Amount by Number of Passengers')
plt.show()

print(data.groupby('passenger_count')['fare_amount'].agg(['mean','count']))

data.groupby('passenger_count')['fare_amount'].mean().plot.bar(color = 'g')
plt.title('Average Fare by Passenger Count')
plt.show()

#测试集数据处理
test = pd.read_csv('./data/test.csv',parse_dates = ['pickup_datetime'])

test['abs_lat_diff'] = (test['dropoff_latitude'] - test['pickup_latitude']).abs()
test['abs_lon_diff'] = (test['dropoff_longitude'] - test['pickup_longitude']).abs()

# Save the id for submission
test_id = list(test.pop('key'))
test.describe()

test['manhattan'] = minkowski_distance(test['pickup_longitude'], test['dropoff_longitude'],
                                       test['pickup_latitude'], test['dropoff_latitude'], 1)

test['euclidean'] = minkowski_distance(test['pickup_longitude'], test['dropoff_longitude'],
                                       test['pickup_latitude'], test['dropoff_latitude'], 2)
#计算Haversine距离
R = 6378
def haversine_np(lon1,lat1,lon2,lat2):
    #转化为角度
    lon1,lat1,lon2,lat2 = map(np.radians,[lon1,lat1,lon2,lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    km = R*c
    return km

data['haversine'] = haversine_np(data['pickup_longitude'], data['pickup_latitude'],
                         data['dropoff_longitude'], data['dropoff_latitude'])
test['haversine'] = haversine_np(test['pickup_longitude'], test['pickup_latitude'],
                         test['dropoff_longitude'], test['dropoff_latitude'])

#haversine距离和价格的关系
plt.figure(figsize=(10,6))
for f,group in data.groupby('fare-bin'):
    sns.kdeplot(group['haversine'],label=f,color = list(group['color'])[0])
plt.title("Distribution of Haversine Distance by Fare Bin")
plt.show()

print(data.groupby('fare-bin')['haversine'].agg(['mean','count']))

data.groupby('fare-bin')['haversine'].mean().sort_index().plot.bar(color = 'g');
plt.title('Average Haversine Distance by Fare Amount');
plt.ylabel('Mean Haversine Distance')
plt.show()

#测试集haversine距离
sns.kdeplot(test['haversine'])
plt.show()


#日期相关性
data["new_date"] = data['pickup_datetime'].map(lambda x:str(x[5:7]))
print(data.groupby('new_date')['fare_amount'].agg(['mean','count']))
data.groupby('new_date')['fare_amount'].mean().plot.bar(color = 'b')

plt.figure(figsize = (12, 6))
for f, grouped in data.groupby('new_date'):
    sns.kdeplot(grouped['fare_amount'], label = f'{f}', color = list(grouped['color'])[0]);
plt.xlabel('fare_amount'); plt.ylabel('density')
plt.title('Distribution of Fare Amount by date')
plt.show()

data['new_date'].value_counts().plot.bar(color = 'b',edgecolor = 'k')
plt.title('new_date counts')
plt.xlabel('new_date')
plt.ylabel('count')
plt.show()

#相关性
corrs = data.corr()
corrs['fare_amount'].plot.bar(color = 'b')
plt.title('correlation with Fare Amount')
plt.show()

#开始建立模型
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

lr = LinearRegression()
#为了使stratify拆分数据集更加均衡
x_t,x_v,y_t,y_v = train_test_split(data,np.array(data['fare_amount']),stratify=data['fare-bin'],random_state=RSEED,test_size=10000)
lr.fit(x_t[["abs_lat_diff","abs_lon_diff","passenger_count"]],y_t)
print('截距：%d'%round(lr.intercept_,4))
print("abs_laf_diff:%f \n abs_lon_diff:%f \n passenger_count:%f \n"%(round(lr.coef_[0],4),round(lr.coef_[1],4),round(lr.coef_[2],4)))

from sklearn.metrics import mean_squared_error
#rmse均方根误差 mape平均绝对百分误差
def metrics(t_p,v_p,y_t,y_v):
    train_rmse = np.sqrt(mean_squared_error(y_t,t_p))
    valid_rmse = np.sqrt(mean_squared_error(y_v,v_p))

    train_ape = abs((y_t-t_p)/y_t)
    valid_ape = abs((y_v-v_p)/y_v)
    #把无穷大的值置为0
    train_ape[train_ape == np.inf] = 0
    train_ape[train_ape == -np.inf] = 0
    valid_ape[valid_ape == np.inf] = 0
    valid_ape[valid_ape == -np.inf] = 0

    train_mape = 100*np.mean(train_ape)
    valid_mape = 100*np.mean(valid_ape)
    return train_rmse,valid_rmse,train_mape,valid_mape

def eva(model,fea,x_t,x_v,y_t,y_v):
    if fea !=0:
        train_pred = model.predict(x_t[fea])
        valid_pred = model.predict(x_v[fea])
    else:
        train_pred = model.predict(x_t)
        valid_pred = model.predict(x_v)
    train_rmse,valid_rmse,train_mape,valid_mape = metrics(train_pred,valid_pred,y_t,y_v)
    print(f'Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_mape, 2)}')
    print(f'Validation: rmse = {round(valid_rmse, 2)} \t mape = {round(valid_mape, 2)}')

eva(lr,["abs_lat_diff","abs_lon_diff","passenger_count"],x_t,x_v,y_t,y_v)

#随便预测，评价好坏
train_mean = y_t.mean()
train_preds = [train_mean for _ in range(len(y_t))]
valid_preds = [train_mean for _ in range(len(y_v))]
tr,vr,tm,vm = metrics(train_preds,valid_preds,y_t,y_v)
print(f'Baseline Training:   rmse = {round(tr, 2)} \t mape = {round(tm, 2)}')
print(f'Baseline Validation: rmse = {round(vr, 2)} \t mape = {round(vm, 2)}')

preds = lr.predict(test[['abs_lat_diff','abs_lon_diff','passenger_count']])
sub = pd.DataFrame({'key':test_id,'fare_amount':preds})
sub.to_csv('sub_lr_simple.csv',index=False)
#查看预测价格的分布
sns.distplot(sub['fare_amount'])
plt.title("Distribution of Linear Regression Predictions")

#查看预测大于100的异常值
print(test.loc[sub[sub['fare_amount']>100].index])
print(sub[sub['fare_amount']>100])

simple_over_100 = list(sub[sub['fare_amount'] > 100].index)
sub['fare_amount'].describe()

#使用更多的特征
lr.fit(x_t[['haversine','abs_lat_diff','abs_lon_diff','passenger_count']],y_t)

eva(lr,['haversine','abs_lat_diff','abs_lon_diff','passenger_count'],x_t,x_v,y_t,y_v)
print(lr.intercept_)
print(lr.coef_)

#使用日期
data2 = pd.get_dummies(data[['haversine','abs_lat_diff','abs_lon_diff','passenger_count','pickup_datetime','fare_amount','fare-bin']])
x_t,x_v,y_t,y_v = train_test_split(data2,np.array(data2['fare_amount']),random_state=RSEED,test_size=10000,stratify=data2['fare-bin'])
x_t = x_t.drop(columns="fare-bin")
x_v = y_t.drop(columns="fare-bin")
lr.fit(x_t[['haversine','abs_lat_diff','abs_lon_diff','passenger_count'],y_t])

eva(lr,1,x_t,x_v,y_t,y_v)
print('带有日期:')
print(lr.intercept_)
print(lr.coef_)

#热力图
corrs = data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corrs,annot=True,vmin=-1,vmax=1,fmt='.3f')
