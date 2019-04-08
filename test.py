def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)** p) ) ** (1 / p)
print(minkowski_distance(0,3,0,4,2))

import numpy as np
lon1,lat1,lon2,lat2 = map(np.radians,[1,2,3,180])
print(lon1,lat1,lon2,lat2)

import pandas as pd

attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Days_in_month', 'is_leap_year']

data = pd.read_csv('./data/train.csv',nrows=500,parse_dates=['pickup_datetime'])
data['ss'] = data['pickup_datetime']
for n in attr:
    data[ n] = getattr(data['ss'].dt,n.lower())
print(data[0:5])
m = 1
print([m for _ in range(5)])