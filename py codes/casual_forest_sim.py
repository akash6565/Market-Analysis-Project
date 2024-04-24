import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Set random seed
np.random.seed(1234)

# Generate data
n = 10000
female = np.random.choice([0, 1], n)
age = np.random.uniform(20, 80, n)
z1 = np.random.choice([0, 1], n)

group1 = age < 30
group2 = (age >= 40) & (age < 50) & (female == 1)
group3 = (age > 70) & (female == 0) & (z1 == 1)
group = group1 + group2 * 2 + group3 * 3

y0 = 100 - 10 * female + 10 * group1 + np.random.normal(0, 1, n)
tau = 10 * group1 + 10 * group2 - 10 * group3 + np.random.normal(0, 1, n)
y1 = y0 + tau

w = np.random.choice([0, 1], n)
y = np.where(w == 0, y0, y1)

dat = pd.DataFrame({'y': y, 'w': w, 'female': female, 'age': age, 'z1': z1, 'group': group, 'tau': tau, 'y0': y0, 'y1': y1})

X = dat[['female', 'age', 'z1']].values

cf = RandomForestRegressor(n_estimators=100)
cf.fit(X, dat['y'])

pred = cf.predict(X)
dat['tau_hat'] = pred

group_summary = dat.groupby('group').agg({'tau_hat': 'mean', 'tau': 'mean', 'y': 'mean'}).reset_index()
print(group_summary)
