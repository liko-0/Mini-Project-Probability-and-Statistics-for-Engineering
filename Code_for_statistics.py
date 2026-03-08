import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy.linalg import inv

df = pd.read_csv('financial_regression.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

df['CPI'] = df['CPI'].ffill()
df['us_rates_%'] = df['us_rates_%'].ffill()
df['GDP'] = df['GDP'].ffill()
df = df.dropna(subset=['gold volume'])

df = df[['date', 'oil high', 'nasdaq high', 'silver low', 'gold high']]

df['prev_gold_high'] = df['gold high'].shift(1)

cols_to_log = ['oil high', 'nasdaq high', 'silver low', 'gold high']
for col in cols_to_log:
    df[col] = np.log(df[col] / df[col].shift(1))

df[['oil high', 'nasdaq high', 'silver low']] = df[['oil high', 'nasdaq high', 'silver low']].shift(1)

df = df.dropna().reset_index(drop=True)

train = df[(df['date'].dt.year >= 2010) & (df['date'].dt.year <= 2023)].copy()
test = df[df['date'].dt.year == 2024].copy()

prev_gold_test = test['prev_gold_high'].to_numpy()

del train['date']
del train['prev_gold_high']
del test['date']
del test['prev_gold_high']

train_np = train.to_numpy()
X_train, Y_train = train_np[:, :3], train_np[:, -1]
test_np = test.to_numpy()
X_test, Y_test = test_np[:, :3], test_np[:, -1]

def get_predictions(model, X):
  (n, p_minus_one) = X.shape
  p = p_minus_one + 1
  new_X = np.ones(shape=(n, p))
  new_X[:, 1:] = X
  return np.dot(new_X, model)

def get_best_model(X, Y):
  (n, p_minus_one) = X.shape
  p = p_minus_one + 1
  new_X = np.ones(shape=(n, p))
  new_X[:, 1:] = X
  return np.dot(np.dot(inv(np.dot(new_X.T, new_X)), new_X.T), Y)

best_model = get_best_model(X_train, Y_train)
beta0 = best_model[0]
beta1 = best_model[1]
beta2 = best_model[2]
beta3 = best_model[3]
print(f"Intercept (beta0): {beta0:.6f}")
print(f"Oil high (beta1): {beta1:.6f}")
print(f"Nasdaq high (beta2): {beta2:.6f}")
print(f"Silver low (beta3): {beta3:.6f}")

test_predictions_log = get_predictions(best_model, X_test)

test_predictions_2024_price = prev_gold_test * np.exp(test_predictions_log)

actual_gold_high_2024 = prev_gold_test * np.exp(Y_test)

percentage_errors = (actual_gold_high_2024 - test_predictions_2024_price) / actual_gold_high_2024
mspe_2024 = np.mean(np.square(percentage_errors))

print(f"Mean Squared Percentage Error (MSPE) for 2024: {mspe_2024:.10f}")

import matplotlib.pyplot as plt

test_dates = df[df['date'].dt.year == 2024]['date']
plt.figure(figsize=(12, 6))
plt.plot(test_dates, actual_gold_high_2024, label='Actual Gold High', color='gold', linewidth=2)
plt.plot(test_dates, test_predictions_2024_price, label='Predicted Gold High', color='midnightblue', linestyle='--', alpha=0.8)
plt.title('Gold Price Prediction 2024')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
plt.show()