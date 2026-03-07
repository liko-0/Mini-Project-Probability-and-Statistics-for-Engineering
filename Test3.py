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
corr = df[(df['date'].dt.year >= 2010) & (df['date'].dt.year <= 2023)].copy()

# +++ จุดที่ 1: ดึงราคาทองจริงของเมื่อวานเก็บไว้ก่อน เพื่อใช้แปลงค่ากลับในตอนท้าย +++
df['prev_gold_high'] = df['gold high'].shift(1)

# +++ จุดที่ 2: คำนวณ Log Return ให้ตัวแปรราคาทั้ง 4 ตัว +++
cols_to_log = ['oil high', 'nasdaq high', 'silver low', 'gold high']
for col in cols_to_log:
    df[col] = np.log(df[col] / df[col].shift(1))

# Shift ข้อมูลเพื่อให้ใช้ข้อมูลตัวแปรอิสระของ "เมื่อวาน" 
df['oil high'] = df['oil high'].shift(1)
df['nasdaq high'] = df['nasdaq high'].shift(1)
df['silver low'] = df['silver low'].shift(1)
df = df.dropna()

df.corr()['gold high']
print("\nCorrelations with Gold High (Sorted):")
print(df.corr()['gold high'].sort_values(ascending=False))

train = df[(df['date'].dt.year >= 2010) & (df['date'].dt.year <= 2023)].copy()
test = df[df['date'].dt.year == 2024].copy()

# +++ นำ prev_gold_high ของชุด Test แยกออกมาเก็บไว้เป็น Array ก่อนลบทิ้ง +++
prev_gold_test = test['prev_gold_high'].to_numpy()

# ลบคอลัมน์ที่ไม่ใช้ออกจากชุด Train / Test
del train['date']
del train['prev_gold_high']
del test['date']
del test['prev_gold_high']

df = df.dropna()
train_np = train.to_numpy()
X_train, Y_train = train_np[:, :3], train_np[:, -1]
test_np = test.to_numpy()
X_test, Y_test = test_np[:, :3], test_np[:, -1]

# สร้าง Model ด้วย Sklearn (เพื่อให้โค้ดคุณรันได้เหมือนเดิม)
sklearn_model = LinearRegression().fit(X_train, Y_train)
sklearn_Y_predictions = sklearn_model.predict(X_train)
# print(mean_absolute_error(sklearn_Y_predictions, Y_train), mean_squared_error(sklearn_Y_predictions, Y_train))

# สร้างฟังก์ชันคำนวณ OLS ด้วยตัวเอง (ตามโค้ดเดิมของคุณ)
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
beta_0 = best_model[0]
print(f"Intercept (beta_0): {beta_0}")

# +++ จุดที่ 3: ทำนายค่าของปี 2024 และแปลง Log Return กลับเป็น "ราคาจริง" +++
# 1. ทำนายออกมาเป็น Log Return ก่อน
test_predictions_log = get_predictions(best_model, X_test)

# 2. แปลงค่า Log Return ให้กลับเป็นราคาทำนาย (Predicted Price) 
# สูตร: ราคาวันนี้ = ราคาเมื่อวาน * e^(log_return)
test_predictions_2024_price = prev_gold_test * np.exp(test_predictions_log)

# 3. แปลงค่า Y_test (ซึ่งตอนนี้เป็น Log Return) ให้กลับเป็นราคาจริง (Actual Price) เพื่อนำมาหา Error
actual_gold_high_2024 = prev_gold_test * np.exp(Y_test)

# ประเมินผลแบบ MSPE ด้วย "ราคาจริง" ไม่ใช่เปอร์เซ็นต์ Log Return
percentage_errors = (actual_gold_high_2024 - test_predictions_2024_price) / actual_gold_high_2024
mspe_2024 = np.mean(np.square(percentage_errors))

print(f"Mean Squared Percentage Error (MSPE) for 2024: {mspe_2024}")
print(f"MSPE as Percentage: {mspe_2024 * 100}%")

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