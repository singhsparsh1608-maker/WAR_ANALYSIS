import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

equipment = pd.read_csv('/Users/sparshsingh/Desktop/RUSSIA UKRAIN WAR ANALYSIS/DATA/russia_losses_equipment.csv')
print(equipment)
personnel = pd.read_csv('/Users/sparshsingh/Desktop/RUSSIA UKRAIN WAR ANALYSIS/DATA/russia_losses_personnel.csv')
print(personnel)

print(equipment.head())
print(equipment.info())
print(equipment.describe())

print(personnel.head())
print(personnel.info())
print(personnel.describe())

equipment['date'] = pd.to_datetime(equipment['date'])
personnel['date'] = pd.to_datetime(personnel['date'])

print(equipment.isnull().sum())
print(personnel.isnull().sum())

equipment.drop_duplicates(inplace=True)
personnel.drop_duplicates(inplace=True)

equipment.columns = equipment.columns.str.lower()
personnel.columns = personnel.columns.str.lower()

df = pd.merge(
equipment,
personnel,
on='date',
how='inner')
print(df)

df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
print(df)

df['day_number'] = np.arange(len(df))
print(df)

plt.figure(figsize=(14,6))
plt.plot(df['date'], df['personnel'])
plt.title('Personnel Losses Over Time')
plt.xlabel('Date')
plt.ylabel('Personnel Losses')
plt.show()
#Personnel losses fluctuate heavily during major conflict phases.

cols = ['tank', 'aircraft', 'helicopter', 'drone']
losses = df[cols].sum()
losses.plot(kind='bar', figsize=(10,6))
plt.title('Equipment Loss Comparison')
plt.ylabel('Total Losses')
plt.show()
#Tank and drone losses are significantly higher.

monthly = df.groupby('month')['personnel'].mean()
monthly.plot(kind='line', figsize=(10,5))
plt.title('Average Monthly Personnel Losses')
plt.xlabel('Month')
plt.ylabel('Average Personnel Losses')
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(
df.corr(numeric_only=True),
cmap='coolwarm'
)
plt.title('Correlation Heatmap')
plt.show()

df['rolling_avg'] = df['personnel'].rolling(7).mean()
plt.figure(figsize=(14,6))
plt.plot(df['date'], df['personnel'], label='Daily Losses')
plt.plot(df['date'], df['rolling_avg'], label='7-Day Average')
plt.legend()
plt.title('Rolling Average Trend')
plt.show()
#Rolling averages smooth daily fluctuations and reveal long-term war intensity trends.

X = df[['day_number']]
y = df['personnel']

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
print(predictions)

from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("R2 Score:", r2)
print("Mean Absolute Error:", mae)

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, predictions, label="Predicted")
plt.title("Actual vs Predicted Personnel Losses")
plt.legend()
plt.show()

# -------- 400-Day Forecast --------

# Create future day numbers as a DataFrame
future_days = pd.DataFrame({
    "day_number": np.arange(len(df), len(df) + 400)
})

# Predict future values
future_predictions = model.predict(future_days)

# Print predictions
print(future_predictions)

# Plot historical data and forecast
plt.figure(figsize=(12, 6))
plt.plot(df["day_number"], df["personnel"], label="Historical Data")
plt.plot(future_days["day_number"], future_predictions, label="Forecast")
plt.title("400-Day Forecast")
plt.legend()
plt.show()