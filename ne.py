# House Price Prediction (Beginner Version)
# -----------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Load dataset
df = pd.read_csv("submission.csv")

# 2. Univariate EDA - Distribution of SalePrice
plt.figure(figsize=(8,5))
sns.histplot(df["SalePrice"], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()

# Boxplot for outliers
plt.figure(figsize=(6,4))
sns.boxplot(x=df["SalePrice"])
plt.title("Boxplot of SalePrice")
plt.show()

# 3. Statistical Summary
mean_price = df["SalePrice"].mean()
median_price = df["SalePrice"].median()
skewness = skew(df["SalePrice"])
kurt = kurtosis(df["SalePrice"])

print("📊 Statistical Summary of SalePrice")
print(f"Mean: {mean_price:.2f}")
print(f"Median: {median_price:.2f}")
print(f"Skewness: {skewness:.2f}")
print(f"Kurtosis: {kurt:.2f}")

# 4. Dummy Regression Model (Id → SalePrice)
X = df[["Id"]]   # feature (Id, dummy)
y = df["SalePrice"]   # target

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# 5. Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n🤖 Model Performance (Dummy LR with Id)")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
