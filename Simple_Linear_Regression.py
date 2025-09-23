# House Price Prediction using Area (Simple Linear Regression)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv("Housing.csv")

# Select feature and target
X = df[["area"]]
y = df["price"]

# 1. Scatterplot
plt.figure(figsize=(7,5))
sns.scatterplot(x=X["area"], y=y)
plt.title("House Price vs Area")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.show()

# 2. Correlation coefficient
corr = X["area"].corr(y)
print(f"Correlation between Area and Price: {corr:.2f}")

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# 6. Plot regression line
plt.figure(figsize=(7,5))
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.title("Linear Regression: Area vs Price")
plt.legend()
plt.show()
