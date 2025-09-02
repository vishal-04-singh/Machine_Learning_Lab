import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Dataset.csv')
X = dataset.iloc[:, :-1].values #independent variables
y = dataset.iloc[:, -1].values #dependent variable


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression

regression.fit(X_train, y_train)

y_pred = regression.predict(X_test)
y_pred


plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Linear Regression')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()



from sklearn.metrics import mean_squared_error

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


from sklearn.metrics import mean_absolute_error

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

