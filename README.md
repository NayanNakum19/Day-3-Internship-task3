# Day-3-Internship-task3
 🏠 Predict housing prices using simple and multiple linear regression with evaluation metrics on the Housing.csv dataset using scikit-learn.
# 🏠 Housing Price Prediction using Linear Regression

This project demonstrates the use of **Simple**, **Categorical**, and **Multiple Linear Regression** to predict house prices based on the `Housing.csv` dataset.

## 📌 Task Overview

**Objective**: Implement and understand simple & multiple linear regression using:
- `scikit-learn`
- `pandas`
- `matplotlib`

## 📊 Dataset

- **File**: `Housing.csv`
- **Target column**: `price`
##Here is full code of linear and multi linear regration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and preprocess data
df = pd.read_csv("Housing.csv")
df.columns = df.columns.str.strip()  # Clean column names
df = pd.get_dummies(df, drop_first=True)  # Encode categorical features

# 1️⃣ Simple Linear Regression (area → price)
X = df[['area']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
simple_model = LinearRegression()
simple_model.fit(X_train, y_train)
y_pred = simple_model.predict(X_test)

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression: Area vs Price')
plt.legend()
plt.show()

# Evaluation
print("Simple Linear Regression:")
print("Intercept:", simple_model.intercept_)
print("Coefficient (Area):", simple_model.coef_[0])
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 2  Simple Linear Regression (area → price)
X = df.drop("price", axis=1)
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Coefficients
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print(coeff_df)

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.show()



# 2️⃣ Linear Regression with a Categorical Feature (furnishingstatus)
X_cat = df[['furnishingstatus_semi-furnished']]
y_cat = df['price']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42)
cat_model = LinearRegression()
cat_model.fit(X_train_c, y_train_c)
y_pred_c = cat_model.predict(X_test_c)

print("\nLinear Regression with Categorical Feature (furnishingstatus_semi-furnished):")
print("Intercept:", cat_model.intercept_)
print("Coefficient:", cat_model.coef_[0])
print("MAE:", mean_absolute_error(y_test_c, y_pred_c))
print("MSE:", mean_squared_error(y_test_c, y_pred_c))
print("R² Score:", r2_score(y_test_c, y_pred_c))


# 3️⃣ Multiple Linear Regression (All Features)
X_multi = df.drop(columns=['price'])
y_multi = df['price']
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)
y_pred_m = multi_model.predict(X_test_m)

print("\nMultiple Linear Regression Coefficients:")
for feature, coef in zip(X_multi.columns, multi_model.coef_):
    print(f"{feature}: {coef:.2f}")

print("\nMultiple Linear Regression Evaluation:")
print("Intercept:", multi_model.intercept_)
print("MAE:", mean_absolute_error(y_test_m, y_pred_m))
print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("R² Score:", r2_score(y_test_m, y_pred_m))




