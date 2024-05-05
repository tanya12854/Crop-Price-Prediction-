# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('dataset__crop__price.csv')

# Preprocess the data
df['Price Date'] = pd.to_datetime(df['Price Date'], format='%d-%b-%y', errors='coerce')
df = df.dropna(subset=['Price Date'])
df['Price Date'] = pd.to_datetime(df["Price Date"], format='%d-%b-%y')
df['Day'] = df['Price Date'].dt.day
df['Month'] = df['Price Date'].dt.month
df['Year'] = df['Price Date'].dt.year

# Select features and target variable
features = ['Day', 'Month', 'Year']
target = 'Modal Price (Rs./Quintal)'
X = df[features]
y = df[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert X_train to DataFrame
X_train = pd.DataFrame(X_train)
y_train = y_train.values.ravel()

# Train a decision tree regression model
model_dt = DecisionTreeRegressor()
model_dt.fit(X_train, y_train)
score_dt = model_dt.score(X_train, y_train)

# Train a Support Vector Regression (SVR) model
model_svr = SVR()
model_svr.fit(X_train, y_train)
score_svr = model_svr.score(X_train, y_train)

# Train a Random Forest Regression model
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
score_rf = model_rf.score(X_train, y_train)

# Make predictions on the test set for Decision Tree
y_pred_dt = model_dt.predict(X_test)

# Make predictions on the test set for SVR
y_pred_svr = model_svr.predict(X_test)

# Make predictions on the test set for Random Forest
y_pred_rf = model_rf.predict(X_test)

# Evaluate the models
mse_dt = mean_squared_error(y_test, y_pred_dt)
mse_svr = mean_squared_error(y_test, y_pred_svr)
mse_rf = mean_squared_error(y_test, y_pred_rf)

print(f'Mean Squared Error (Decision Tree): {mse_dt}')
print(f'Mean Squared Error (SVR): {mse_svr}')
print(f'Mean Squared Error (Random Forest): {mse_rf}')

print(f'Decision Tree Model Score: {score_dt}')
print(f'SVR Model Score: {score_svr}')
print(f'Random Forest Model Score: {score_rf}')

