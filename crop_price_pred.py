
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

# Train a Ridge Regression model
model_ridge = Ridge()
model_ridge.fit(X_train, y_train)
score_ridge = model_ridge.score(X_train, y_train)

# Train a Random Forest Regression model
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
score_rf = model_rf.score(X_train, y_train)

# Make predictions on the test set for Decision Tree
y_pred_dt = model_dt.predict(X_test)

# Make predictions on the test set for Ridge Regression
y_pred_ridge = model_ridge.predict(X_test)

# Make predictions on the test set for Random Forest
y_pred_rf = model_rf.predict(X_test)

# Evaluate the models
mse_dt = mean_squared_error(y_test, y_pred_dt)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_rf = mean_squared_error(y_test, y_pred_rf)

#Printing the results
print(f'Mean Squared Error (Decision Tree): {mse_dt}')
print(f'Mean Squared Error (Ridge Regression): {mse_ridge}')
print(f'Mean Squared Error (Random Forest): {mse_rf}')

print(f'Decision Tree Model Score: {score_dt}')
print(f'Ridge Regression Model Score: {score_ridge}')
print(f'Random Forest Model Score: {score_rf}')

# Plotting the actual vs predicted values for Decision Tree
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_dt, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.title('Actual vs Predicted values for Decision Tree')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.legend()
plt.savefig('decision_tree_plot.png')  # Save the plot as an image file
plt.close()

# Plotting the actual vs predicted values for Ridge Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_ridge, color='green', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.title('Actual vs Predicted values for Ridge Regression')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.legend()
plt.savefig('ridge_regression_plot.png')  # Save the plot as an image file
plt.close()

# Plotting the actual vs predicted values for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, color='orange', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.title('Actual vs Predicted values for Random Forest')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.legend()
plt.savefig('random_forest_plot.png')  # Save the plot as an image file
plt.close()


# Plotting the comparison of actual vs predicted values for all models
plt.figure(figsize=(10, 6))

# Plot actual vs predicted values for Decision Tree
plt.scatter(y_test, y_pred_dt, color='blue', label='Decision Tree Predictions')

# Plot actual vs predicted values for Ridge Regression
plt.scatter(y_test, y_pred_ridge, color='green', label='Ridge Regression Predictions')

# Plot actual vs predicted values for Random Forest
plt.scatter(y_test, y_pred_rf, color='orange', label='Random Forest Predictions')

# Plot the diagonal red line representing perfect predictions
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')

plt.title('Comparison of Actual vs Predicted values for all models')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.legend()
plt.savefig('comparison_plot.png')  # Save the plot as an image file
plt.close()

# Define the MSE values for each algorithm
mse_values = [mse_dt, mse_ridge, mse_rf]

# Define the labels for each algorithm
labels = ['Decision Tree', 'Ridge Regression', 'Random Forest']

# Plot the comparison of Mean Squared Error (MSE) for all models
plt.figure(figsize=(10, 6))
plt.plot(labels, mse_values, marker='o', color='b', linestyle='-')
plt.title('Comparison of Mean Squared Error (MSE) for all models')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('mse_comparison.png')  # Save the plot as an image file
plt.close()

# Define the model scores for each algorithm
scores = [score_dt, score_ridge, score_rf]

# Define the labels for each algorithm
labels = ['Decision Tree', 'Ridge Regression', 'Random Forest']

# Plot the comparison of model scores for all models using a line graph
plt.figure(figsize=(10, 6))
plt.plot(labels, scores, marker='o', color='b', linestyle='-')
plt.title('Comparison of Model Scores for all models')
plt.xlabel('Models')
plt.ylabel('Model Score')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_scores_line_graph.png')  # Save the plot as an image file
plt.close()

# Define the model scores for each algorithm
scores = [score_dt, score_ridge, score_rf]

# Define the labels for each algorithm
labels = ['Decision Tree', 'Ridge Regression', 'Random Forest']

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))

# Set bar width
bar_width = 0.5

# Create bars with centered positions
x = np.arange(len(labels))
bars = ax.bar(x, scores, width=bar_width, color='skyblue', edgecolor='black')

# Add labels and title
ax.set_title('Comparison of Model Scores for all Models', fontsize=16)
ax.set_xlabel('Models', fontsize=14)
ax.set_ylabel('Model Score', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0, 1)  # Adjust the y-axis limits if necessary
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add data labels above the bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=12)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show plot
plt.tight_layout()
plt.savefig('model_scores_bar_graph.png')  # Save the plot as an image file
plt.close()