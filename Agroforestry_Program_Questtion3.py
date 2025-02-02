import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Step 1: Load the dataset from the Excel file
data = pd.read_excel(r"C:\Users\phoenix\Desktop\q3_data_for_assignment.xlsx", engine='openpyxl') 

# valide the data is imported by displaying the first few rows of the dataset
print(data.head())

# Step 2: Preprocess the data
# Convert categorical 'Tree species' into numerical values using one-hot encoding so that we can train the model better as numerical numbers help to train better
data = pd.get_dummies(data, columns=['Tree species'], drop_first=True)

# Separate features (X) and target (y)
X = data.drop('TreeDBH_cm', axis=1)  # Features: TreeHeight_foot, TreeCrown_foot, and encoded Tree species
y = data['TreeDBH_cm']  # Target: TreeDBH_cm

# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Step 4: Evaluate the model
# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the predictions vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual DBH')
plt.ylabel('Predicted DBH')
plt.title('Actual vs Predicted DBH')
plt.show()


# saving the trained model to use for new predictions 

joblib.dump(model, 'tree_dbh_model.pkl')


# Load the trained model
model = joblib.load('tree_dbh_model.pkl')

# Example new data => Feeding this manually for testing
new_data = pd.DataFrame({
    'TreeHeight_foot': [50, 60, 70],
    'TreeCrown_foot': [20, 25, 30],
    'Tree species': ['SpeciesA', 'SpeciesB', 'SpeciesC']  # Replace with actual species
})

# One-hot encode the 'Tree species' column (same as during training)
new_data = pd.get_dummies(new_data, columns=['Tree species'], drop_first=True)

# Ensure the new data has the same columns as the training data
missing_cols = set(X_train.columns) - set(new_data.columns)
for col in missing_cols:
    new_data[col] = 0

# Reorder columns to match the training data
new_data = new_data[X_train.columns]

# Predict DBH for the new data
new_predictions = model.predict(new_data)

# Display the predictions
print("Predicted DBH values:", new_predictions)