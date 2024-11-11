import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
file_path = 'Employee_Salary.xlsx'
data = pd.read_excel(file_path)

# Ensure 'Job Code' is a string type
data['Job Code'] = data['Job Code'].astype(str)

# Selecting features and target
features = data[['Job Code', 'Total Benefits', 'Overtime', 'Age']]
target = data['Salaries']

# Preprocessor setup for column transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Job Code'])
    ], remainder='passthrough')

# Building the pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Training the model
model_pipeline.fit(X_train, y_train)

# Predicting and calculating RMSE
y_pred = model_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse = np.round(rmse, 2)
print(f'root_mean_squared_error = {rmse}')

# Function to predict salaries for new data
def predict_salary(new_data):
    predictions = model_pipeline.predict(new_data)
    predictions = np.round(predictions, 2)  # Round to 2 decimal places
    return predictions

# Example data prediction
example_data = X_test.sample(min(5, len(X_test)))
predicted_salaries = predict_salary(example_data)
print("Example data for prediction:\n", example_data)
print("Predicted salaries:\n", predicted_salaries)
