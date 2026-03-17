# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 2. Load Dataset
# Make sure your CSV file is in the same folder
df = pd.read_csv("salary_data.csv")

print("Dataset Preview:")
print(df.head())

# 3. Handle Categorical Data (Encoding)
df = pd.get_dummies(df, drop_first=True)

# 4. Define Features and Target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# 9. Save Model
with open("salary_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as salary_model.pkl")

# 10. Test with New Input
# Example input (must match training features)
sample_input = X.iloc[0:1]  # taking first row as sample

predicted_salary = model.predict(sample_input)
print("\nSample Prediction:", predicted_salary[0])

# 11. Save Predictions
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results.to_csv("predictions.csv", index=False)
print("\nPredictions saved to predictions.csv")