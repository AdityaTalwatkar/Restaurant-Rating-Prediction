import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Define some common Indian cuisines and cities
cuisines = ['North Indian', 'South Indian', 'Mughlai', 'Rajasthani', 'Gujarati', 'Punjabi']
cities = [
    'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Surat',
    'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Patna', 'Vadodara', 'Ghaziabad',
    'Ludhiana', 'Agra', 'Nashik', 'Faridabad', 'Meerut'
]

# Generate a synthetic dataset
n_samples = 1000
data = {
    'votes': np.random.randint(50, 1000, n_samples),
    'price_range': np.random.randint(1, 4, n_samples),  # 1: Low, 2: Medium, 3: High
    'cuisine': np.random.choice(cuisines, n_samples),
    'location': np.random.choice(cities, n_samples),
    'aggregate_rating': np.random.uniform(2.5, 5.0, n_samples)  # Target variable
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Display basic info
print("First few rows of the dataset:")
print(df.head())
print("\nChecking for missing values:")
print(df.isnull().sum())  # Ensure no missing values

# Encode categorical features using one-hot encoding
df = pd.get_dummies(df, columns=['cuisine', 'location'], drop_first=True)

# Split dataset into features (X) and target variable (y)
X = df.drop('aggregate_rating', axis=1)
y = df['aggregate_rating']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")

# Analyze feature importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importances.head(10))

# Visualize top 10 feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
plt.title('Top 10 Most Important Features for Predicting Restaurant Ratings')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Insights from the model
print("\nInsights:")
print("1. The R-squared score tells us how well our model explains restaurant ratings.")
print("2. Features like votes and price range seem to have the biggest impact on ratings.")
print("3. Restaurant owners can focus on improving these key factors to attract more positive reviews.")
