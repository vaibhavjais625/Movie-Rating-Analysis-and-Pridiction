import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the movie dataset
df = pd.read_csv("final_movie_data.csv")

# Select the independent variables (year, runtime, votes, label, gross) and dependent variable (rating)
X = df[['year', 'runtime', 'votes', 'label', 'gross(in $)']]
y = df['rating']

# Remove any rows with missing values in selected columns to avoid dtype issues
X = X.dropna()
y = y.loc[X.index]

# Clean and preprocess data types
X['gross(in $)'] = pd.to_numeric(X['gross(in $)'], errors='coerce')
X['gross(in $)'].fillna(X['gross(in $)'].median(), inplace=True)

X['year'] = pd.to_numeric(X['year'], errors='coerce')
X['year'].fillna(X['year'].median(), inplace=True)

# Perform one-hot encoding on the categorical variables
X_encoded = X.copy()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a decision tree regression model
regression_model = DecisionTreeRegressor(random_state=42)

# Train the model
regression_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regression_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)

# Save the trained model as a pickle file
joblib.dump(regression_model, "movie_rating_model.pkl")
print("Model saved as movie_rating_model.pkl")

# Perform accuracy evaluation using the training set
accuracy = regression_model.score(X_train, y_train)
print("Training Accuracy:", accuracy)
