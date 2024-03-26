"""
AirLine Dynamic Profit Model

Problem Objective:
To predict the optimal pricing for airline tickets to maximize revenue, utilizing flight data and demand patterns.

Step 1: Creating a Synthetic Dataset
Generate synthetic dataset simulating real-world scenario for airline pricing including factors like date, time, origin, destination, price, seats available, demand level, and more.

Step 2: Building the Model
Use machine learning to build a model that predicts the optimal ticket prices based on the dataset.

Step 3: Saving the Model
Save the trained model for future use in making predictions.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from joblib import dump

# Generating the synthetic dataset remains the same

def generate_synthetic_dataset(num_records=10000):
    np.random.seed(42)  # For reproducibility

    # Generate dates and times
    start_date = pd.to_datetime('2024-01-01')
    hours_increment = np.random.randint(1, 24, num_records)
    dates = start_date + pd.to_timedelta(hours_increment.cumsum(), unit='h')

    # Generate other features
    day_of_week = dates.dayofweek
    time_of_day = ['Morning' if 5 <= dt.hour < 12 else 'Afternoon' if 12 <= dt.hour < 18 else 'Evening' for dt in dates]
    duration = np.random.uniform(1, 12, num_records)
    origins = ['Origin_' + str(np.random.randint(1, 101)) for _ in range(num_records)]
    destinations = ['Destination_' + str(np.random.randint(1, 101)) for _ in range(num_records)]
    base_price = np.random.uniform(100, 1000, num_records)
    seats_available = np.random.randint(50, 300, num_records)
    demand_factor = np.random.uniform(0.5, 1.5, num_records)
    holiday_effect = [1.5 if dt.month == 12 or dt.month == 7 else 1.0 for dt in dates]
    historical_load_factor = np.random.uniform(0.6, 0.95, num_records)
    competitor_price = base_price * np.random.uniform(0.8, 1.2, num_records)

    # Adjust price based on demand, holiday effect
    prices = base_price * demand_factor * holiday_effect

    # Assemble the dataset
    dataset = pd.DataFrame({
        'Flight ID': range(1, num_records + 1),
        'Date': dates,
        'Day of Week': day_of_week,
        'Time of Day': time_of_day,
        'Duration': duration,
        'Origin': origins,
        'Destination': destinations,
        'Price': prices,
        'Seats Available': seats_available,
        'Historical Load Factor': historical_load_factor,
        'Competitor Price': competitor_price
    })

    return dataset

synthetic_dataset = generate_synthetic_dataset()

X = synthetic_dataset.drop(['Flight ID', 'Date', 'Price'], axis=1)
y = synthetic_dataset['Price']

# Feature engineering and preprocessing
# Assuming categorical_features remains unchanged
categorical_features = ['Day of Week', 'Time of Day', 'Origin', 'Destination']
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(transformers=[('cat', one_hot_encoder, categorical_features)], remainder='passthrough')

# Switching to a more complex model: RandomForestRegressor
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42))])

# Hyperparameter tuning setup
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
}

# Setup GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model using GridSearchCV to find the best model
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
accuracy_percentage = r2 * 100
print(f"Accuracy: {accuracy_percentage:.2f}%")


os.makedirs('./model', exist_ok=True)
dump(best_model, './model/ticket_price_predictor_rf.joblib')
