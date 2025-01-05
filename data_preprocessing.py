
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and clean data
def load_and_clean_data(file_path):
    data = pd.read_excel(file_path)
    data['time'] = pd.to_datetime(data['time'])  # Convert time column to datetime
    X = data.drop(columns=['y', 'time'])
    y = data['y']
    return X, y

# Scale features
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

