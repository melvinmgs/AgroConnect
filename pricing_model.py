import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Sample feature set: Weight, Age, Health Score (1-10), Market Trend
def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df[['weight', 'age', 'health_score', 'market_index']]
    y = df['price']
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    # Save the model for the backend to use
    with open('livestock_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved as livestock_model.pkl")

if __name__ == "__main__":
    train_model('livestock_data.csv')
