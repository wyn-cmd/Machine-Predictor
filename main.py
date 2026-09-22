import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def predict_by_house_type(data):
    """Performs linear regression on housing prices per house type to evaluate MSE and predict prices for the year 2020."""
    for house_type, group in data.groupby('House type'):
        print(f"Predicting for {house_type}...")
        
        X = group[['Year']]
        y = group['Housing Prices']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")

        next_price = model.predict([[2020]])
        print(f"Predicted housing price for {house_type}s in 2020: ${next_price[0]:.2f}\n")

def main():
    try:
        data = pd.read_csv("data.csv")
        predict_by_house_type(data)
    except FileNotFoundError:
        print("Error: 'data.csv' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()