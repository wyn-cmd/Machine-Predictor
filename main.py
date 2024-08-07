# Version 1.0


# import modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



def predict_by_house_type(data): 
    # group data by house type
    data_grouped = data.groupby('House type')

    for house_type, group in data_grouped:


        print(f"Predicting for {house_type}")
        x = group[['Year']]
        y = group['Housing Prices']

        # split into training and testing sets


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)




        # train the model
        model = LinearRegression()
        model.fit(x_train, y_train)

        # prediction prices
        y_pred = model.predict(x_test)


        # evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error for {house_type}:", mse)

        # predict for the next year (in this case 2020)
        next_year = 2020
        next_price = model.predict([[next_year]])
        print(f"Predicted housing price for {house_type}s in 2020: ${next_price[0]:.2f}")
        print()

# Load the data
data = pd.read_csv("data.csv")

predict_by_house_type(data)

