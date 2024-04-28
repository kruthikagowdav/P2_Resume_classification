import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Function to forecast next n days
def forecast_next_n_days(model, data, last_date, n):
    temp_input = list(data)
    lst_output = []
    n_steps = 100

    for i in range(n):
        if len(temp_input) >= n_steps:
            x_input = np.array(temp_input[-n_steps:]).reshape((1, n_steps, 1))
        else:
            # Pad the input with zeros if it's shorter than n_steps
            padding = np.zeros((n_steps - len(temp_input), 1))
            x_input = np.concatenate((padding, np.array(temp_input).reshape((-1, 1))), axis=0)
            x_input = x_input.reshape((1, n_steps, 1))

        yhat = model.predict(x_input)
        temp_input.append(yhat[0, 0])
        lst_output.append(yhat[0, 0])

    return lst_output, pd.date_range(start=last_date, periods=n+1)[1:]

# Load the model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.load_weights("lstm_model.h5")  # Load your model weights here

# Streamlit app
st.title("Crude Oil Price Prediction")

# Sidebar inputs
st.sidebar.title("Model Parameters")
days_input = st.sidebar.number_input("Enter the number of days for forecast", min_value=1, max_value=30, value=1)
file_upload = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if file_upload is not None:
    # Read data from CSV file
    df = pd.read_csv(file_upload)

    # Extract historical data from CSV file
    historical_data = df["Price"].values
    last_date = df["Date"].iloc[-1]  # Assuming the last column is "Date"

    # Perform data preprocessing
    scaler = StandardScaler()
    historical_data = scaler.fit_transform(historical_data.reshape(-1, 1)).reshape(-1)

    # Forecast next n days button
    if st.sidebar.button("Forecast Next {} Days".format(days_input)):
        forecasted_prices, forecasted_dates = forecast_next_n_days(model, historical_data, last_date, days_input)
        forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1)).reshape(-1)
        forecasted_data = pd.DataFrame({"Date": forecasted_dates, "Forecasted Price": forecasted_prices})
        st.write("Forecasted Prices for Next {} Days:".format(days_input))
        st.write(forecasted_data)

    # Display historical data
    st.subheader("Historical Data")
    st.write(df)

    # Display model summary
    st.subheader("Model Summary")
    st.text(model.summary())
else:
    st.sidebar.write("Please upload a CSV file.")
