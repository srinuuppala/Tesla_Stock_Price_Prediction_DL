import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Tesla Stock Price Prediction", layout="wide")

st.title("📈 Stock Price Prediction using LSTM")
st.write("Predict Tesla closing price for 1-Day, 5-Day, and 10-Day horizons")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ Prediction Settings")

forecast_days = st.sidebar.selectbox(
    "Select Forecast Horizon",
    ["1-Day", "5-Day", "10-Day"]
)

# -------------------------------
# Load Models
# -------------------------------
if forecast_days == "1-Day":
    model = load_model("Tesla_Stock_Price_Prediction_DL/lstm_1day.h5")
elif forecast_days == "5-Day":
    model = load_model("Tesla_Stock_Price_Prediction_DL/lstm_5day.h5")
else:
    model = load_model("Tesla_Stock_Price_Prediction_DL/lstm_10day.h5")

TIME_STEPS = 60

# -------------------------------
# Prediction Function
# -------------------------------
def predict_and_plot(df, title):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    if 'Close' not in df.columns:
        st.error("❌ Dataset must contain a 'Close' column")
        return

    if len(df) < TIME_STEPS:
        st.error("❌ Dataset must contain at least 60 rows")
        return

    close_data = df[['Close']]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data)

    last_sequence = scaled_data[-TIME_STEPS:]
    X_input = last_sequence.reshape(1, TIME_STEPS, 1)

    prediction = model.predict(X_input)
    prediction = prediction.reshape(-1, 1)
    prediction_actual = scaler.inverse_transform(prediction)

    # -------------------------------
    # Table Output
    # -------------------------------
    st.subheader(f"📊 {title} – {forecast_days} Prediction")

    pred_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(len(prediction_actual))],
        "Predicted Price ($)": prediction_actual.flatten().round(2)
    })

    st.table(pred_df)

    # -------------------------------
    # Plot
    # -------------------------------
    st.subheader("📉 Historical Closing Price")
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['Close'])
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    st.pyplot(plt)

# =====================================================
# 1️⃣ DEFAULT DATASET
# =====================================================
st.header("📁 Default Dataset: TSLA.csv")

df_default = pd.read_csv("TSLA.csv")
st.dataframe(df_default.head())

predict_and_plot(df_default, "Default TSLA.csv")

# =====================================================
# 2️⃣ OPTIONAL UPLOAD CSV
# =====================================================
st.divider()
st.header("📂 Upload Your Stock CSV File (Optional)")

uploaded_file = st.file_uploader(
    "Upload CSV (must contain Date & Close columns)",
    type=["csv"]
)

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(df_uploaded.head())

    predict_and_plot(df_uploaded, "Uploaded CSV")

# =====================================================
# Info
# =====================================================
st.warning("⚠️ This is a predictive model for educational purposes only. Not financial advice.")
st.info("Model Performance: LSTM 1-Day MSE ≈ 519")
st.info("Model Performance: LSTM 5-Day MSE ≈ 1450")

st.info("Model Performance: LSTM 10-Day MSE ≈ 2900")
