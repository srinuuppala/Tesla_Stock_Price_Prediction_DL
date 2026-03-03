# 📈 Tesla Stock Price Prediction using LSTM

## 🔹 Overview

This project implements a Deep Learning-based time-series forecasting model to predict Tesla stock closing prices using LSTM (Long Short-Term Memory) networks.

The application predicts stock prices for:

- 1-Day Forecast  
- 5-Day Forecast  
- 10-Day Forecast  

The trained model is deployed using Streamlit as an interactive web application.

---

## 🔹 Problem Statement

Stock market prices are highly volatile and sequential in nature. Traditional machine learning models struggle to capture long-term temporal dependencies.

This project uses LSTM networks, which are well-suited for time-series forecasting, to predict Tesla's future closing prices.

---

## 🔹 Dataset

Dataset: Tesla historical stock prices (TSLA.csv)

Features:
- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

Target Variable:
- Close Price

---

## 🔹 Project Workflow

1. Data Loading  
2. Data Cleaning  
3. Exploratory Data Analysis (EDA)  
4. Feature Scaling (MinMaxScaler)  
5. Time-Series Sequence Creation (60-day window)  
6. Model Training (LSTM)  
7. Model Evaluation (Mean Squared Error)  
8. Deployment using Streamlit  

---

## 🔹 Model Architecture

- Input: Past 60 days closing prices  
- LSTM Layer  
- Dense Output Layer  
- Loss Function: Mean Squared Error (MSE)  
- Optimizer: Adam  

---

## 🔹 Model Performance

Forecast Horizon | MSE
----------------|------
1-Day | ~519  
5-Day | ~1450  
10-Day | ~2900  

Observation: Prediction error increases as forecast horizon increases due to accumulated uncertainty.

---

## 🔹 Streamlit Application Features

- Sidebar to select forecast horizon (1-Day, 5-Day, 10-Day)  
- Default Tesla dataset prediction  
- Optional CSV upload support  
- Tabular prediction output  
- Historical price visualization  
- Clean and interactive UI  

---

## 🔹 How to Run Locally

1. Install Dependencies:

pip install -r requirements.txt

2. Run Streamlit App:

streamlit run app.py

---

## 🔹 Project Structure

tesla_stock_lstm/
│
├── app.py  
├── TSLA.csv  
├── lstm_1day.h5  
├── lstm_5day.h5  
├── lstm_10day.h5  
├── requirements.txt  
├── runtime.txt  
└── README.md  

---

## 🔹 Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  
- TensorFlow (LSTM)  
- Streamlit  

---

## 🔹 Future Improvements

- Add technical indicators (RSI, MACD)  
- Integrate news sentiment analysis  
- Implement GRU or Transformer models  
- Add real-time stock API integration  

---

## ⚠️ Disclaimer

This project is for educational and research purposes only.  
It should not be considered financial advice.

---

## 👤 Author

UPPALA VENKATA SATYA SRINIVAS 
Deep Learning & Data Science Enthusiast  

---
