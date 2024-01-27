
# Machine Learning Stock Market Predictor

## Overview
The Stock Market Predictor is a web-based application designed to forecast stock prices using a pre-trained machine learning model. It provides an interactive interface for users to visualize historical stock data, moving averages, and predicted prices. Built with Streamlit, the application offers an intuitive way for users to enter stock symbols and analyze stock performance over time.

## Image Demo
This is an image of what the website outputs: [Demo Image](/Image-Demo)

## Features
- **Stock Symbol Input**: Users can input any stock symbol to fetch historical data and predictions.
- **Historical Stock Data Visualization**: Displays the stock's historical closing prices.
- **Moving Averages Visualization**: Plots the 50-day, 100-day, and 200-day moving averages alongside the closing prices to show trends.
- **Price Predictions**: Showcases the model's predicted stock prices against the actual historical prices.
- **Interactive Interface**: Built with Streamlit, offering a user-friendly and interactive web interface.

## Installation

### Prerequisites
Before running the application, ensure you have the following installed:
- Python 3.6 or later
- Pip (Python package installer)

### Libraries
Install the required Python libraries using pip:

```bash
pip install numpy pandas yfinance keras streamlit matplotlib scikit-learn
```

### Clone the Repository
Clone this repository to your local machine:

```bash
git clone https://github.com/ofarrag9/ML-Stock_Predictor.git
cd ML-Stock_Predictor
```

## Usage

1. Navigate to the application directory:

```bash
cd path/to/stock-market-predictor
```

2. Run the Streamlit application:

```bash
streamlit run app.py
```

3. Open your web browser and go to the address provided by Streamlit, typically `http://localhost:8501`.

4. Enter a stock symbol and analyze the historical data, moving averages, and predicted prices.

## Model Information
The stock price predictions are made using a pre-trained neural network model. The model was trained on historical stock data, incorporating features like closing prices and moving averages to forecast future stock prices.

