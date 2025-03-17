# Advanced Stock Prediction System

## Overview

The Advanced Stock Prediction System is a powerful machine learning-powered web application built with Streamlit that forecasts stock prices using LSTM and GRU neural networks. This sophisticated tool analyzes historical market data to provide short-term and long-term price predictions for various stocks.

## Features

- **ML-Powered Price Predictions**: Advanced forecasting using LSTM and GRU networks
- **Multiple Timeframes**: Choose between short-term (6 months - 1 year) and long-term (2+ years) predictions
- **Interactive Visualizations**: Detailed charts showing historical data alongside predictions
- **Technical Indicators**: Incorporates multiple indicators including moving averages, MACD, RSI, and Bollinger Bands
- **Responsive Design**: Modern dark-themed UI optimized for desktop and mobile devices


## Installation

```bash
# Clone the repository
git clone https://github.com/username/Advanced-Stock-Prediction-System.git
cd Advanced-Stock-Prediction-System

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`

## System Requirements

- Python 3.8+
- 4GB RAM (8GB recommended for faster predictions)
- Internet connection (for retrieving stock data via yfinance)

## Dependencies

- streamlit
- pandas
- numpy
- yfinance
- plotly
- tensorflow/keras
- joblib
- matplotlib

## Project Structure

```
Advanced-Stock-Prediction-System/
├── app.py                  # Main Streamlit application
├── models/                 # Pre-trained ML models
│   └── [TICKER]/
│       ├── short_term_model.keras
│       ├── long_term_model.keras
│       └── ...
├── requirements.txt        # Python dependencies
└── README.md               # Project information
```

## How It Works

1. **Data Collection**: Retrieves historical stock data using the yfinance API
2. **Feature Engineering**: Calculates technical indicators like moving averages, MACD, RSI
3. **Model Prediction**: Utilizes pre-trained LSTM/GRU models to forecast future prices
4. **Visualization**: Presents predictions through interactive charts and tables

## Known Limitations

- Predictions are based on historical patterns and technical indicators only
- Market sentiment, news events, and macroeconomic factors are not considered
- Models require periodic retraining to maintain accuracy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for providing free stock data
- [Streamlit](https://streamlit.io/) for the web application framework
- [TensorFlow](https://www.tensorflow.org/) for the machine learning capabilities

## Disclaimer

This software is for educational and research purposes only. Do not use these predictions for actual trading decisions. The developers are not responsible for any financial losses incurred from using this system.
