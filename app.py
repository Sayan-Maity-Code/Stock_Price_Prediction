import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import joblib
import yfinance as yf
import plotly.graph_objects as go
from keras.models import load_model
import matplotlib.pyplot as plt

# MUST be the first Streamlit command
st.set_page_config(page_title="Advanced Stock Prediction System", layout="wide")

# ---------------------------
# Custom CSS for Slate Smoky Dark Theme
# ---------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #1e2130;
            color: #e0e0e0;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2c3142, #1e2130);
            color: #e0e0e0;
        }
        .stButton > button {
            background-color: #3d4663;
            color: #e0e0e0;
            border: 1px solid #5a6482;
        }
        .stButton > button:hover {
            background-color: #5a6482;
            border: 1px solid #8391b4;
        }
        .stSlider .rc-slider-track {
            background-color: #6c7aa1;
        }
        .stSlider .rc-slider-handle {
            border-color: #8391b4;
        }
        .stTextInput > div > div > input {
            background-color: #2c3142;
            color: #e0e0e0;
        }
        .stSelectbox > div > div > div {
            background-color: #2c3142;
            color: #e0e0e0;
        }
        .stMarkdown {
            color: #e0e0e0;
        }
        div[data-testid="stDecoration"] {
            background-image: linear-gradient(90deg, #3d4663, #6c7aa1);
        }
        div[role="alert"] {
            background-color: #2c3142;
            color: #e0e0e0;
        }
        .stPlotlyChart {
            background-color: #2c3142;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Utility Functions
# ---------------------------

def get_enhanced_stock_data(ticker, period='1y'):
    """
    Download and prepare enhanced stock data using yfinance with technical indicators.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
            
        # Calculate technical indicators
        # Moving Averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA100'] = df['Close'].rolling(window=100).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle20'] = df['Close'].rolling(window=20).mean()
        df['BB_std20'] = df['Close'].rolling(window=20).std()
        df['BB_upper20'] = df['BB_middle20'] + 2 * df['BB_std20']
        df['BB_lower20'] = df['BB_middle20'] - 2 * df['BB_std20']
        df['BB_width20'] = (df['BB_upper20'] - df['BB_lower20']) / df['BB_middle20']
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI14'] = 100 - (100 / (1 + rs))
        
        # Volume indicators
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio20'] = df['Volume'] / df['Volume_MA20']
        
        # Volatility
        df['Volatility10'] = df['Close'].rolling(window=10).std() / df['Close'].rolling(window=10).mean()
        df['Volatility21'] = df['Close'].rolling(window=21).std() / df['Close'].rolling(window=21).mean()
        
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def prepare_data_for_advanced_prediction(df, features, seq_length, scaler):
    """
    Prepare the latest sequence of data for model prediction.
    This function checks if all required features exist in the DataFrame.
    """
    # Convert seq_length to integer in case it comes in as something else
    seq_length = int(seq_length)
    
    # Check if all features exist in the DataFrame
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"Missing features in the data: {missing_features}")
        return None

    # Make sure we have enough data
    if len(df) < seq_length:
        st.error(f"Not enough data to create the required sequence. Need {seq_length} rows, but only have {len(df)}.")
        return None
    
    # Select only the required features from the DataFrame
    selected_df = df[features].copy().fillna(method='ffill')
    if selected_df.isna().any().any():
        # If there are still NaN values after forward fill, use backward fill
        selected_df = selected_df.fillna(method='bfill')
    
    # If there are still NaN values, replace them with zeros
    if selected_df.isna().any().any():
        selected_df = selected_df.fillna(0)
    
    # Get the most recent data rows based on seq_length
    recent_data = selected_df.tail(seq_length).values
    
    # Scale the data
    try:
        scaled_data = scaler.transform(recent_data)
    except Exception as e:
        st.error(f"Error scaling data: {str(e)}. Data shape: {recent_data.shape}, Features: {features}")
        return None
    
    # Reshape for model input [1, seq_length, n_features]
    X = np.array([scaled_data])
    return X

def load_models(ticker):
    """
    Load the pre-trained short-term and long-term models along with scalers,
    feature lists, and training history.
    """
    models_dir = f"models/{ticker.upper()}"
    if not os.path.exists(models_dir):
        return None, f"No trained models found for {ticker.upper()}. Please train models first."
    try:
        models_data = {
            'short_term': {},
            'long_term': {},
        }
        
        required_files = [
            "short_term_model.keras", 
            "long_term_model.keras",
            "short_term_feature_scaler.pkl",
            "short_term_target_scaler.pkl",
            "long_term_feature_scaler.pkl",
            "long_term_target_scaler.pkl",
            "short_term_features.txt",
            "long_term_features.txt"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f"{models_dir}/{f}")]
        if missing_files:
            return None, f"Missing required model files: {', '.join(missing_files)}"
        
        models_data['short_term']['model'] = load_model(f"{models_dir}/short_term_model.keras")
        models_data['long_term']['model'] = load_model(f"{models_dir}/long_term_model.keras")
        
        models_data['short_term']['feature_scaler'] = joblib.load(f"{models_dir}/short_term_feature_scaler.pkl")
        models_data['short_term']['target_scaler'] = joblib.load(f"{models_dir}/short_term_target_scaler.pkl")
        models_data['long_term']['feature_scaler'] = joblib.load(f"{models_dir}/long_term_feature_scaler.pkl")
        models_data['long_term']['target_scaler'] = joblib.load(f"{models_dir}/long_term_target_scaler.pkl")
        
        with open(f"{models_dir}/short_term_features.txt", "r") as f:
            models_data['short_term']['features'] = f.read().strip().split(",")
        with open(f"{models_dir}/long_term_features.txt", "r") as f:
            models_data['long_term']['features'] = f.read().strip().split(",")
        
        if os.path.exists(f"{models_dir}/short_term_history.pkl"):
            models_data['short_term']['history'] = joblib.load(f"{models_dir}/short_term_history.pkl")
        else:
            models_data['short_term']['history'] = None
            
        if os.path.exists(f"{models_dir}/long_term_history.pkl"):
            models_data['long_term']['history'] = joblib.load(f"{models_dir}/long_term_history.pkl")
        else:
            models_data['long_term']['history'] = None
        
        if os.path.exists(f"{models_dir}/last_trained.txt"):
            with open(f"{models_dir}/last_trained.txt", "r") as f:
                models_data['last_trained'] = f.read().strip()
        else:
            models_data['last_trained'] = "Unknown"
            
        return models_data, None
    except Exception as e:
        return None, f"Error loading models: {str(e)}"

def make_predictions(ticker, prediction_days=30, model_type='both'):
    """
    Load models and make predictions for the next `prediction_days`.
    Returns a dictionary with prediction results.
    """
    models_dict, error = load_models(ticker)
    if error:
        return None, error

    df = get_enhanced_stock_data(ticker, period='1y')
    if df is None:
        return None, f"Could not fetch recent data for {ticker.upper()}"
    
    prediction_results = {
        'dates': [(df.index[-1] + pd.Timedelta(days=i+1)).date() for i in range(prediction_days)],
        'last_price': df['Close'].iloc[-1],
        'last_date': df.index[-1].date(),
        'historical': df['Close'].tail(30).tolist(),
        'historical_dates': [d.date() for d in df.index[-30:].tolist()]
    }
    
    # Short-term predictions
    if model_type in ['short_term', 'both']:
        try:
            short_term_features = models_dict['short_term']['features']
            
            # Make sure 'Close' is in the DataFrame even if it's not in the feature list
            # This ensures we can work with the Close price for predictions
            if 'Close' not in df.columns:
                return None, "Error: DataFrame is missing 'Close' column"
                
            # Make sure all required features are in the DataFrame
            missing_features = [f for f in short_term_features if f not in df.columns]
            if missing_features:
                return None, f"Error: DataFrame is missing features: {missing_features}"
            
            short_term_X = prepare_data_for_advanced_prediction(
                df, 
                short_term_features, 
                len(short_term_features), 
                models_dict['short_term']['feature_scaler']
            )
            
            if short_term_X is None:
                return None, "Insufficient data for short term prediction."
                
            short_term_preds = []
            last_sequence = df[short_term_features].tail(len(short_term_features)).values
            scaled_last_sequence = models_dict['short_term']['feature_scaler'].transform(last_sequence)
            last_sequence_reshaped = np.array([scaled_last_sequence])
            
            # We will need to update all features for the next prediction
            # Create a mapping of the current values to work with
            feature_indices = {feature: idx for idx, feature in enumerate(short_term_features)}
            
            for i in range(prediction_days):
                pred_scaled = models_dict['short_term']['model'].predict(last_sequence_reshaped, verbose=0)
                pred_value = models_dict['short_term']['target_scaler'].inverse_transform(pred_scaled)[0][0]
                short_term_preds.append(pred_value)
                
                # Create a copy of the last row
                new_row = scaled_last_sequence[-1:].copy()
                
                # Update features that typically change based on Close price
                # Since 'Close' might not be in the feature list, we need to adapt
                # the features that would typically depend on 'Close'
                for feature in feature_indices:
                    # For simplicity, we'll keep most features unchanged
                    # In a real scenario, you'd want to calculate proper values for moving averages, etc.
                    # based on the predicted closing price
                    pass
                    
                # Roll the sequence and replace the last row
                rolled_sequence = np.roll(scaled_last_sequence, -1, axis=0)
                rolled_sequence[-1] = new_row[0]
                scaled_last_sequence = rolled_sequence
                last_sequence_reshaped = np.array([scaled_last_sequence])
                
            prediction_results['short_term'] = short_term_preds
        except Exception as e:
            import traceback
            error_msg = f"Error in short-term prediction: {str(e)}\n{traceback.format_exc()}"
            st.error(error_msg)
            prediction_results['short_term_error'] = error_msg
    
    # Long-term predictions
    if model_type in ['long_term', 'both']:
        try:
            long_term_features = models_dict['long_term']['features']
            
            # Make sure 'Close' is in the DataFrame even if it's not in the feature list
            if 'Close' not in df.columns:
                return None, "Error: DataFrame is missing 'Close' column"
                
            # Make sure all required features are in the DataFrame
            missing_features = [f for f in long_term_features if f not in df.columns]
            if missing_features:
                return None, f"Error: DataFrame is missing features: {missing_features}"
            
            long_term_X = prepare_data_for_advanced_prediction(
                df, 
                long_term_features, 
                len(long_term_features), 
                models_dict['long_term']['feature_scaler']
            )
            
            if long_term_X is None:
                return None, "Insufficient data for long term prediction."
                
            long_term_preds = []
            last_sequence = df[long_term_features].tail(len(long_term_features)).values
            scaled_last_sequence = models_dict['long_term']['feature_scaler'].transform(last_sequence)
            last_sequence_reshaped = np.array([scaled_last_sequence])
            
            # Create a mapping of features to indices
            feature_indices = {feature: idx for idx, feature in enumerate(long_term_features)}
            
            for i in range(prediction_days):
                pred_scaled = models_dict['long_term']['model'].predict(last_sequence_reshaped, verbose=0)
                pred_value = models_dict['long_term']['target_scaler'].inverse_transform(pred_scaled)[0][0]
                long_term_preds.append(pred_value)
                
                # Create a copy of the last row
                new_row = scaled_last_sequence[-1:].copy()
                
                # Keep most features the same, as we can't properly calculate
                # new technical indicators without historical context
                
                # Roll the sequence and replace the last row
                rolled_sequence = np.roll(scaled_last_sequence, -1, axis=0)
                rolled_sequence[-1] = new_row[0]
                scaled_last_sequence = rolled_sequence
                last_sequence_reshaped = np.array([scaled_last_sequence])
                
            prediction_results['long_term'] = long_term_preds
        except Exception as e:
            import traceback
            error_msg = f"Error in long-term prediction: {str(e)}\n{traceback.format_exc()}"
            st.error(error_msg)
            prediction_results['long_term_error'] = error_msg
    
    # Ensemble predictions if both models are used
    if model_type == 'both' and 'short_term' in prediction_results and 'long_term' in prediction_results:
        if 'short_term_error' not in prediction_results and 'long_term_error' not in prediction_results:
            ensemble_preds = []
            for i in range(prediction_days):
                short_weight = max(0.2, 1.0 - i/prediction_days)
                long_weight = 1.0 - short_weight
                ensemble_value = (prediction_results['short_term'][i] * short_weight +
                                 prediction_results['long_term'][i] * long_weight)
                ensemble_preds.append(ensemble_value)
            prediction_results['ensemble'] = ensemble_preds

    return prediction_results, None
# ---------------------------
# Streamlit App Configuration
# ---------------------------

st.title("Advanced Stock Prediction System")
st.markdown("""
    <div style='background-color: #2c3142; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        <h3 style='margin-top: 0; color: #e0e0e0;'>ML-Powered Price Predictions</h3>
        <p style='color: #c0c0c0;'>Using LSTM and GRU networks to forecast stock prices</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for navigation and ticker input
with st.sidebar:
    st.markdown("<h2 style='color: #8391b4;'>Settings</h2>", unsafe_allow_html=True)
    page = st.selectbox("Select Page", ["Home", "Short Term Predictions", "Long Term Predictions"])
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL)", value="AAPL")
    prediction_range = st.slider("Select Prediction Range (Days)", min_value=5, max_value=90, value=30)
    st.markdown("---")
    st.info("This app uses pre-trained models saved in the models/{ticker} directory to predict stock prices.")
    
    if ticker.strip() != "":
        try:
            models_data, error = load_models(ticker)
            if not error and models_data:
                st.markdown(f"<h3 style='color: #8391b4;'>Model Information</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #c0c0c0;'>Last trained: {models_data.get('last_trained', 'Unknown')}</p>", unsafe_allow_html=True)
        except:
            pass

# ---------------------------
# Home Page with Diagram
# ---------------------------
if page == "Home":
    st.markdown("""
        <div style='background-color: #2c3142; padding: 20px; border-radius: 5px;'>
            <h2 style='margin-top: 0; color: #e0e0e0;'>Welcome to the Advanced Stock Prediction System</h2>
            <p style='color: #c0c0c0;'>This application uses deep learning models to predict stock prices over short and long time horizons.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### System Flow Diagram:")
    diagram = """
    digraph {
        rankdir=LR;
        node [shape=box, style=filled, color="#2c3142", fontcolor="#e0e0e0"];
        edge [color="#8391b4"];
        A [label="Data Acquisition\n(yfinance)"];
        B [label="Preprocessing &\nFeature Engineering"];
        C [label="Model Training\n(LSTM/GRU)"];
        D [label="Model Saving\n(.keras, .pkl)"];
        E [label="Model Loading\n(Streamlit)"];
        F [label="Predictions\n(Short/Long Term)"];
        
        A -> B -> C -> D -> E -> F;
    }
    """
    st.graphviz_chart(diagram)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background-color: #2c3142; padding: 15px; border-radius: 5px; height: 100%;'>
                <h3 style='margin-top: 0; color: #e0e0e0;'>How to Use</h3>
                <ul style='color: #c0c0c0;'>
                    <li>Use the sidebar to navigate between pages</li>
                    <li>Enter the desired stock ticker</li>
                    <li>Select the prediction range (number of days)</li>
                    <li>View predictions on the Short Term or Long Term pages</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background-color: #2c3142; padding: 15px; border-radius: 5px; height: 100%;'>
                <h3 style='margin-top: 0; color: #e0e0e0;'>Features</h3>
                <ul style='color: #c0c0c0;'>
                    <li>Short-term predictions using LSTM</li>
                    <li>Long-term predictions using GRU</li>
                    <li>Ensemble predictions combining both models</li>
                    <li>Technical indicators for enhanced accuracy</li>
                    <li>Interactive visualizations</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Sample Visualization")
    sample_img = "https://th.bing.com/th/id/OIP.gJgO3LIWzHw4CCgMvbDIvAHaE8?w=268&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7"
    st.markdown(f"""
        <div style='background-color: #2c3142; padding: 15px; border-radius: 5px; text-align: center;'>
            <img src="{sample_img}" alt="Stock Prediction Dashboard (Demo)" style="max-width: 100%; border-radius: 5px;">
            <p style='color: #c0c0c0; margin-top: 10px;'>Stock Prediction Dashboard (Demo)</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background-color: #2c3142; padding: 15px; border-radius: 5px; margin-top: 20px;'>
            <h3 style='margin-top: 0; color: #e0e0e0;'>Model Architecture</h3>
            <p style='color: #c0c0c0;'>
                Our prediction system employs a dual-model approach:
            </p>
            <ul style='color: #c0c0c0;'>
                <li><strong>Short-term model:</strong> LSTM-based neural network optimized for capturing recent market patterns</li>
                <li><strong>Long-term model:</strong> GRU-based neural network designed to identify long-term trends</li>
                <li><strong>Ensemble approach:</strong> Combines both predictions with dynamic weighting for improved accuracy</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Short Term Predictions Page
# ---------------------------
elif page == "Short Term Predictions":
    st.markdown("""
        <div style='background-color: #2c3142; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
            <h2 style='margin-top: 0; color: #e0e0e0;'>Short Term Stock Predictions</h2>
            <p style='color: #c0c0c0;'>Predict near-future stock prices using our LSTM-based model</p>
        </div>
    """, unsafe_allow_html=True)
    
    if ticker.strip() == "":
        st.error("Please enter a valid stock ticker in the sidebar.")
    else:
        with st.spinner(f"Generating short-term predictions for {ticker.upper()} over the next {prediction_range} days..."):
            pred_results, error = make_predictions(ticker, prediction_days=prediction_range, model_type='short_term')
            
        if error:
            st.error(error)
        elif pred_results is None or 'short_term' not in pred_results:
            st.error("Short term predictions could not be generated. " +
                     pred_results.get('short_term_error', "Unknown error."))
        else:
            st.success(f"Last recorded price on {pred_results['last_date']}: ${pred_results['last_price']:.2f}")
            
            col1, col2, col3 = st.columns(3)
            predicted_price_end = pred_results['short_term'][-1]
            predicted_change = predicted_price_end - pred_results['last_price']
            predicted_pct_change = (predicted_change / pred_results['last_price']) * 100
            
            col1.metric(
                label="Current Price", 
                value=f"${pred_results['last_price']:.2f}"
            )
            col2.metric(
                label=f"Predicted Price ({prediction_range} days)", 
                value=f"${predicted_price_end:.2f}",
                delta=f"{predicted_change:.2f} ({predicted_pct_change:.2f}%)"
            )
            col3.metric(
                label="Prediction Confidence", 
                value="Medium"
            )
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pred_results['historical_dates'], 
                y=pred_results['historical'],
                mode='lines',
                name='Historical',
                line=dict(color='#8391b4', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=pred_results['dates'], 
                y=pred_results['short_term'],
                mode='lines+markers', 
                name='Short Term Predictions',
                line=dict(color='#6c7aa1', width=3, dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=[pred_results['last_date']], 
                y=[pred_results['last_price']],
                mode='markers', 
                name='Last Price',
                marker=dict(size=10, color='#e0e0e0')
            ))
            fig.update_layout(
                title=f"Short Term Predictions for {ticker.upper()}",
                xaxis_title="Date", 
                yaxis_title="Price ($)",
                template="plotly_dark",
                plot_bgcolor='#2c3142',
                paper_bgcolor='#2c3142',
                font=dict(color='#e0e0e0'),
                legend=dict(
                    bgcolor='#1e2130',
                    bordercolor='#8391b4',
                    borderwidth=1
                ),
                hovermode='x unified'
            )
            fig.update_xaxes(gridcolor='#3d4663', showgrid=True)
            fig.update_yaxes(gridcolor='#3d4663', showgrid=True)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
                <div style='background-color: #2c3142; padding: 15px; border-radius: 5px; margin-top: 20px;'>
                    <h3 style='margin-top: 0; color: #e0e0e0;'>Price Forecast Details</h3>
                </div>
            """, unsafe_allow_html=True)
            
            forecast_df = pd.DataFrame({
                'Date': pred_results['dates'],
                'Predicted Price': ["${:.2f}".format(price) for price in pred_results['short_term']],
                'Change from Today': ["{:.2f}%".format(((price - pred_results['last_price']) / pred_results['last_price']) * 100) for price in pred_results['short_term']]
            })
            if len(forecast_df) <= 10:
                display_indices = list(range(len(forecast_df)))
            else:
                display_indices = list(range(0, 5))
                display_indices.append(len(forecast_df) // 2)
                display_indices.extend(list(range(len(forecast_df) - 5, len(forecast_df))))
                display_indices = sorted(list(set(display_indices)))
            
            st.dataframe(forecast_df.iloc[display_indices], use_container_width=True)

# ---------------------------
# Long Term Predictions Page
# ---------------------------
elif page == "Long Term Predictions":
    st.markdown("""
        <div style='background-color: #2c3142; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
            <h2 style='margin-top: 0; color: #e0e0e0;'>Long Term Stock Predictions</h2>
            <p style='color: #c0c0c0;'>Predict extended stock price trends using our GRU-based model</p>
        </div>
    """, unsafe_allow_html=True)
    
    if ticker.strip() == "":
        st.error("Please enter a valid stock ticker in the sidebar.")
    else:
        with st.spinner(f"Generating long-term predictions for {ticker.upper()} over the next {prediction_range} days..."):
            pred_results, error = make_predictions(ticker, prediction_days=prediction_range, model_type='long_term')
            
        if error:
            st.error(error)
        elif pred_results is None or 'long_term' not in pred_results:
            st.error("Long term predictions could not be generated. " +
                     pred_results.get('long_term_error', "Unknown error."))
        else:
            st.success(f"Last recorded price on {pred_results['last_date']}: ${pred_results['last_price']:.2f}")
            
            col1, col2, col3 = st.columns(3)
            predicted_price_end = pred_results['long_term'][-1]
            predicted_change = predicted_price_end - pred_results['last_price']
            predicted_pct_change = (predicted_change / pred_results['last_price']) * 100
            col1.metric(
                label="Current Price", 
                value=f"${pred_results['last_price']:.2f}"
            )
            col2.metric(
                label=f"Predicted Price ({prediction_range} days)", 
                value=f"${predicted_price_end:.2f}",
                delta=f"{predicted_change:.2f} ({predicted_pct_change:.2f}%)"
            )
            
            # Determine market trend based on percentage change
            trend = "Bullish" if predicted_pct_change > 3 else "Bearish" if predicted_pct_change < -3 else "Neutral"
            trend_color = "#4CAF50" if trend == "Bullish" else "#F44336" if trend == "Bearish" else "#FFC107"
            col3.metric(
                label="Market Trend", 
                value=trend,
                delta="",
            )
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pred_results['historical_dates'], 
                y=pred_results['historical'],
                mode='lines',
                name='Historical',
                line=dict(color='#8391b4', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=pred_results['dates'], 
                y=pred_results['long_term'],
                mode='lines+markers', 
                name='Long Term Predictions',
                line=dict(color='#6c7aa1', width=3, dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=[pred_results['last_date']], 
                y=[pred_results['last_price']],
                mode='markers', 
                name='Last Price',
                marker=dict(size=10, color='#e0e0e0')
            ))
            fig.update_layout(
                title=f"Long Term Predictions for {ticker.upper()}",
                xaxis_title="Date", 
                yaxis_title="Price ($)",
                template="plotly_dark",
                plot_bgcolor='#2c3142',
                paper_bgcolor='#2c3142',
                font=dict(color='#e0e0e0'),
                legend=dict(
                    bgcolor='#1e2130',
                    bordercolor='#8391b4',
                    borderwidth=1
                ),
                hovermode='x unified'
            )
            fig.update_xaxes(gridcolor='#3d4663', showgrid=True)
            fig.update_yaxes(gridcolor='#3d4663', showgrid=True)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
                <div style='background-color: #2c3142; padding: 15px; border-radius: 5px; margin-top: 20px;'>
                    <h3 style='margin-top: 0; color: #e0e0e0;'>Price Forecast Details</h3>
                </div>
            """, unsafe_allow_html=True)
            
            forecast_df = pd.DataFrame({
                'Date': pred_results['dates'],
                'Predicted Price': ["${:.2f}".format(price) for price in pred_results['long_term']],
                'Change from Today': ["{:.2f}%".format(((price - pred_results['last_price']) / pred_results['last_price']) * 100) for price in pred_results['long_term']]
            })
            if len(forecast_df) <= 10:
                display_indices = list(range(len(forecast_df)))
            else:
                display_indices = list(range(0, 5))
                display_indices.append(len(forecast_df) // 2)
                display_indices.extend(list(range(len(forecast_df) - 5, len(forecast_df))))
                display_indices = sorted(list(set(display_indices)))
            
            st.dataframe(forecast_df.iloc[display_indices], use_container_width=True)
