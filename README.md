# Marketanalyzer

📈 Stock Price Forecasting App
A comprehensive, interactive web application for stock price prediction using Facebook Prophet and TensorFlow. Built with Streamlit for an intuitive user experience.

🎯 Features
📊 Data Management

Yahoo Finance Integration: Automatically download stock data for any ticker symbol
CSV Upload Support: Import your own historical stock data
Auto-Update: Fetches the latest available data by default
Session State: Data persists across interactions - no need to reload

🔮 Advanced Forecasting
Multiple Timeframes: 20 days, 90 days, and 1-year predictions
Confidence Intervals: Upper and lower bounds for risk assessment
Interactive Visualization: Dynamic charts that update based on selected timeframe

📈 Comprehensive Analysis
Stock Statistics: Detailed price statistics and historical charts
Key Predictions: Quick overview of short, medium, and long-term forecasts
Risk Assessment: Color-coded analysis (High Growth, Moderate, Stable, Decline)
Detailed Forecast Table: Day-by-day predictions with confidence bounds

🎨 User Experience
Responsive Design: Works on desktop and mobile devices
Intuitive Interface: Easy-to-use tabs and interactive elements
Real-time Updates: Charts and metrics update instantly when changing timeframes
Professional Visualizations: Clean, publication-ready charts and graphs

💻 Usage Guide
Getting Started

Select Data Source: Choose between "Download" (Yahoo Finance) or "Import" (CSV file)

Option 1: Download from Yahoo Finance

Enter a stock symbol (e.g., AAPL, GOOGL, TSLA, MSFT)
Select date range (defaults to 2021 to today)
Check "Get latest data" for current information
Click "Download"

Option 2: Import CSV File

Upload a CSV file with columns: Date, Open, High, Low, Close, Volume
The app will automatically process and analyze your data

Analyzing Results
📊 Stock Statistics Tab

View comprehensive price statistics
Interactive historical price chart with proper date formatting
Summary statistics (mean, std, min, max, etc.)

🔮 Predictions Tab

Key Predictions: Quick overview of 30-day, 90-day, and 1-year forecasts
Interactive Forecast: Select timeframe (20 Days/90 Days/1 Year) for detailed analysis
Prediction Details: Comprehensive metrics and risk assessment
Forecast Table: Day-by-day predictions with confidence intervals
Components Analysis: Trend, seasonal, and weekly patterns

📈 Understanding the Predictions
Key Metrics Explained

Predicted Price: The model's best estimate for the target date
Expected Change: Dollar amount and percentage change from current price
Confidence Range: Upper and lower bounds showing prediction uncertainty
Risk Assessment: Color-coded evaluation of the forecast

Risk Assessment Categories

🚀 High Growth Potential (>20% increase): Significant upward movement predicted
📈 Moderate Growth (5-20% increase): Positive but modest growth expected
⚖️ Stable/Sideways (-5% to +5%): Relatively stable price predicted
📉 Potential Decline (<-5%): Downward movement forecasted

Chart Elements

Blue Line: Historical actual stock prices
Red Line: Future price predictions
Pink Shaded Area: Confidence interval (uncertainty range)
Gray Dashed Line: "Today" marker separating historical from predicted data

🛠️ Technical Details
Libraries Used

Streamlit: Web application framework
yfinance: Yahoo Finance data downloading
Prophet: Time series forecasting by Facebook
Plotly: Interactive visualizations
Pandas: Data manipulation and analysis
NumPy: Numerical computing
Matplotlib: Additional plotting capabilities
TensorFlow/Keras: Neural network framework (for future LSTM implementation)
Scikit-learn: Machine learning utilities

Model Details

Algorithm: Facebook Prophet
Seasonality: Yearly and weekly patterns enabled
Changepoints: Automatic detection of trend changes
Confidence Intervals: 95% prediction intervals
Data Frequency: Daily stock prices

🔧 Troubleshooting
Common Issues
"No data found for stock symbol"

Verify the ticker symbol is correct
Check if the stock is publicly traded
Try a different date range

"Error running Prophet model"

Ensure you have enough historical data (minimum 10 data points)
Check for data quality issues
Verify date format in uploaded CSV files

Chart display issues

Refresh the browser page
Clear browser cache
Check internet connection for external data

Getting Help

Check the debug information displayed in error messages
Verify all required packages are installed
Ensure Python version compatibility (3.8+)

🚀 Future Enhancements
Planned Features
Multiple Stock Comparison: Side-by-side analysis
Portfolio Optimization: Multi-asset allocation suggestions
Export Functionality: Download predictions as PDF/Excel
Real-time Updates: Live data streaming
Technical Indicators: RSI, MACD, Moving averages

📄 License
This project is open source and available under the MIT License.

👨‍💻 Authors
Anouzla Yassine "TheUchiwa"  anouzlay@gmail.com

Built with ❤️ for the financial analysis and data science community
