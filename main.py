import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import streamlit as st 
import datetime

st.title("Stock Price Forecasting")
if 'dataFrameyahoo' not in st.session_state:
    st.session_state.dataFrameyahoo = pd.DataFrame()
if 'ticker' not in st.session_state:
    st.session_state.ticker = "Unknown"
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def yahoofinance(symbol, start, end):
    data = yf.download(symbol, start, end)
    data.reset_index(inplace=True)
    return pd.DataFrame(data)


if st.session_state.data_loaded and not st.session_state.dataFrameyahoo.empty:
    dataFrameyahoo = st.session_state.dataFrameyahoo
    ticker = st.session_state.ticker
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**📈 Analyzing:** {ticker}")
    with col2:
        if st.button("🔄 Change Stock"):
            st.session_state.dataFrameyahoo = pd.DataFrame()
            st.session_state.ticker = "Unknown"
            st.session_state.data_loaded = False
            st.rerun()
    
else:
    st.text('Select the data source:')
    data_option = st.selectbox(
       "How would you like to choose your data?",
       ("Import", "Download"),
       index=None,
       placeholder="Select option method...",
    )

    if data_option == "Download":
        st.text('Enter stock details:')
        stock_symbol = st.text_input("Enter stock symbol (e.g., AAPL for Apple)")
        stock_start = st.date_input("Start date", datetime.date(2021, 1, 1))
        stock_end = st.date_input("End date", datetime.date.today())
        auto_latest = st.checkbox("📈 Get latest data (recommended)", value=True)
        if auto_latest:
            stock_end = datetime.date.today()
            st.info(f"Will fetch data up to: {stock_end}")
        
        download_btn = st.button("Download")

        if download_btn and stock_symbol:
            with st.spinner('Downloading data...'):
                try:
                    downloaded_data = yahoofinance(stock_symbol, stock_start, stock_end)
                    if downloaded_data.empty:
                        st.error("No data found for the selected stock symbol. Please check the symbol and try again.")
                    else:
                        st.session_state.dataFrameyahoo = downloaded_data
                        st.session_state.ticker = stock_symbol
                        st.session_state.data_loaded = True
                        st.success("Data downloaded successfully!")
                        st.rerun() 
                except Exception as e:
                    st.error(f"Error downloading data: {str(e)}")

    if data_option == "Import":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                file_name = uploaded_file.name.rsplit('.', 1)[0]
                st.session_state.dataFrameyahoo = df
                st.session_state.ticker = file_name
                st.session_state.data_loaded = True
                st.success("File uploaded successfully!")
                st.rerun()  
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

if st.session_state.data_loaded and not st.session_state.dataFrameyahoo.empty:
    dataFrameyahoo = st.session_state.dataFrameyahoo
    ticker = st.session_state.ticker

    st.markdown("### Stock Information")
    tab1, tab4 = st.tabs(["Stock Statistics" , "Predictions"])

    with tab1:
        st.write("Stock Price Statistics:")
        st.dataframe(dataFrameyahoo[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

        fig, ax = plt.subplots(figsize=(20, 8))
        ax.set_title(f"Close price of {ticker}")

        if isinstance(dataFrameyahoo.columns, pd.MultiIndex):

            date_col = None
            close_col = None
            for col in dataFrameyahoo.columns:
                if col[0] == 'Date':
                    date_col = col
                elif col[0] == 'Close':
                    close_col = col
            
            if date_col is not None and close_col is not None:
                dates = pd.to_datetime(dataFrameyahoo[date_col])
                prices = dataFrameyahoo[close_col]
                ax.plot(dates, prices)
            else:
                # Fallback to index if no date column found
                ax.plot(dataFrameyahoo.index, dataFrameyahoo[close_col] if close_col else dataFrameyahoo.iloc[:, -1])
        else:
            # For regular columns
            if 'Date' in dataFrameyahoo.columns:
                dates = pd.to_datetime(dataFrameyahoo['Date'])
                prices = dataFrameyahoo['Close']
                ax.plot(dates, prices)
            else:
                # If no Date column, check if index is datetime
                if pd.api.types.is_datetime64_any_dtype(dataFrameyahoo.index):
                    ax.plot(dataFrameyahoo.index, dataFrameyahoo['Close'])
                else:
                    # Last resort: use index but try to make it meaningful
                    ax.plot(dataFrameyahoo.index, dataFrameyahoo['Close'])
        
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Close Price', fontsize=10)
        ax.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)

    with tab4:
        st.markdown("### PredictionModel")
        try:
            with st.spinner('Running  model...'):
                # First, flatten the multi-level column names
                if isinstance(dataFrameyahoo.columns, pd.MultiIndex):

                    new_columns = []
                    for col in dataFrameyahoo.columns:
                        if col[0] == 'Date' or col[0] == 'Year':
                            new_columns.append(col[0])  # Use just 'Date' or 'Year'
                        else:
                            new_columns.append(col[0]) 
                    
                    df_work = dataFrameyahoo.copy()
                    df_work.columns = new_columns
                else:
                    df_work = dataFrameyahoo.copy()
                
                # Create a clean DataFrame for Prophet
                df_prophet = pd.DataFrame()
                
                # Handle Date column
                if 'Date' in df_work.columns:
                    df_prophet['ds'] = pd.to_datetime(df_work['Date'])
                else:
                    st.error("No Date column found in the data.")
                    st.stop()
                
                # Handle Close price column
                if 'Close' in df_work.columns:
                    df_prophet['y'] = pd.to_numeric(df_work['Close'], errors='coerce')
                else:
                    st.error("No Close column found in the data.")
                    st.stop()
                
                # Remove any rows with NaN values
                df_prophet = df_prophet.dropna()
                
                # Sort by date and ensure proper datetime format
                df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)
                
                # Ensure ds column is datetime and properly formatted
                df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
                
                # Ensure regular frequency (daily) for Prophet
                df_prophet = df_prophet.set_index('ds').asfreq('D').reset_index()
                df_prophet['y'] = df_prophet['y'].interpolate(method='linear')
                
                # Check if we have enough data
                if len(df_prophet) > 10:
                    st.write(f"Using {len(df_prophet)} data points for prediction")
                    
                    # Initialize Prophet model with explicit frequency
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False,  # Set to False for stock data
                        changepoint_prior_scale=0.05,
                        interval_width=0.95
                    )
                    
                    # Fit the model
                    model.fit(df_prophet)
                    last_date = df_prophet['ds'].max()
                    future_dates_full = pd.date_range(
                        start=last_date + pd.DateOffset(days=1),
                        periods=365,
                        freq='D'
                    )
                    
                    # Combine historical and future dates
                    all_dates = pd.concat([
                        df_prophet['ds'],
                        pd.Series(future_dates_full)
                    ]).reset_index(drop=True)
                    
                    future_df = pd.DataFrame({'ds': all_dates})
                    
                    # Make predictions
                    forecast = model.predict(future_df)
                    
                    # Get current price and date info
                    current_price = df_prophet['y'].iloc[-1]
                    current_date = df_prophet['ds'].iloc[-1]
                    
                    # Check if data is current
                    days_behind = (pd.Timestamp.now().normalize() - current_date).days
                    
                    if days_behind > 5:
                        st.warning(f"⚠️ Data is {days_behind} days old (last update: {current_date.strftime('%Y-%m-%d')}). Consider updating to get current predictions.")
                    else:
                        st.success(f"✅ Data is current (last update: {current_date.strftime('%Y-%m-%d')})")
                    
                    # Get predictions for specific time periods
                    future_forecast = forecast[forecast['ds'] > current_date]
                    
                    # 30 days prediction
                    prediction_30d = future_forecast.iloc[29] if len(future_forecast) > 29 else None
                    # 90 days prediction  
                    prediction_90d = future_forecast.iloc[89] if len(future_forecast) > 89 else None
                    # 1 year prediction
                    prediction_1year = future_forecast.iloc[-1] if len(future_forecast) > 0 else None
                    
                    # Display key predictions with current context
                    st.markdown("### 🎯 **Key Predictions**")
                    st.caption(f"Based on data through: {current_date.strftime('%B %d, %Y')}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        price_label = "Latest Price" if days_behind <= 1 else f"Price ({days_behind}d ago)"
                        st.metric(
                            label=price_label,
                            value=f"${current_price:.2f}",
                            delta=None
                        )
                    
                    with col2:
                        if prediction_30d is not None:
                            change_30d = prediction_30d['yhat'] - current_price
                            change_pct_30d = (change_30d / current_price) * 100
                            pred_date_30d = prediction_30d['ds'].strftime('%b %d')
                            st.metric(
                                label=f"30 Days ({pred_date_30d})",
                                value=f"${prediction_30d['yhat']:.2f}",
                                delta=f"{change_pct_30d:.1f}%"
                            )
                    
                    with col3:
                        if prediction_90d is not None:
                            change_90d = prediction_90d['yhat'] - current_price
                            change_pct_90d = (change_90d / current_price) * 100
                            pred_date_90d = prediction_90d['ds'].strftime('%b %d')
                            st.metric(
                                label=f"90 Days ({pred_date_90d})",
                                value=f"${prediction_90d['yhat']:.2f}",
                                delta=f"{change_pct_90d:.1f}%"
                            )
                    
                    with col4:
                        if prediction_1year is not None:
                            change_1year = prediction_1year['yhat'] - current_price
                            change_pct_1year = (change_1year / current_price) * 100
                            pred_date_1year = prediction_1year['ds'].strftime('%b %Y')
                            st.metric(
                                label=f"1 Year ({pred_date_1year})",
                                value=f"${prediction_1year['yhat']:.2f}",
                                delta=f"{change_pct_1year:.1f}%"
                            )
                    
                    # Display recent forecast data
                    st.markdown("### 📅 **Detailed Forecast Table**")
                    forecast_display = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(20).copy()
                    forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d')
                    forecast_display.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
                    forecast_display = forecast_display.round(2)
                    st.dataframe(forecast_display, use_container_width=True)
                    
                    # Plot the forecast with interactive time period selection
                    st.markdown("### 📈 **Interactive Forecast Visualization**")
                    
                    # Time period selector
                    time_period = st.selectbox(
                        "Select forecast period:",
                        ["20 Days", "90 Days", "1 Year"],
                        index=2,  # Default to 1 Year
                        key="forecast_period"  # Add unique key to prevent conflicts
                    )
                    
                    # Create future dates for selected period using pd.date_range
                    if time_period == "20 Days":
                        periods = 20
                        title_suffix = "20 Days"
                    elif time_period == "90 Days":
                        periods = 90
                        title_suffix = "90 Days"
                    else:  # 1 Year
                        periods = 365
                        title_suffix = "1 Year"
                    
                    # Create interactive forecast for selected period
                    try:
                        interactive_future_dates = pd.date_range(
                            start=pd.Timestamp(current_date) + pd.DateOffset(days=1),
                            periods=periods,
                            freq='D'
                        )
                        
                        # Combine historical and selected future dates
                        interactive_all_dates = pd.concat([
                            df_prophet['ds'],
                            pd.Series(interactive_future_dates)
                        ]).reset_index(drop=True)
                        
                        interactive_future_df = pd.DataFrame({'ds': interactive_all_dates})
                        forecast_interactive = model.predict(interactive_future_df)
                    except Exception as e:
                        st.error(f"Error creating interactive forecast: {str(e)}")
                        # Fallback: use the original forecast data
                        forecast_interactive = forecast
                    
                    # Create interactive plotly chart
                    import plotly.graph_objects as go
                    
                    # Get historical data (last 60 days for context)
                    historical_context = df_prophet.tail(60)
                    
                    # Get future data for the selected period
                    try:
                        future_data = forecast_interactive[forecast_interactive['ds'] > pd.Timestamp(current_date)]
                        if len(future_data) == 0:
                            future_data = forecast_interactive.tail(periods)
                    except:
                        future_data = forecast_interactive.tail(periods)
                    
                    # Create connection point between historical and prediction
                    connection_point = pd.DataFrame({
                        'ds': [current_date],
                        'yhat': [current_price],
                        'yhat_lower': [current_price],
                        'yhat_upper': [current_price]
                    })
                    
                    # Combine connection point with future data for smooth transition
                    if len(future_data) > 0:
                        prediction_data = pd.concat([connection_point, future_data]).reset_index(drop=True)
                    else:
                        prediction_data = connection_point
                    
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=historical_context['ds'],
                        y=historical_context['y'],
                        mode='lines',
                        name='Historical Data',
                        line=dict(color='blue', width=2),
                        hovertemplate='<b>%{x}</b><br>Actual: $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Add prediction line (now connected to historical data)
                    fig.add_trace(go.Scatter(
                        x=prediction_data['ds'],
                        y=prediction_data['yhat'],
                        mode='lines',
                        name='Prediction',
                        line=dict(color='red', width=2),
                        hovertemplate='<b>%{x}</b><br>Predicted: $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Add confidence interval (upper bound)
                    fig.add_trace(go.Scatter(
                        x=prediction_data['ds'],
                        y=prediction_data['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        name='Upper Bound',
                        showlegend=False,
                        hoverinfo='none'
                    ))
                    
                    # Add confidence interval (lower bound with fill)
                    fig.add_trace(go.Scatter(
                        x=prediction_data['ds'],
                        y=prediction_data['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        name='Confidence Interval',
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        showlegend=True,
                        hovertemplate='<b>%{x}</b><br>Range: $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Add vertical line to separate historical and predicted data
                    try:
                        fig.add_vline(
                            x=pd.Timestamp(current_date), 
                            line_dash="dash", 
                            line_color="gray",
                            annotation_text="Today",
                            annotation_position="top"
                        )
                    except:
                        # Fallback if vline fails
                        pass
                    
                    # Update layout
                    fig.update_layout(
                        title=f'{ticker} Stock Price Forecast - {title_suffix}',
                        xaxis_title='Date',
                        yaxis_title='Stock Price ($)',
                        hovermode='closest',
                        width=None,
                        height=600,
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        ),
                        # Remove the default hover box styling
                        hoverlabel=dict(
                            bgcolor="white",
                            bordercolor="black",
                            font_size=12,
                        )
                    )
                    
                    # Display the interactive chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show interactive summary for selected period
                    if len(prediction_data) > 1:  # More than just connection point
                        last_prediction = prediction_data.iloc[-1]
                        change = last_prediction['yhat'] - current_price
                        change_pct = (change / current_price) * 100
                        
                        st.markdown(f"### 📊 **{title_suffix} Prediction Details & Summary**")
                        
                        # Create columns for metrics and details

                        
                        
                        st.markdown("#### 📈 **Key Metrics**")
                            
                            # Three metric columns
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                        with metric_col1:
                                st.metric(
                                    label=f"Price in {title_suffix}",
                                    value=f"${last_prediction['yhat']:.2f}",
                                    delta=f"{change_pct:.1f}%"
                                )
                            
                        with metric_col2:
                                st.metric(
                                    label="Expected Change",
                                    value=f"${abs(change):.2f}",
                                    delta="Gain" if change > 0 else "Loss"
                                )
                            
                        with metric_col3:
                                confidence_range = last_prediction['yhat_upper'] - last_prediction['yhat_lower']
                                st.metric(
                                    label="Confidence Range",
                                    value=f"±${confidence_range/2:.2f}",
                                    delta=f"{(confidence_range/last_prediction['yhat']*50):.1f}%"
                                )
                        
                        
                        st.markdown("#### 📋 **Prediction Details**")
                            
                        pred_date = last_prediction['ds'].strftime('%Y-%m-%d')
                        pred_price = last_prediction['yhat']
                        lower_bound = last_prediction['yhat_lower']
                        upper_bound = last_prediction['yhat_upper']
                            
                        st.write(f"**Target Date:** {pred_date}")
                        st.write(f"**Predicted Price:** ${pred_price:.2f}")
                        st.write(f"**Confidence Range:** ${lower_bound:.2f} - ${upper_bound:.2f}")
                        st.write(f"**Expected Change:** {change_pct:.1f}% ({'+' if change > 0 else ''}${change:.2f})")
                            
                            # Risk assessment based on selected period
                        st.markdown("**Risk Assessment:**")
                        if change_pct > 20:
                                st.success("🚀 **High Growth Potential** - Model predicts significant upward movement")
                        elif change_pct > 5:
                                st.info("📈 **Moderate Growth** - Model predicts positive but modest growth")
                        elif change_pct > -5:
                                st.warning("⚖️ **Stable/Sideways** - Model predicts relatively stable price")
                        else:
                                st.error("📉 **Potential Decline** - Model predicts downward movement")
                    
                    # Plot components
                    st.markdown(" #### 📈 **Forecast Components:** ")
                    fig2 = model.plot_components(forecast)
                    st.pyplot(fig2)
                    
                else:
                    st.warning("Insufficient data for Prophet model. Need more than 10 data points.")
                    
        except Exception as e:
            st.error(f"Error running Prophet model: {str(e)}")
            st.info("Debug information:")
            st.write(f"DataFrame shape: {dataFrameyahoo.shape}")
            st.write(f"DataFrame columns: {list(dataFrameyahoo.columns)}")
            if not dataFrameyahoo.empty:
                st.write("First few rows:")
                st.dataframe(dataFrameyahoo.head())
            st.info("Try using a different date range or check your data quality.")