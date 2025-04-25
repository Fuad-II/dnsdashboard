import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from scipy import stats

# Configure page settings
st.set_page_config(
    page_title="Advanced Sales & Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'file_upload_time' not in st.session_state:
    st.session_state.file_upload_time = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'time_col' not in st.session_state:
    st.session_state.time_col = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None

# Function to load data
def load_data(uploaded_file):
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(uploaded_file)
        elif file_extension == "json":
            df = pd.read_json(uploaded_file)
        elif file_extension == "txt":
            df = pd.read_csv(uploaded_file, sep="\t")
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
        
        st.session_state.df = df
        st.session_state.file_name = uploaded_file.name
        st.session_state.file_upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Function to detect date columns
def detect_date_columns(df):
    date_columns = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            date_columns.append(col)
        except:
            pass
    return date_columns

# Function to preprocess time series data
def preprocess_time_series(df, date_column, target_column):
    try:
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Set date column as index
        df_ts = df.copy()
        df_ts.set_index(date_column, inplace=True)
        
        # Sort by date
        df_ts.sort_index(inplace=True)
        
        # Handle any missing values in target column
        if df_ts[target_column].isnull().sum() > 0:
            df_ts[target_column] = df_ts[target_column].interpolate(method='linear')
        
        return df_ts
    except Exception as e:
        st.error(f"Error preprocessing time series data: {e}")
        return None

# Function to decompose time series
def decompose_time_series(df, column, period=None):
    """Decompose time series into trend, seasonal, and residual components"""
    try:
        # If period is not specified, try to infer from data frequency
        if not period:
            if df.index.inferred_freq == 'D':
                period = 7  # Weekly seasonality for daily data
            elif df.index.inferred_freq in ['M', 'MS']:
                period = 12  # Monthly data
            else:
                # Default to 12 if cannot infer
                period = 12
        
        # Perform decomposition
        decomposition = seasonal_decompose(df[column], model='additive', period=period)
        return decomposition
    except Exception as e:
        st.error(f"Error decomposing time series: {e}")
        return None

# Function for Holt-Winters exponential smoothing
def holt_winters_forecast(df, column, forecast_periods=30):
    """Apply Holt-Winters exponential smoothing for forecasting"""
    try:
        # Fit model
        model = ExponentialSmoothing(
            df[column],
            trend='add',
            seasonal='add',
            seasonal_periods=12  # Assuming monthly data, adjust as needed
        ).fit()
        
        # Make forecast
        forecast = model.forecast(forecast_periods)
        
        return model, forecast
    except Exception as e:
        st.error(f"Error in Holt-Winters forecasting: {e}")
        return None, None

# Function for linear regression forecast
def linear_regression_forecast(df, target_column, feature_columns, forecast_periods=30):
    """Train linear regression and make forecast"""
    try:
        # Prepare features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store model and evaluation metrics
        model_info = {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'r2': r2,
            'feature_columns': feature_columns
        }
        
        return model_info
    except Exception as e:
        st.error(f"Error in linear regression forecasting: {e}")
        return None

# Function for XGBoost forecast
def xgboost_forecast(df, target_column, feature_columns):
    """Train XGBoost model and make forecast"""
    try:
        # Prepare features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store model and evaluation metrics
        model_info = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'feature_columns': feature_columns,
            'feature_importance': dict(zip(feature_columns, model.feature_importances_))
        }
        
        return model_info
    except Exception as e:
        st.error(f"Error in XGBoost forecasting: {e}")
        return None

# Function to detect outliers
def detect_outliers(df, column, z_threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = stats.zscore(df[column])
    outliers = np.where(np.abs(z_scores) > z_threshold)
    return df.iloc[outliers]

# Function to calculate sales growth
def calculate_sales_growth(df, date_col, sales_col, freq='M'):
    """Calculate period-over-period sales growth"""
    # Convert date column to datetime if not already
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Group by time period and calculate sales
    sales_by_period = df.groupby(pd.Grouper(key=date_col, freq=freq))[sales_col].sum()
    
    # Calculate period-over-period growth
    sales_growth = sales_by_period.pct_change() * 100
    
    return pd.DataFrame({
        'Period': sales_by_period.index,
        'Sales': sales_by_period.values,
        'Growth_Rate': sales_growth.values
    })

# Function to perform demand elasticity analysis
def demand_elasticity(df, price_col, quantity_col):
    """Calculate price elasticity of demand"""
    # Calculate percentage changes
    price_pct_change = df[price_col].pct_change()
    quantity_pct_change = df[quantity_col].pct_change()
    
    # Calculate elasticity (avoiding division by zero)
    elasticity = np.where(
        price_pct_change != 0,
        quantity_pct_change / price_pct_change,
        np.nan
    )
    
    result_df = pd.DataFrame({
        'Price_Change_Pct': price_pct_change,
        'Quantity_Change_Pct': quantity_pct_change,
        'Elasticity': elasticity
    })
    
    # Interpret elasticity
    result_df['Demand_Type'] = np.where(
        np.abs(result_df['Elasticity']) > 1,
        'Elastic',
        np.where(
            np.abs(result_df['Elasticity']) < 1,
            'Inelastic',
            'Unitary'
        )
    )
    
    return result_df

# Title and introduction
st.title("📊 Advanced Sales & Demand Forecasting")
st.markdown("""
Upload your sales data to get AI-powered insights, visualizations, and forecasts.
Analyze past performance and predict future demand and sales patterns!
""")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Upload Data File")
    uploaded_file = st.file_uploader("Choose a file", 
                                    type=["csv", "xlsx", "xls", "json", "txt"])
    
    if uploaded_file is not None:
        if st.button("Process File"):
            with st.spinner("Processing file..."):
                df = load_data(uploaded_file)
                if df is not None:
                    st.success(f"File '{uploaded_file.name}' loaded successfully!")

    if st.session_state.df is not None:
        st.header("File Information")
        st.write(f"**Name:** {st.session_state.file_name}")
        st.write(f"**Uploaded at:** {st.session_state.file_upload_time}")
        st.write(f"**Rows:** {len(st.session_state.df)}")
        st.write(f"**Columns:** {len(st.session_state.df.columns)}")
        
        st.header("Configure Analysis")
        
        # Detect date columns
        date_columns = detect_date_columns(st.session_state.df)
        if date_columns:
            time_col = st.selectbox(
                "Select Date/Time Column",
                options=date_columns,
                index=0 if date_columns else None
            )
            st.session_state.time_col = time_col
        else:
            st.warning("No date columns detected. Please ensure your data includes date information.")
        
        # Select target column for forecasting
        numeric_cols = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            target_col = st.selectbox(
                "Select Target Column for Forecasting",
                options=numeric_cols,
                index=0 if numeric_cols else None
            )
            st.session_state.target_col = target_col
        
        if st.button("Clear Data"):
            st.session_state.df = None
            st.session_state.file_name = None
            st.session_state.file_upload_time = None
            st.session_state.time_col = None
            st.session_state.target_col = None
            st.session_state.predictions = None
            st.session_state.models = {}
            st.rerun()

# Main content area
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "📈 Sales Performance", 
        "🔮 Demand Forecasting", 
        "📉 Price Elasticity",
        "🧪 Advanced Analysis"
    ])
    
    with tab1:
        # Data overview
        st.header("Data Overview")
        st.markdown("### Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Basic statistics
        st.markdown("### Basic Statistics")
        st.dataframe(df.describe().T, use_container_width=True)
        
        # Missing values information
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.markdown("### Missing Values")
            missing_df = pd.DataFrame({
                'Column': missing_values.index,
                'Missing Values': missing_values.values,
                'Percentage': (missing_values.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
        
        # Data types information
        st.markdown("### Data Types")
        dtypes_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values.astype(str)
        })
        st.dataframe(dtypes_df, use_container_width=True)
        
        # Detect outliers if numeric target column is selected
        if st.session_state.target_col:
            st.markdown("### Outlier Detection")
            z_threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1)
            outliers = detect_outliers(df, st.session_state.target_col, z_threshold)
            
            if not outliers.empty:
                st.markdown(f"Found {len(outliers)} outliers in '{st.session_state.target_col}' using Z-score threshold of {z_threshold}")
                st.dataframe(outliers)
                
                # Plot outliers
                fig = px.scatter(df, x=df.index, y=st.session_state.target_col, 
                                title=f"Outlier Detection for {st.session_state.target_col}")
                fig.add_scatter(x=outliers.index, y=outliers[st.session_state.target_col], 
                                mode='markers', marker=dict(color='red', size=10),
                                name='Outliers')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No outliers detected in '{st.session_state.target_col}' using Z-score threshold of {z_threshold}")
    
    with tab2:
        st.header("Sales Performance Analysis")
        
        if st.session_state.time_col and st.session_state.target_col:
            # Time period selector for analysis
            time_freq = st.selectbox(
                "Select Time Period for Analysis",
                options=[("Daily", "D"), ("Weekly", "W"), ("Monthly", "M"), ("Quarterly", "Q"), ("Yearly", "Y")],
                format_func=lambda x: x[0],
                index=2
            )
            
            # Calculate sales growth
            try:
                sales_growth_df = calculate_sales_growth(
                    df, 
                    st.session_state.time_col, 
                    st.session_state.target_col,
                    freq=time_freq[1]
                )
                
                # Display sales over time
                st.subheader(f"{time_freq[0]} Sales Trend")
                fig = px.line(
                    sales_growth_df, 
                    x='Period', 
                    y='Sales',
                    title=f"{time_freq[0]} Sales Trend"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display growth rate
                st.subheader(f"{time_freq[0]} Growth Rate (%)")
                fig = px.bar(
                    sales_growth_df.dropna(), 
                    x='Period', 
                    y='Growth_Rate',
                    title=f"{time_freq[0]} Sales Growth Rate (%)",
                    color='Growth_Rate',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    labels={'Growth_Rate': 'Growth Rate (%)'}
                )
                fig.update_layout(yaxis_ticksuffix='%')
                st.plotly_chart(fig, use_container_width=True)
                
                # Display sales growth data
                st.subheader("Sales Growth Data")
                st.dataframe(sales_growth_df, use_container_width=True)
                
                # Year-over-Year comparison if enough data
                if time_freq[1] in ['D', 'W', 'M'] and len(sales_growth_df) > 12:
                    st.subheader("Year-over-Year Comparison")
                    # Convert Period to datetime if not already
                    if not isinstance(sales_growth_df['Period'].iloc[0], (pd.Timestamp, datetime)):
                        sales_growth_df['Period'] = pd.to_datetime(sales_growth_df['Period'])
                    
                    # Extract year and month/week/day
                    if time_freq[1] == 'M':
                        sales_growth_df['Year'] = sales_growth_df['Period'].dt.year
                        sales_growth_df['Month'] = sales_growth_df['Period'].dt.month
                        pivot_df = sales_growth_df.pivot(index='Month', columns='Year', values='Sales')
                        
                        fig = px.line(
                            pivot_df, 
                            title="Monthly Sales by Year",
                            labels={'value': 'Sales', 'Month': 'Month', 'variable': 'Year'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif time_freq[1] == 'W':
                        sales_growth_df['Year'] = sales_growth_df['Period'].dt.year
                        sales_growth_df['Week'] = sales_growth_df['Period'].dt.isocalendar().week
                        pivot_df = sales_growth_df.pivot(index='Week', columns='Year', values='Sales')
                        
                        fig = px.line(
                            pivot_df, 
                            title="Weekly Sales by Year",
                            labels={'value': 'Sales', 'Week': 'Week', 'variable': 'Year'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif time_freq[1] == 'D':
                        # For daily data, compare by month and day
                        sales_growth_df['Year'] = sales_growth_df['Period'].dt.year
                        sales_growth_df['Month'] = sales_growth_df['Period'].dt.month
                        sales_growth_df['Day'] = sales_growth_df['Period'].dt.day
                        
                        # Group by month
                        monthly_df = sales_growth_df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
                        pivot_monthly = monthly_df.pivot(index='Month', columns='Year', values='Sales')
                        
                        fig = px.line(
                            pivot_monthly, 
                            title="Monthly Sales by Year",
                            labels={'value': 'Sales', 'Month': 'Month', 'variable': 'Year'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error analyzing sales performance: {e}")
        else:
            st.warning("Please select both a time column and a target column in the sidebar to analyze sales performance.")
    
    with tab3:
        st.header("Demand Forecasting")
        
        if st.session_state.time_col and st.session_state.target_col:
            # Preprocess time series data
            df_ts = preprocess_time_series(df, st.session_state.time_col, st.session_state.target_col)
            
            if df_ts is not None:
                st.subheader("Time Series Analysis")
                
                # Plot time series
                fig = px.line(
                    df_ts, 
                    y=st.session_state.target_col,
                    title=f"Time Series for {st.session_state.target_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Time series decomposition
                st.subheader("Time Series Decomposition")
                
                # Allow user to select seasonality period
                freq_mapping = {
                    'Daily': 7,       # Weekly seasonality
                    'Weekly': 52,     # Yearly seasonality
                    'Monthly': 12,    # yearly seasonality
                    'Quarterly': 4,   # Yearly seasonality
                    'Custom': 0       # User-defined
                }
                
                selected_freq = st.selectbox(
                    "Select Seasonality Period",
                    options=list(freq_mapping.keys()),
                    index=2  # Default to Monthly
                )
                
                if selected_freq == 'Custom':
                    custom_period = st.number_input("Enter custom seasonality period", min_value=2, value=12)
                    period = custom_period
                else:
                    period = freq_mapping[selected_freq]
                
                # Decompose time series
                if len(df_ts) > period:  # Ensure enough data points for decomposition
                    decomposition = decompose_time_series(df_ts, st.session_state.target_col, period)
                    
                    if decomposition:
                        # Plot decomposition
                        fig = plt.figure(figsize=(12, 10))
                        plt.subplot(411)
                        plt.plot(decomposition.observed)
                        plt.title('Observed')
                        
                        plt.subplot(412)
                        plt.plot(decomposition.trend)
                        plt.title('Trend')
                        
                        plt.subplot(413)
                        plt.plot(decomposition.seasonal)
                        plt.title('Seasonality')
                        
                        plt.subplot(414)
                        plt.plot(decomposition.resid)
                        plt.title('Residuals')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Calculate seasonality strength
                        var_seasonal = np.var(decomposition.seasonal)
                        var_resid = np.var(decomposition.resid)
                        seasonality_strength = var_seasonal / (var_seasonal + var_resid)
                        
                        st.metric("Seasonality Strength", f"{seasonality_strength:.4f}")
                        
                        if seasonality_strength > 0.6:
                            st.info("Strong seasonal patterns detected. Consider seasonal forecasting methods.")
                        elif seasonality_strength > 0.3:
                            st.info("Moderate seasonal patterns detected.")
                        else:
                            st.info("Weak seasonal patterns. Simple forecasting methods might be sufficient.")
                else:
                    st.warning(f"Not enough data points for decomposition with period {period}. Need at least {period+1} data points.")
                
                # Forecasting section
                st.subheader("Generate Demand Forecast")
                
                forecast_periods = st.slider(
                    "Number of periods to forecast",
                    min_value=1,
                    max_value=365,
                    value=30
                )
                
                forecasting_method = st.selectbox(
                    "Select Forecasting Method",
                    options=["Holt-Winters Exponential Smoothing", "Linear Regression", "XGBoost"],
                    index=0
                )
                
                if st.button("Generate Forecast"):
                    with st.spinner("Generating forecast..."):
                        if forecasting_method == "Holt-Winters Exponential Smoothing":
                            model, forecast = holt_winters_forecast(
                                df_ts, 
                                st.session_state.target_col,
                                forecast_periods
                            )
                            
                            if model and forecast is not None:
                                # Create forecast dataframe
                                forecast_index = pd.date_range(
                                    start=df_ts.index[-1] + pd.Timedelta(days=1),
                                    periods=forecast_periods,
                                    freq=pd.infer_freq(df_ts.index)
                                )
                                forecast_df = pd.DataFrame({
                                    'Date': forecast_index,
                                    'Forecast': forecast.values
                                })
                                
                                # Plot actual vs forecast
                                fig = go.Figure()
                                
                                # Add actual data
                                fig.add_trace(go.Scatter(
                                    x=df_ts.index,
                                    y=df_ts[st.session_state.target_col],
                                    name='Actual',
                                    line=dict(color='blue')
                                ))
                                
                                # Add forecast
                                fig.add_trace(go.Scatter(
                                    x=forecast_index,
                                    y=forecast.values,
                                    name='Forecast',
                                    line=dict(color='red', dash='dot')
                                ))
                                
                                fig.update_layout(
                                    title='Actual vs Forecast',
                                    xaxis_title='Date',
                                    yaxis_title=st.session_state.target_col,
                                    legend_title='Legend'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show forecast data
                                st.subheader("Forecast Data")
                                st.dataframe(forecast_df)
                                
                                # Save forecast to session state
                                st.session_state.predictions = forecast_df
                        
                        elif forecasting_method == "Linear Regression":
                            # Create features from date
                            df_features = df_ts.reset_index()
                            df_features['year'] = df_features[st.session_state.time_col].dt.year
                            df_features['month'] = df_features[st.session_state.time_col].dt.month
                            df_features['day'] = df_features[st.session_state.time_col].dt.day
                            df_features['dayofweek'] = df_features[st.session_state.time_col].dt.dayofweek
                            df_features['quarter'] = df_features[st.session_state.time_col].dt.quarter
                            
                            # Select features for regression
                            feature_cols = ['year', 'month', 'day', 'dayofweek', 'quarter']
                            
                            # Train regression model
                            model_info = linear_regression_forecast(
                                df_features, 
                                st.session_state.target_col,
                                feature_cols
                            )
                            
                            if model_info:
                                # Create future dates for prediction
                                last_date = df_features[st.session_state.time_col].max()
                                future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_periods)]
                                
                                future_df = pd.DataFrame({st.session_state.time_col: future_dates})
                                future_df['year'] = future_df[st.session_state.time_col].dt.year
                                future_df['month'] = future_df[st.session_state.time_col].dt.month
                                future_df['day'] = future_df[st.session_state.time_col].dt.day
                                future_df['dayofweek'] = future_df[st.session_state.time_col].dt.dayofweek
                                future_df['quarter'] = future_df[st.session_state.time_col].dt.quarter
                                
                                # Scale future features
                                future_features = future_df[feature_cols]
                                future_features_scaled = model_info['scaler'].transform(future_features)
                                
                                # Make predictions
                                predictions = model_info['model'].predict(future_features_scaled)
                                
                                # Create forecast dataframe
                                forecast_df = pd.DataFrame({
                                    'Date': future_dates,
                                    'Forecast': predictions
                                })
                                
                                # Plot actual vs forecast
                                fig = go.Figure()
                                
                                # Add actual data
                                fig.add_trace(go.Scatter(
                                    x=df_features[st.session_state.time_col],
                                    y=df_features[st.session_state.target_col],
                                    name='Actual',
                                    line=dict(color='blue')
                                ))
                                
                                # Add forecast
                                fig.add_trace(go.Scatter(
                                    x=future_dates,
                                    y=predictions,
                                    name='Forecast',
                                    line=dict(color='red', dash='dot')
                                ))
                                
                                fig.update_layout(
                                    title='Actual vs Forecast (Linear Regression)',
                                    xaxis_title='Date',
                                    yaxis_title=st.session_state.target_col,
                                    legend_title='Legend'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                # Show forecast data
                                st.subheader("Forecast Data")
                                st.dataframe(forecast_df)
                                
                                # Display model metrics
                                st.subheader("Model Performance")
                                st.metric("Mean Squared Error", f"{model_info['mse']:.4f}")
                                st.metric("R² Score", f"{model_info['r2']:.4f}")
                                
                                # Save forecast to session state
                                st.session_state.predictions = forecast_df
                                st.session_state.models['linear_regression'] = model_info
                        
                        elif forecasting_method == "XGBoost":
                            # Create features from date
                            df_features = df_ts.reset_index()
                            df_features['year'] = df_features[st.session_state.time_col].dt.year
                            df_features['month'] = df_features[st.session_state.time_col].dt.month
                            df_features['day'] = df_features[st.session_state.time_col].dt.day
                            df_features['dayofweek'] = df_features[st.session_state.time_col].dt.dayofweek
                            df_features['quarter'] = df_features[st.session_state.time_col].dt.quarter
                            df_features['is_weekend'] = df_features['dayofweek'].isin([5, 6]).astype(int)
                            
                            # Create lag features if enough data
                            if len(df_features) > 30:
                                df_features['lag_1'] = df_features[st.session_state.target_col].shift(1)
                                df_features['lag_7'] = df_features[st.session_state.target_col].shift(7)
                                df_features['lag_30'] = df_features[st.session_state.target_col].shift(30)
                                df_features['rolling_mean_7'] = df_features[st.session_state.target_col].rolling(window=7).mean()
                                df_features['rolling_std_7'] = df_features[st.session_state.target_col].rolling(window=7).std()
                                df_features = df_features.dropna()
                            
                            # Select features for XGBoost
                            feature_cols = [col for col in df_features.columns if col not in 
                                           [st.session_state.time_col, st.session_state.target_col]]
                            
                            # Train XGBoost model
                            model_info = xgboost_forecast(
                                df_features,
                                st.session_state.target_col,
                                feature_cols
                            )
                            
                            if model_info:
                                # Create future dates for prediction
                                last_date = df_features[st.session_state.time_col].max()
                                future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_periods)]
                                
                                future_df = pd.DataFrame({st.session_state.time_col: future_dates})
                                future_df['year'] = future_df[st.session_state.time_col].dt.year
                                future_df['month'] = future_df[st.session_state.time_col].dt.month
                                future_df['day'] = future_df[st.session_state.time_col].dt.day
                                future_df['dayofweek'] = future_df[st.session_state.time_col].dt.dayofweek
                                future_df['quarter'] = future_df[st.session_state.time_col].dt.quarter
                                future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
                                
                                # For advanced features like lags and rolling stats, we need to forecast iteratively
                                # This is a simplified approach - for real applications, use recursive forecasting
                                if 'lag_1' in feature_cols:
                                    # Get last values from training data
                                    last_values = df_features[st.session_state.target_col].tail(30).values
                                    predictions = []
                                    
                                    for i in range(forecast_periods):
                                        # Create a single row for prediction
                                        current_features = future_df.iloc[i:i+1].copy()
                                        
                                        # Add lag features
                                        if i == 0:
                                            current_features['lag_1'] = last_values[-1]
                                            current_features['lag_7'] = last_values[-7] if len(last_values) >= 7 else last_values.mean()
                                            current_features['lag_30'] = last_values[-30] if len(last_values) >= 30 else last_values.mean()
                                        else:
                                            current_features['lag_1'] = predictions[-1]
                                            current_features['lag_7'] = last_values[-7+i] if i < 7 else predictions[i-7]
                                            current_features['lag_30'] = last_values[-30+i] if i < 30 else predictions[i-30]
                                        
                                        # Add rolling features (simplified)
                                        current_features['rolling_mean_7'] = np.mean(last_values[-7:])
                                        current_features['rolling_std_7'] = np.std(last_values[-7:])
                                        
                                        # Make prediction
                                        pred = model_info['model'].predict(current_features[feature_cols])
                                        predictions.append(pred[0])
                                        
                                        # Update last_values for next iteration
                                        last_values = np.append(last_values[1:], pred[0])
                                else:
                                    # Simple prediction without lag features
                                    predictions = model_info['model'].predict(future_df[feature_cols])
                                
                                # Create forecast dataframe
                                forecast_df = pd.DataFrame({
                                    'Date': future_dates,
                                    'Forecast': predictions
                                })
                                
                                # Plot actual vs forecast
                                fig = go.Figure()
                                
                                # Add actual data
                                fig.add_trace(go.Scatter(
                                    x=df_features[st.session_state.time_col],
                                    y=df_features[st.session_state.target_col],
                                    name='Actual',
                                    line=dict(color='blue')
                                ))
                                
                                # Add forecast
                                fig.add_trace(go.Scatter(
                                    x=future_dates,
                                    y=predictions,
                                    name='Forecast',
                                    line=dict(color='red', dash='dot')
                                ))
                                
                                fig.update_layout(
                                    title='Actual vs Forecast (XGBoost)',
                                    xaxis_title='Date',
                                    yaxis_title=st.session_state.target_col,
                                    legend_title='Legend'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show forecast data
                                st.subheader("Forecast Data")
                                st.dataframe(forecast_df)
                                
                                # Display model metrics
                                st.subheader("Model Performance")
                                st.metric("Mean Squared Error", f"{model_info['mse']:.4f}")
                                st.metric("R² Score", f"{model_info['r2']:.4f}")
                                
                                # Feature importance
                                st.subheader("Feature Importance")
                                feature_imp = pd.DataFrame({
                                    'Feature': list(model_info['feature_importance'].keys()),
                                    'Importance': list(model_info['feature_importance'].values())
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(
                                    feature_imp,
                                    x='Feature',
                                    y='Importance',
                                    title='Feature Importance',
                                    color='Importance'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Save forecast to session state
                                st.session_state.predictions = forecast_df
                                st.session_state.models['xgboost'] = model_info
                
                # If previous predictions exist, show download button
                if st.session_state.predictions is not None:
                    st.subheader("Download Forecast")
                    
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')
                    
                    csv = convert_df_to_csv(st.session_state.predictions)
                    
                    st.download_button(
                        "Download Forecast as CSV",
                        csv,
                        f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        key='download-csv'
                    )
        else:
            st.warning("Please select both a time column and a target column in the sidebar to perform demand forecasting.")
    
    with tab4:
        st.header("Price Elasticity Analysis")
        
        # Check if we have columns that could be used for price elasticity
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            st.markdown("""
            Price elasticity of demand measures how sensitive the quantity demanded is to price changes.
            - If elasticity < -1: **Elastic demand** (quantity changes more than price)
            - If elasticity between -1 and 0: **Inelastic demand** (quantity changes less than price)
            - If elasticity = -1: **Unit elastic demand** (quantity changes proportionally to price)
            """)
            
            # Let user select price and quantity columns
            price_col = st.selectbox(
                "Select Price Column",
                options=numeric_cols,
                index=0
            )
            
            quantity_col = st.selectbox(
                "Select Quantity/Demand Column",
                options=[col for col in numeric_cols if col != price_col],
                index=0
            )
            
            if st.button("Calculate Price Elasticity"):
                with st.spinner("Calculating price elasticity..."):
                    if st.session_state.time_col:
                        # If time column is available, sort by time first
                        df_sorted = df.sort_values(st.session_state.time_col)
                    else:
                        df_sorted = df
                    
                    # Calculate elasticity
                    elasticity_df = demand_elasticity(df_sorted, price_col, quantity_col)
                    
                    if not elasticity_df.empty:
                        # Drop rows with NaN elasticity (happens at first row and when price doesn't change)
                        elasticity_df = elasticity_df.dropna()
                        
                        if not elasticity_df.empty:
                            # Calculate average elasticity
                            avg_elasticity = elasticity_df['Elasticity'].mean()
                            
                            # Interpret elasticity
                            if avg_elasticity < -1:
                                interpretation = "Elastic Demand (highly sensitive to price changes)"
                                interpretation_color = "red"
                            elif avg_elasticity > -1 and avg_elasticity < 0:
                                interpretation = "Inelastic Demand (less sensitive to price changes)"
                                interpretation_color = "green"
                            elif avg_elasticity == -1:
                                interpretation = "Unit Elastic Demand"
                                interpretation_color = "orange"
                            else:
                                interpretation = "Unusual Demand Pattern (positive elasticity)"
                                interpretation_color = "purple"
                            
                            # Display results
                            st.metric("Average Price Elasticity", f"{avg_elasticity:.4f}")
                            st.markdown(f"**Interpretation:** <span style='color:{interpretation_color}'>{interpretation}</span>", unsafe_allow_html=True)
                            
                            # Plot elasticity
                            fig = px.scatter(
                                elasticity_df,
                                x='Price_Change_Pct',
                                y='Quantity_Change_Pct',
                                title='Price vs Demand Changes',
                                color='Elasticity',
                                color_continuous_scale=['red', 'yellow', 'green'],
                                hover_data=['Elasticity', 'Demand_Type']
                            )
                            
                            # Add reference lines
                            fig.add_shape(
                                type='line',
                                x0=elasticity_df['Price_Change_Pct'].min(),
                                y0=0,
                                x1=elasticity_df['Price_Change_Pct'].max(),
                                y1=0,
                                line=dict(dash='dash', color='gray')
                            )
                            
                            fig.add_shape(
                                type='line',
                                x0=0,
                                y0=elasticity_df['Quantity_Change_Pct'].min(),
                                x1=0,
                                y1=elasticity_df['Quantity_Change_Pct'].max(),
                                line=dict(dash='dash', color='gray')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show elasticity data
                            st.subheader("Price Elasticity Data")
                            st.dataframe(elasticity_df, use_container_width=True)
                            
                            # Business recommendations based on elasticity
                            st.subheader("Business Recommendations")
                            
                            if avg_elasticity < -1.5:
                                st.markdown("""
                                ### High Elasticity Detected
                                
                                **Recommendations:**
                                - Consider **lowering prices** to increase revenue as demand is highly sensitive to price
                                - Focus on **price-based promotions** and discounts
                                - Emphasize **competitive pricing** in marketing
                                - Explore **value-tier products** to capture price-sensitive customers
                                - Implement **loyalty programs** to reduce price sensitivity
                                """)
                            elif avg_elasticity < -1:
                                st.markdown("""
                                ### Moderate Elasticity Detected
                                
                                **Recommendations:**
                                - **Carefully test price changes** as they will impact demand
                                - Consider **bundling strategies** to increase perceived value
                                - Implement **segmented pricing** for different customer groups
                                - Focus on **value communication** in marketing
                                """)
                            elif avg_elasticity > -1 and avg_elasticity < -0.5:
                                st.markdown("""
                                ### Low Elasticity Detected
                                
                                **Recommendations:**
                                - There may be **opportunities for price increases** without significant volume loss
                                - Focus on **product differentiation** and quality improvements
                                - Emphasize **premium positioning** in marketing
                                - Consider **brand-building** initiatives to further reduce price sensitivity
                                """)
                            elif avg_elasticity > -0.5 and avg_elasticity < 0:
                                st.markdown("""
                                ### Very Low Elasticity Detected
                                
                                **Recommendations:**
                                - **Strong opportunity for price optimization** as demand is minimally affected by price
                                - Focus on **premium/luxury positioning**
                                - Emphasize **unique product features** and benefits
                                - Consider **high-value limited editions** or exclusive offerings
                                """)
                            else:
                                st.markdown("""
                                ### Unusual Elasticity Pattern Detected
                                
                                **Recommendations:**
                                - **Conduct further analysis** to understand this unusual relationship
                                - Check for **data quality issues** or confounding factors
                                - Consider if **external factors** (promotions, seasonality, competition) are influencing results
                                - **Segment data** to identify if certain groups exhibit different elasticity patterns
                                """)
                        else:
                            st.warning("Could not calculate elasticity - insufficient price changes in the data.")
            else:
                st.info("Select price and quantity columns and click 'Calculate Price Elasticity' to analyze how price changes affect demand.")
        else:
            st.warning("Price elasticity analysis requires at least two numeric columns (for price and quantity).")
    
    with tab5:
        st.header("Advanced Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            options=[
                "Segmentation Analysis",
                "Anomaly Detection",
                "What-If Scenario Modeling",
                "KPI Dashboard"
            ]
        )
        
        if analysis_type == "Segmentation Analysis":
            st.subheader("Customer/Product Segmentation")
            
            if len(numeric_cols) >= 2:
                # Select columns for segmentation
                segmentation_cols = st.multiselect(
                    "Select columns for segmentation analysis",
                    options=numeric_cols,
                    default=numeric_cols[:2] if len(numeric_cols) >= 2 else []
                )
                
                if len(segmentation_cols) >= 2:
                    # Number of segments
                    num_segments = st.slider("Number of segments", min_value=2, max_value=10, value=3)
                    
                    if st.button("Perform Segmentation"):
                        with st.spinner("Performing segmentation analysis..."):
                            try:
                                from sklearn.cluster import KMeans
                                
                                # Select data for clustering
                                cluster_data = df[segmentation_cols].copy()
                                
                                # Handle missing values
                                cluster_data = cluster_data.fillna(cluster_data.mean())
                                
                                # Scale the data
                                scaler = StandardScaler()
                                scaled_data = scaler.fit_transform(cluster_data)
                                
                                # Perform K-means clustering
                                kmeans = KMeans(n_clusters=num_segments, random_state=42, n_init=10)
                                df['Segment'] = kmeans.fit_predict(scaled_data)
                                
                                # Plot segments
                                if len(segmentation_cols) == 2:
                                    # Use scatter plot for 2D segmentation
                                    fig = px.scatter(
                                        df,
                                        x=segmentation_cols[0],
                                        y=segmentation_cols[1],
                                        color='Segment',
                                        title=f'Segmentation Analysis ({segmentation_cols[0]} vs {segmentation_cols[1]})',
                                        color_discrete_sequence=px.colors.qualitative.Bold
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    # Use pairwise scatter plots for more than 2D
                                    for i in range(len(segmentation_cols)):
                                        for j in range(i + 1, len(segmentation_cols)):
                                            fig = px.scatter(
                                                df,
                                                x=segmentation_cols[i],
                                                y=segmentation_cols[j],
                                                color='Segment',
                                                title=f'Segmentation Analysis ({segmentation_cols[i]} vs {segmentation_cols[j]})',
                                                color_discrete_sequence=px.colors.qualitative.Bold
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                
                                # Segment statistics
                                st.subheader("Segment Statistics")
                                
                                segment_stats = df.groupby('Segment')[segmentation_cols].agg(['mean', 'std', 'min', 'max'])
                                st.dataframe(segment_stats, use_container_width=True)
                                
                                # Segment sizes
                                segment_sizes = df['Segment'].value_counts().reset_index()
                                segment_sizes.columns = ['Segment', 'Count']
                                segment_sizes['Percentage'] = segment_sizes['Count'] / len(df) * 100
                                
                                fig = px.pie(
                                    segment_sizes,
                                    values='Count',
                                    names='Segment',
                                    title='Segment Distribution',
                                    color_discrete_sequence=px.colors.qualitative.Bold
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Business interpretations
                                st.subheader("Business Interpretations")
                                
                                for segment in range(num_segments):
                                    segment_data = segment_stats.xs(segment, level='Segment')
                                    
                                    st.markdown(f"#### Segment {segment} ({segment_sizes[segment_sizes['Segment']==segment]['Percentage'].values[0]:.1f}% of data)")
                                    
                                    # Create a description of this segment based on its values
                                    characteristics = []
                                    for col in segmentation_cols:
                                        mean_val = segment_data.loc[col, 'mean']
                                        overall_mean = df[col].mean()
                                        
                                        if mean_val > overall_mean * 1.2:
                                            characteristics.append(f"High {col}")
                                        elif mean_val < overall_mean * 0.8:
                                            characteristics.append(f"Low {col}")
                                        else:
                                            characteristics.append(f"Average {col}")
                                    
                                    st.write("**Characteristics:** " + ", ".join(characteristics))
                                    
                                    # Business recommendations based on segment
                                    if "High" in characteristics[0] and "High" in characteristics[1]:
                                        st.write("**Recommendation:** Premium offerings, loyalty programs, VIP services")
                                    elif "Low" in characteristics[0] and "Low" in characteristics[1]:
                                        st.write("**Recommendation:** Entry-level offerings, cost-effective solutions, retention focus")
                                    elif "High" in characteristics[0] and "Low" in characteristics[1]:
                                        st.write("**Recommendation:** Value-based offerings, educational content, upgrade paths")
                                    elif "Low" in characteristics[0] and "High" in characteristics[1]:
                                        st.write("**Recommendation:** Feature-focused marketing, technical support, specialized solutions")
                                    else:
                                        st.write("**Recommendation:** Balanced approach, standard offerings, regular engagement")
                            
                            except Exception as e:
                                st.error(f"Error performing segmentation: {e}")
                else:
                    st.info("Please select at least two columns for segmentation analysis.")
            else:
                st.warning("Segmentation analysis requires at least two numeric columns.")
        
        elif analysis_type == "Anomaly Detection":
            st.subheader("Anomaly Detection")
            
            if len(numeric_cols) >= 1:
                # Select column for anomaly detection
                anomaly_col = st.selectbox(
                    "Select column for anomaly detection",
                    options=numeric_cols
                )
                
                # Detection method
                detection_method = st.selectbox(
                    "Detection Method",
                    options=["Z-Score", "IQR (Interquartile Range)", "Isolation Forest"],
                    index=0
                )
                
                if st.button("Detect Anomalies"):
                    with st.spinner("Detecting anomalies..."):
                        try:
                            if detection_method == "Z-Score":
                                # Z-score threshold
                                z_threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1)
                                
                                # Calculate Z-scores
                                z_scores = stats.zscore(df[anomaly_col].dropna())
                        except Exception as e:
                            st.error(f"Error in anomaly detection: {e}")
                        
                        df_z = df.dropna(subset=[anomaly_col]).copy()
                        df_z['Z_Score'] = z_scores
                                
                                # Flag anomalies
                        df_z['Is_Anomaly'] = np.abs(df_z['Z_Score']) > z_threshold
                                
                                # Display anomalies
                        anomalies = df_z[df_z['Is_Anomaly']]
                                
                    if len(anomalies) > 0:
                                    st.write(f"Found {len(anomalies)} anomalies using Z-score method:")
                                    st.dataframe(anomalies)
                                    
                                    # Visualize anomalies
                                    if st.session_state.time_col and st.session_state.time_col in df_z.columns:
                                        # Time series visualization
                                        fig = px.line(
                                            df_z,
                                            x=st.session_state.time_col,
                                            y=anomaly_col,
                                            title=f"Anomaly Detection for {anomaly_col}"
                                        )
                                        
                                        # Add anomalies as points
                                        fig.add_scatter(
                                            x=anomalies[st.session_state.time_col],
                                            y=anomalies[anomaly_col],
                                            mode='markers',
                                            marker=dict(color='red', size=10),
                                            name='Anomalies'
                                        )
                                    else:
                                        # Index-based visualization
                                        fig = px.line(
                                            df_z,
                                            y=anomaly_col,
                                            title=f"Anomaly Detection for {anomaly_col}"
                                        )
                                        
                                        # Add anomalies as points
                                        fig.add_scatter(
                                            x=anomalies.index,
                                            y=anomalies[anomaly_col],
                                            mode='markers',
                                            marker=dict(color='red', size=10),
                                            name='Anomalies'
                                        )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display anomaly statistics
                                    st.subheader("Anomaly Statistics")
                                    
                                    # Calculate percentage of anomalies
                                    anomaly_pct = len(anomalies) / len(df_z) * 100
                                    st.metric("Percentage of Anomalies", f"{anomaly_pct:.2f}%")
                                    
                                    # Compare anomalies with normal data
                                    comparison = pd.DataFrame({
                                        'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                                        'Normal Data': [
                                            df_z[~df_z['Is_Anomaly']][anomaly_col].mean(),
                                            df_z[~df_z['Is_Anomaly']][anomaly_col].median(),
                                            df_z[~df_z['Is_Anomaly']][anomaly_col].std(),
                                            df_z[~df_z['Is_Anomaly']][anomaly_col].min(),
                                            df_z[~df_z['Is_Anomaly']][anomaly_col].max()
                                        ],
                                        'Anomalies': [
                                            anomalies[anomaly_col].mean(),
                                            anomalies[anomaly_col].median(),
                                            anomalies[anomaly_col].std(),
                                            anomalies[anomaly_col].min(),
                                            anomalies[anomaly_col].max()
                                        ]
                                    })
                                    
                                    st.dataframe(comparison, use_container_width=True)
                                    
                                    # Business impact assessment
                                    st.subheader("Business Impact Assessment")
                                    
                                    # Calculate total value of anomalies
                                    anomaly_sum = anomalies[anomaly_col].sum()
                                    normal_sum = df_z[~df_z['Is_Anomaly']][anomaly_col].sum()
                                    total_sum = df_z[anomaly_col].sum()
                                    
                                    impact_pct = (anomaly_sum / total_sum) * 100 if total_sum != 0 else 0
                                    
                                    st.metric("Impact of Anomalies", f"{impact_pct:.2f}% of total {anomaly_col}")
                                    
                                    if impact_pct > 15:
                                        st.warning(f"Anomalies represent a significant portion ({impact_pct:.2f}%) of total {anomaly_col}.")
                                    elif impact_pct > 5:
                                        st.info(f"Anomalies represent a moderate portion ({impact_pct:.2f}%) of total {anomaly_col}.")
                                    else:
                                        st.success(f"Anomalies represent a small portion ({impact_pct:.2f}%) of total {anomaly_col}.")
                else:
                    st.success(f"No anomalies found using Z-score method with threshold {z_threshold}.")
            
            if detection_method == "IQR (Interquartile Range)":
                # Calculate Q1, Q3 and IQR
                Q1 = df[anomaly_col].quantile(0.25)
                Q3 = df[anomaly_col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Set threshold
                iqr_multiplier = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
                
                # Define bounds for outliers
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                # Flag anomalies
                df_iqr = df.copy()
                df_iqr['Is_Anomaly'] = (df_iqr[anomaly_col] < lower_bound) | (df_iqr[anomaly_col] > upper_bound)
                
                # Display anomalies
                anomalies = df_iqr[df_iqr['Is_Anomaly']]
                                
                if len(anomalies) > 0:
                    st.write(f"Found {len(anomalies)} anomalies using IQR method:")
                    st.dataframe(anomalies)
                    
                    # Visualize anomalies using box plot
                    fig = px.box(
                        df_iqr,
                        y=anomaly_col,
                        title=f"Box Plot with IQR Anomalies for {anomaly_col}"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Visualize anomalies on time series if available
                    if st.session_state.time_col and st.session_state.time_col in df_iqr.columns:
                        fig = px.line(
                            df_iqr,
                            x=st.session_state.time_col,
                            y=anomaly_col,
                            title=f"IQR Anomaly Detection for {anomaly_col}"
                        )

            elif detection_method == "Isolation Forest":
                # Import required library
                from sklearn.ensemble import IsolationForest
                
                # Set contamination parameter
                contamination = st.slider("Contamination (expected proportion of anomalies)", 
                                    0.01, 0.5, 0.1, 0.01)
                
                # Prepare data for isolation forest
                X = df[[anomaly_col]].copy()
                
                # Handle missing values
                X = X.fillna(X.mean())
                
                # Train isolation forest model
                model = IsolationForest(contamination=contamination, random_state=42)
                model.fit(X)
                                        
            # Complete the IQR anomaly detection visualization
            fig.add_scatter(
                x=anomalies[st.session_state.time_col],
                y=anomalies[anomaly_col],
                mode='markers',
                marker=dict(color='red', size=10),
                name='Anomalies'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display anomaly statistics
            st.subheader("Anomaly Statistics")

            # Calculate percentage of anomalies
            anomaly_pct = len(anomalies) / len(df_iqr) * 100
            st.metric("Percentage of Anomalies", f"{anomaly_pct:.2f}%")

            # Compare anomalies with normal data
            comparison = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Normal Data': [
                    df_iqr[~df_iqr['Is_Anomaly']][anomaly_col].mean(),
                    df_iqr[~df_iqr['Is_Anomaly']][anomaly_col].median(),
                    df_iqr[~df_iqr['Is_Anomaly']][anomaly_col].std(),
                    df_iqr[~df_iqr['Is_Anomaly']][anomaly_col].min(),
                    df_iqr[~df_iqr['Is_Anomaly']][anomaly_col].max()
                ],
                'Anomalies': [
                    anomalies[anomaly_col].mean(),
                    anomalies[anomaly_col].median(),
                    anomalies[anomaly_col].std(),
                    anomalies[anomaly_col].min(),
                    anomalies[anomaly_col].max()
                ]
            })

            st.dataframe(comparison, use_container_width=True)
        
elif detection_method == "Isolation Forest":
    # Import required library
    from sklearn.ensemble import IsolationForest
    
    # Set contamination parameter
    contamination = st.slider("Contamination (expected proportion of anomalies)", 
                            0.01, 0.5, 0.1, 0.01)
    
    # Prepare data for isolation forest
    X = df[[anomaly_col]].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Train isolation forest model
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
            
            # Predict anomalies
    df_if = df.copy()
    df_if['Anomaly_Score'] = model.decision_function(X)
    df_if['Is_Anomaly'] = model.predict(X) == -1  # -1 for anomalies, 1 for normal
            
    # Display anomalies
    anomalies = df_if[df_if['Is_Anomaly']]
            
if len(anomalies) > 0:
    st.write(f"Found {len(anomalies)} anomalies using Isolation Forest method:")
    st.dataframe(anomalies)
    
    # Visualize anomalies
    if st.session_state.time_col and st.session_state.time_col in df_if.columns:
        # Time series visualization
        fig = px.line(
            df_if,
            x=st.session_state.time_col,
            y=anomaly_col,
            title=f"Isolation Forest Anomaly Detection for {anomaly_col}"
        )
        
        # Add anomalies as points
        fig.add_scatter(
            x=anomalies[st.session_state.time_col],
            y=anomalies[anomaly_col],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Anomalies'
        )
    else:
        # Index-based visualization
        fig = px.line(
            df_if,
            y=anomaly_col,
            title=f"Isolation Forest Anomaly Detection for {anomaly_col}"
        )
        
        # Add anomalies as points
        fig.add_scatter(
            x=anomalies.index,
            y=anomalies[anomaly_col],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Anomalies'
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Visualize anomaly scores
    fig = px.histogram(
        df_if, 
        x='Anomaly_Score',
        color='Is_Anomaly',
        title='Distribution of Anomaly Scores',
        marginal='box'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display anomaly statistics
    st.subheader("Anomaly Statistics")
    anomaly_pct = len(anomalies) / len(df_if) * 100
    st.metric("Percentage of Anomalies", f"{anomaly_pct:.2f}%")
else:
        st.success(f"No anomalies found using Isolation Forest method with contamination {contamination}.")

# What-If Scenario Modeling section
if analysis_type == "What-If Scenario Modeling":
    st.subheader("What-If Scenario Analysis")
    
    if len(numeric_cols) >= 2:
        # Select dependent variable (target)
        dependent_var = st.selectbox(
            "Select dependent variable (outcome)",
            options=numeric_cols
        )
        
        # Select independent variables (predictors)
        independent_vars = st.multiselect(
            "Select independent variables (predictors)",
            options=[col for col in numeric_cols if col != dependent_var],
            default=[col for col in numeric_cols if col != dependent_var][:2]
        )
        
        if len(independent_vars) >= 1 and st.button("Build What-If Model"):
            with st.spinner("Building regression model for What-If analysis..."):
                try:
                    from sklearn.linear_model import LinearRegression
                    from sklearn.model_selection import train_test_split
                    
                    # Prepare data
                    X = df[independent_vars].copy()
                    y = df[dependent_var].copy()
                    
                    # Handle missing values
                    X = X.fillna(X.mean())
                    y = y.fillna(y.mean())
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Build linear regression model
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Model evaluation
                    train_score = model.score(X_train, y_train)
                    test_score = model.score(X_test, y_test)
                    
                    # Display model metrics
                    st.subheader("Model Performance")
                    col1, col2 = st.columns(2)
                    col1.metric("Training R² Score", f"{train_score:.4f}")
                    col2.metric("Testing R² Score", f"{test_score:.4f}")
                    
                    # Display coefficients
                    coefs = pd.DataFrame({
                        'Variable': independent_vars,
                        'Coefficient': model.coef_
                    })
                    
                    st.subheader("Variable Importance")
                    fig = px.bar(
                        coefs,
                        x='Variable',
                        y='Coefficient',
                        title='Regression Coefficients',
                        color='Coefficient',
                        color_continuous_scale=px.colors.diverging.RdBu,
                        color_continuous_midpoint=0
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # What-If Scenario Builder
                    st.subheader("Scenario Builder")
                    st.markdown("""
                    Adjust the values of the independent variables to see how the dependent variable would change.
                    Current values are set to the average from your data.
                    """)
                    
                    # Create scenario inputs
                    scenario_values = {}
                    for var in independent_vars:
                        min_val = float(df[var].min())
                        max_val = float(df[var].max())
                        mean_val = float(df[var].mean())
                        
                        # Determine step size based on the range
                        range_val = max_val - min_val
                        step = range_val / 100
                        
                        # For small ranges, use smaller step
                        if range_val < 1:
                            step = range_val / 20
                        
                        scenario_values[var] = st.slider(
                            f"{var}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=step
                        )
                    
                    # Create input array for prediction
                    scenario_input = np.array([[scenario_values[var] for var in independent_vars]])
                    
                    # Make prediction
                    prediction = model.predict(scenario_input)[0]
                    
                    # Display prediction
                    st.subheader("Scenario Prediction")
                    st.metric(
                        f"Predicted {dependent_var}",
                        f"{prediction:.2f}",
                        delta=f"{prediction - df[dependent_var].mean():.2f} vs. average"
                    )
                    
                    # Create scenario comparison
                    st.subheader("Sensitivity Analysis")
                    st.markdown("See how changes in each variable affect the prediction")
                    
                    # Select a variable for sensitivity analysis
                    sensitivity_var = st.selectbox(
                        "Select variable for sensitivity analysis",
                        options=independent_vars
                    )
                    
                    # Generate range of values for sensitivity analysis
                    var_min = df[sensitivity_var].min()
                    var_max = df[sensitivity_var].max()
                    var_range = np.linspace(var_min, var_max, 20)
                    
                    # Create predictions for each value
                    sensitivity_results = []
                    base_scenario = scenario_values.copy()
                    
                    for val in var_range:
                        # Update the selected variable
                        temp_scenario = base_scenario.copy()
                        temp_scenario[sensitivity_var] = val
                        
                        # Create input array
                        temp_input = np.array([[temp_scenario[var] for var in independent_vars]])
                        
                        # Make prediction
                        temp_prediction = model.predict(temp_input)[0]
                        
                        # Save result
                        sensitivity_results.append({
                            sensitivity_var: val,
                            f"Predicted {dependent_var}": temp_prediction
                        })
                    
                    # Create sensitivity dataframe
                    sensitivity_df = pd.DataFrame(sensitivity_results)
                    
                    # Plot sensitivity analysis
                    fig = px.line(
                        sensitivity_df,
                        x=sensitivity_var,
                        y=f"Predicted {dependent_var}",
                        title=f"Sensitivity Analysis: Impact of {sensitivity_var} on {dependent_var}",
                        markers=True
                    )
                    
                    # Add current selected value vertical line
                    fig.add_vline(
                        x=scenario_values[sensitivity_var],
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Current Value",
                        annotation_position="top right"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Business insights
                    st.subheader("Business Insights")
                    
                    # Find the most impactful variable
                    abs_coefs = coefs.copy()
                    abs_coefs['Abs_Coefficient'] = abs_coefs['Coefficient'].abs()
                    most_important_var = abs_coefs.sort_values('Abs_Coefficient', ascending=False).iloc[0]['Variable']
                    
                    st.markdown(f"""
                    ### Key Findings:
                    
                    1. **Most Impactful Factor**: {most_important_var} has the strongest effect on {dependent_var}
                    
                    2. **Positive Drivers**: {', '.join(coefs[coefs['Coefficient'] > 0]['Variable'].tolist())} 
                    are positively associated with {dependent_var}
                    
                    3. **Negative Drivers**: {', '.join(coefs[coefs['Coefficient'] < 0]['Variable'].tolist())} 
                    are negatively associated with {dependent_var}
                    
                    4. **Model Reliability**: This model explains {test_score:.1%} of the variation in {dependent_var}
                    """)
                    
                    # Save model to session state for reuse
                    st.session_state.what_if_model = {
                        'model': model,
                        'independent_vars': independent_vars,
                        'dependent_var': dependent_var,
                        'test_score': test_score
                    }
                
                except Exception as e:
                    st.error(f"Error building What-If model: {e}")
        else:
            st.info("Select at least one independent variable and the dependent variable, then click 'Build What-If Model'.")
    else:
        st.warning("What-If analysis requires at least two numeric columns.")

elif analysis_type == "KPI Dashboard":
    st.subheader("Business KPI Dashboard")
    
    if len(numeric_cols) >= 1:
        # Select KPI metrics
        kpi_metrics = st.multiselect(
            "Select KPI metrics to display",
            options=numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )
        
        if len(kpi_metrics) >= 1:
            # Get time column if available
            time_col = st.session_state.time_col if 'time_col' in st.session_state and st.session_state.time_col in df.columns else None
            
            # Optional group by column
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            group_by_col = None
            
            if len(categorical_cols) > 0:
                use_groupby = st.checkbox("Group metrics by category", value=False)
                
                if use_groupby:
                    group_by_col = st.selectbox(
                        "Select category for grouping",
                        options=categorical_cols
                    )
            
            # Generate KPI dashboard
            st.subheader("Key Performance Indicators")
            
            # Create metrics summary
            metric_cols = st.columns(len(kpi_metrics))
            
            for i, metric in enumerate(kpi_metrics):
                current_val = df[metric].mean()
                
                # If time column exists, calculate trend
                if time_col:
                    # Sort by time
                    df_sorted = df.sort_values(time_col)
                    
                    # Calculate first half and second half average
                    mid_point = len(df_sorted) // 2
                    first_half = df_sorted.iloc[:mid_point][metric].mean()
                    second_half = df_sorted.iloc[mid_point:][metric].mean()
                    
                    # Calculate change
                    change = second_half - first_half
                    
                    metric_cols[i].metric(
                        f"Avg. {metric}",
                        f"{current_val:.2f}",
                        f"{change:.2f}"
                    )
                else:
                    metric_cols[i].metric(f"Avg. {metric}", f"{current_val:.2f}")
            
            # Create charts for each KPI
            for metric in kpi_metrics:
                st.subheader(f"{metric} Analysis")
                
                # Create two columns for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Time series or distribution chart
                    if time_col:
                        # Time series chart
                        if group_by_col:
                            # Grouped time series
                            fig = px.line(
                                df.sort_values(time_col),
                                x=time_col,
                                y=metric,
                                color=group_by_col,
                                title=f"{metric} Over Time by {group_by_col}"
                            )
                        else:
                            # Simple time series
                            fig = px.line(
                                df.sort_values(time_col),
                                x=time_col,
                                y=metric,
                                title=f"{metric} Over Time"
                            )
                            
                            # Add trend line
                            df_sorted = df.sort_values(time_col)
                            fig.add_trace(
                                go.Scatter(
                                    x=df_sorted[time_col],
                                    y=df_sorted[metric].rolling(window=max(2, len(df_sorted)//10)).mean(),
                                    mode='lines',
                                    name=f'Trend ({max(2, len(df_sorted)//10)}-period MA)',
                                    line=dict(color='red', dash='dot')
                                )
                            )
                    else:
                        # Distribution chart
                        fig = px.histogram(
                            df,
                            x=metric,
                            title=f"Distribution of {metric}",
                            marginal="box"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Group comparison or correlation chart
                    if group_by_col:
                        # Bar chart by group
                        group_stats = df.groupby(group_by_col)[metric].mean().reset_index()
                        fig = px.bar(
                            group_stats,
                            x=group_by_col,
                            y=metric,
                            title=f"Average {metric} by {group_by_col}",
                            color=metric
                        )
                    else:
                        # Find another numeric column for correlation
                        other_numeric = [col for col in numeric_cols if col != metric]
                        
                        if other_numeric:
                            corr_col = other_numeric[0]
                            fig = px.scatter(
                                df,
                                x=metric,
                                y=corr_col,
                                title=f"{metric} vs {corr_col}",
                                trendline="ols"
                            )
                        else:
                            # If no other numeric column, show cumulative distribution
                            fig = px.ecdf(
                                df,
                                x=metric,
                                title=f"Cumulative Distribution of {metric}"
                            )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # KPI summary insights
            st.subheader("KPI Insights")
            
            # Generate some basic insights
            insights = []
            
            for metric in kpi_metrics:
                # Time trend insight
                if time_col:
                    df_sorted = df.sort_values(time_col)
                    first_period = df_sorted.iloc[:len(df_sorted)//3][metric].mean()
                    last_period = df_sorted.iloc[-len(df_sorted)//3:][metric].mean()
                    
                    percent_change = (last_period - first_period) / first_period * 100 if first_period != 0 else 0
                    
                    if abs(percent_change) > 10:
                        direction = "increased" if percent_change > 0 else "decreased"
                        insights.append(f"**{metric}** has {direction} by {abs(percent_change):.1f}% from the first period to the last period.")
                
                # Variation insight
                cv = df[metric].std() / df[metric].mean() * 100 if df[metric].mean() != 0 else 0
                
                if cv > 50:
                    insights.append(f"**{metric}** shows high variability (CV = {cv:.1f}%), suggesting inconsistent performance.")
                elif cv < 10:
                    insights.append(f"**{metric}** is highly stable (CV = {cv:.1f}%), suggesting consistent performance.")
                
                # Group differences insight
                if group_by_col:
                    group_stats = df.groupby(group_by_col)[metric].mean()
                    max_group = group_stats.idxmax()
                    min_group = group_stats.idxmin()
                    max_val = group_stats.max()
                    min_val = group_stats.min()
                    
                    if max_val > min_val * 1.5:
                        insights.append(f"**{metric}** varies significantly across {group_by_col}: {max_group} performs {(max_val/min_val - 1)*100:.1f}% better than {min_group}.")
            
            if insights:
                for i, insight in enumerate(insights):
                    st.markdown(f"{i+1}. {insight}")
            else:
                st.info("No significant insights detected in the selected KPIs.")
        else:
            st.info("Please select at least one KPI metric to generate the dashboard.")
    else:
        st.warning("KPI Dashboard requires at least one numeric column.")