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
        file_extension = uploaded_file.name.split(".")[-1].lower()

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
