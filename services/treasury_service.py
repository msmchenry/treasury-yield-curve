from typing import Dict, Optional, Any, List
import pandas as pd
import plotly.graph_objects as go
import json
from fredapi import Fred
from datetime import datetime, timedelta
import logging
import numpy as np
from pathlib import Path
from config import Config
from dataclasses import dataclass
from plotly.graph_objs import Figure
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)

@dataclass
class PlotConfig:
    """Configuration for plot styling and layout."""
    COLORSCALE = 'Viridis'
    PLOT_BGCOLOR = 'white'
    GRID_COLOR = 'lightgrey'
    ZERO_LINE_COLOR = 'red'
    SPREAD_LINE_COLOR = 'blue'
    CURRENT_CURVE_COLOR = 'red'

class DataValidator:
    """Validates treasury yield curve data."""
    
    REQUIRED_COLUMNS = ['1 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '5 Yr', '10 Yr', '30 Yr']
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """Validate the DataFrame structure and content."""
        if df is None or df.empty:
            return False
        return all(col in df.columns for col in DataValidator.REQUIRED_COLUMNS)

class TreasuryService:
    """Service for handling Treasury yield curve data and visualization."""
    
    def __init__(self):
        self.fred = Fred(api_key=Config.FRED_API_KEY)
        self.data: Optional[pd.DataFrame] = None
        self._setup_data_directory()

    def _setup_data_directory(self) -> None:
        """Create data directory if it doesn't exist."""
        Config.DATA_DIR.mkdir(exist_ok=True)

    def load_data(self) -> None:
        """Load yield curve data, fetching from FRED only if necessary."""
        try:
            should_fetch = True
            if Config.DATA_FILE.exists():
                try:
                    df = pd.read_parquet(Config.DATA_FILE)
                    if DataValidator.validate_data(df):
                        last_date = df.index.max()
                        today = pd.Timestamp.now().normalize()
                        if last_date.date() >= today.date():
                            logger.info("Using cached data")
                            self.data = df
                            should_fetch = False
                except Exception as e:
                    logger.error(f"Error reading cache: {e}")
                    should_fetch = True

            if should_fetch:
                self.data = self._fetch_fresh_data()
        except Exception as e:
            logger.error(f"Error loading yield curve data: {str(e)}")
            raise

    def _fetch_fresh_data(self) -> pd.DataFrame:
        """Fetch fresh data from FRED API."""
        logger.info("Fetching new data from FRED")
        try:
            df_dict = {}
            for label, series_id in Config.SERIES_IDS.items():
                logger.debug(f"Fetching data for {label} (Series ID: {series_id})")
                data = self.fred.get_series(series_id, Config.START_DATE, datetime.now())
                df_dict[label] = data

            df = pd.DataFrame(df_dict)
            df = df.dropna()
            df = df.resample('M').last()
            
            # Save to parquet file with compression
            df.to_parquet(Config.DATA_FILE, compression='snappy')
            logger.info("Data saved to disk")
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def get_yield_curve_plot(self, start_year: str, end_year: str) -> Dict[str, str]:
        """Create 3D plot and spread plot from the stored DataFrame with forecast."""
        try:
            df = self._filter_data_by_date_range(start_year, end_year)
            
            # Generate forecast
            forecast_df = self._generate_forecast(df)
            
            # Combine actual and forecast data
            combined_df = pd.concat([df, forecast_df])
            
            yield_curve_fig = self._create_3d_yield_curve(combined_df, len(forecast_df))
            spread_fig = self._create_spread_plot(combined_df, len(forecast_df))

            return {
                'yield_curve': json.dumps(yield_curve_fig.to_dict()),
                'spread': json.dumps(spread_fig.to_dict())
            }
        except Exception as e:
            logger.error(f"Error creating yield curve plot: {str(e)}")
            raise

    def _filter_data_by_date_range(self, start_year: str, end_year: str) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        start_year_float = float(start_year)
        end_year_float = float(end_year)
        
        start_year_int = int(start_year_float)
        start_month = int((start_year_float % 1) * 12) + 1
        start_date = datetime(start_year_int, start_month, 1)
        
        end_year_int = int(end_year_float)
        end_month = int((end_year_float % 1) * 12) + 1
        end_date = datetime(end_year_int, end_month, 1)
        
        return self.data[(self.data.index >= start_date) & 
                        (self.data.index <= end_date)].copy()

    def _generate_forecast(self, df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
        """Generate yield curve forecast using Ridge regression."""
        forecast_df = pd.DataFrame(columns=df.columns)
        
        # For each maturity, fit a model and predict
        for column in df.columns:
            # Prepare data for prediction
            data = df[column].values
            X = np.arange(len(data)).reshape(-1, 1)
            y = data
            
            # Fit model
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            
            # Predict next periods
            future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
            predictions = model.predict(future_X)
            
            # Add to forecast DataFrame
            future_dates = [df.index[-1] + timedelta(days=30 * (i+1)) for i in range(periods)]
            forecast_df[column] = pd.Series(predictions, index=future_dates)
        
        return forecast_df

    def _create_3d_yield_curve(self, df: pd.DataFrame, forecast_points: int = 0) -> Figure:
        """Create 3D yield curve visualization with forecast."""
        x = np.array([Config.MATURITY_MAP[m] for m in df.columns])
        y = np.array([(d - df.index[-1]).days for d in df.index])
        X, Y = np.meshgrid(x, y)
        Z = df.values

        fig = go.Figure()

        # Plot historical data
        historical_len = len(df) - forecast_points
        fig.add_trace(go.Surface(
            x=X[:historical_len].tolist(),
            y=Y[:historical_len].tolist(),
            z=Z[:historical_len].tolist(),
            colorscale=PlotConfig.COLORSCALE,
            name='Historical',
            showscale=True,
            hovertemplate=(
                'Maturity: %{x:.1f} years<br>'
                'Days ago: %{y:.0f}<br>'
                'Yield: %{z:.2f}%<extra></extra>'
            )
        ))

        # Plot forecast data if available
        if forecast_points > 0:
            fig.add_trace(go.Surface(
                x=X[historical_len:].tolist(),
                y=Y[historical_len:].tolist(),
                z=Z[historical_len:].tolist(),
                colorscale='Reds',
                name='Forecast',
                showscale=False,
                opacity=0.7,
                hovertemplate=(
                    'Maturity: %{x:.1f} years<br>'
                    'Days ahead: %{y:.0f}<br>'
                    'Forecast Yield: %{z:.2f}%<extra>Forecast</extra>'
                )
            ))

        # Add the current yield curve
        latest_data = df.iloc[-forecast_points-1]  # Last actual data point
        fig.add_trace(go.Scatter3d(
            x=x.tolist(),
            y=[0] * len(x),
            z=latest_data.values.tolist(),
            mode='lines+markers',
            line=dict(color=PlotConfig.CURRENT_CURVE_COLOR, width=5),
            marker=dict(size=5),
            name='Current Yield Curve'
        ))

        fig.update_layout(
            title='US Treasury Yield Curve Evolution (with 1Q Forecast)',
            scene=dict(
                xaxis_title='Maturity (Years)',
                yaxis_title='Days',
                zaxis_title='Yield (%)',
                xaxis=dict(type='log', tickformat='.1f'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=-0.2)
                )
            ),
            template='plotly_white',
            showlegend=True,
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig

    def _create_spread_plot(self, df: pd.DataFrame, forecast_points: int = 0) -> Figure:
        """Create spread plot visualization with forecast."""
        spread = df['10 Yr'] - df['2 Yr']
        dates = [d.strftime('%Y-%m-%d') for d in df.index]
        
        fig = go.Figure()
        
        # Plot historical spread
        historical_len = len(df) - forecast_points
        fig.add_trace(go.Scatter(
            x=dates[:historical_len],
            y=spread[:historical_len].values.tolist(),
            mode='lines',
            name='Historical Spread',
            line=dict(color=PlotConfig.SPREAD_LINE_COLOR),
            fill='tozeroy',
            fillcolor=f'rgba(0, 0, 255, 0.1)'
        ))
        
        # Plot forecast spread
        if forecast_points > 0:
            fig.add_trace(go.Scatter(
                x=dates[historical_len:],
                y=spread[historical_len:].values.tolist(),
                mode='lines',
                name='Forecast Spread',
                line=dict(color='red', dash='dash'),
                fill='tozeroy',
                fillcolor=f'rgba(255, 0, 0, 0.1)'
            ))
        
        fig.add_hline(
            y=0,
            line=dict(color=PlotConfig.ZERO_LINE_COLOR, dash='dash'),
            annotation_text="Yield Curve Inversion Line",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title='10-Year minus 2-Year Treasury Spread (with 1Q Forecast)',
            xaxis=dict(
                title='Date',
                tickformat='%Y-%m',
                tickangle=45,
                gridcolor=PlotConfig.GRID_COLOR,
                showgrid=True
            ),
            yaxis=dict(
                title='Spread (%)',
                gridcolor=PlotConfig.GRID_COLOR,
                showgrid=True,
                zeroline=True,
                zerolinecolor=PlotConfig.ZERO_LINE_COLOR,
                zerolinewidth=1
            ),
            template='plotly_white',
            showlegend=True,
            height=400,
            margin=dict(l=0, r=0, b=50, t=40),
            plot_bgcolor=PlotConfig.PLOT_BGCOLOR
        )
        
        return fig 