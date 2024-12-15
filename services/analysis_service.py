import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import Ridge
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class AnalysisService:
    """Service for analyzing Treasury yield curve data."""

    def __init__(self):
        self.metrics: Dict[str, Any] = {}

    def calculate_metrics_over_time(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate yield curve metrics over time."""
        try:
            # Calculate spreads over time
            spreads = {
                '2y10y': df['10 Yr'] - df['2 Yr'],
                '3m10y': df['10 Yr'] - df['3 Mo'],
                '5y30y': df['30 Yr'] - df['5 Yr']
            }

            # Calculate rolling metrics
            window = 30  # 30-day window for rolling calculations
            rolling_metrics = {
                'volatility': df.std(axis=1).rolling(window=window).mean(),
                'mean_yield': df.mean(axis=1).rolling(window=window).mean(),
                'curve_steepness': (df['30 Yr'] - df['1 Mo']).rolling(window=window).mean()
            }

            # Calculate economic stress indicators
            stress_indicators = pd.DataFrame({
                'inversion_depth_2y10y': np.minimum(spreads['2y10y'], 0),
                'inversion_depth_3m10y': np.minimum(spreads['3m10y'], 0),
                'short_term_stress': df['1 Mo'].rolling(window=window).std(),
                'long_term_stress': df['30 Yr'].rolling(window=window).std()
            })

            # Calculate smoothness over time
            maturities = np.array([1/12, 1/4, 1/2, 1, 2, 5, 10, 30])
            smoothness = []
            r_squared = []

            for idx in range(len(df)):
                curve = df.iloc[idx].values
                # Fit polynomial
                coeffs = np.polyfit(maturities, curve, 3)
                poly = np.poly1d(coeffs)
                # Calculate R-squared
                y_pred = poly(maturities)
                ss_tot = np.sum((curve - np.mean(curve)) ** 2)
                ss_res = np.sum((curve - y_pred) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                r_squared.append(r2)
                # Calculate smoothness as inverse of second derivative variance
                x_fine = np.linspace(min(maturities), max(maturities), 100)
                second_deriv = np.polyder(poly, 2)(x_fine)
                smoothness.append(1 / (np.var(second_deriv) + 1e-6))

            curve_shape = pd.DataFrame({
                'r_squared': r_squared,
                'smoothness': smoothness
            }, index=df.index)

            # Calculate recession probability indicator (simplified)
            # Based on yield curve inversion and steepness
            recession_prob = (
                -0.5 * spreads['2y10y'] +  # More negative spread → higher probability
                -0.3 * spreads['3m10y'] +  # More negative spread → higher probability
                0.2 * stress_indicators['short_term_stress']  # Higher stress → higher probability
            ).rolling(window=window).mean()

            # Normalize to 0-1 range
            recession_prob = (recession_prob - recession_prob.min()) / (recession_prob.max() - recession_prob.min())

            self.metrics = {
                'spreads': spreads,
                'rolling_metrics': rolling_metrics,
                'stress_indicators': stress_indicators,
                'curve_shape': curve_shape,
                'recession_probability': recession_prob,
                'current_metrics': {
                    'is_2y10y_inverted': spreads['2y10y'].iloc[-1] < 0,
                    'is_3m10y_inverted': spreads['3m10y'].iloc[-1] < 0,
                    'days_2y10y_inverted': (spreads['2y10y'] < 0).sum(),
                    'days_3m10y_inverted': (spreads['3m10y'] < 0).sum(),
                    'current_recession_prob': recession_prob.iloc[-1],
                    'current_stress_level': stress_indicators.mean(axis=1).iloc[-1],
                    'curve_smoothness': curve_shape['smoothness'].iloc[-1],
                    'r_squared': curve_shape['r_squared'].iloc[-1]
                }
            }

            return self.metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def generate_forecast(self, df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
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