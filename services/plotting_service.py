import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs import Figure
from dataclasses import dataclass
from config import Config
import logging
from typing import Dict, Any, List, Union
import json
from datetime import datetime

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

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling NumPy types and datetime objects."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # Handle NaN and Infinity
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, pd.Timestamp) or isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d')
        return super().default(obj)

    def encode(self, obj):
        """Override encode to handle NaN values in nested structures."""
        if isinstance(obj, (list, tuple)):
            return '[%s]' % ', '.join(self.encode(item) for item in obj)
        if isinstance(obj, dict):
            items = []
            for key, value in obj.items():
                encoded_value = self.encode(value)
                items.append(f'"{key}": {encoded_value}')
            return '{%s}' % ', '.join(items)
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return 'null'
        if isinstance(obj, np.float64):
            if np.isnan(obj) or np.isinf(obj):
                return 'null'
            return str(float(obj))
        return super().encode(obj)

class PlottingService:
    """Service for creating Treasury yield curve visualizations."""

    def __init__(self):
        self.plot_config = PlotConfig()

    def _convert_to_serializable(self, fig: Figure) -> Dict[str, Any]:
        """Convert Plotly figure to JSON-serializable format."""
        try:
            return json.loads(json.dumps(fig.to_dict(), cls=NumpyJSONEncoder))
        except Exception as e:
            logger.error(f"Error converting figure to serializable format: {e}")
            raise

    def create_3d_yield_curve(self, df: pd.DataFrame, forecast_points: int = 0) -> Dict[str, Any]:
        """Create 3D yield curve visualization with forecast."""
        x = np.array([Config.MATURITY_MAP[m] for m in df.columns])
        y = df.index.strftime('%Y-%m-%d')
        X, Y = np.meshgrid(x, y)
        Z = df.values

        fig = go.Figure()

        # Plot historical data
        historical_len = len(df) - forecast_points
        fig.add_trace(go.Surface(
            x=X[:historical_len].tolist(),
            y=Y[:historical_len].tolist(),
            z=Z[:historical_len].tolist(),
            colorscale=self.plot_config.COLORSCALE,
            name='Historical',
            showscale=True,
            hovertemplate=(
                'Maturity: %{x:.1f} years<br>'
                'Date: %{y}<br>'
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
                    'Forecast Date: %{y}<br>'
                    'Forecast Yield: %{z:.2f}%<extra>Forecast</extra>'
                )
            ))

        # Add the current yield curve
        latest_data = df.iloc[-forecast_points-1]  # Last actual data point
        fig.add_trace(go.Scatter3d(
            x=x.tolist(),
            y=[latest_data.name.strftime('%Y-%m-%d')] * len(x),
            z=latest_data.values.tolist(),
            mode='lines+markers',
            line=dict(color=self.plot_config.CURRENT_CURVE_COLOR, width=5),
            marker=dict(size=5),
            name='Current Yield Curve'
        ))

        fig.update_layout(
            title='US Treasury Yield Curve Evolution (with 1Q Forecast)',
            scene=dict(
                xaxis_title='Maturity (Years)',
                yaxis_title='Date',
                zaxis_title='Yield (%)',
                xaxis=dict(type='log', tickformat='.1f'),
                bgcolor=self.plot_config.PLOT_BGCOLOR
            ),
            showlegend=True,
            margin=dict(l=0, r=0, b=0, t=30)
        )

        return self._convert_to_serializable(fig)

    def _clean_values(self, values):
        """Replace NaN values with None for JSON serialization."""
        if isinstance(values, (pd.Series, np.ndarray)):
            values = values.tolist()
        return [None if pd.isna(v) else v for v in values]

    def create_spread_plot(self, df: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create spread visualization with metrics."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Treasury Yield Spreads',
                'Economic Stress Indicators',
                'Curve Shape Metrics'
            ),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )

        # Plot spreads
        for spread_name, spread_data in metrics['spreads'].items():
            fig.add_trace(
                go.Scatter(
                    x=df.index.strftime('%Y-%m-%d').tolist(),
                    y=self._clean_values(spread_data),
                    name=f'{spread_name} spread',
                    line=dict(color={
                        '2y10y': self.plot_config.SPREAD_LINE_COLOR,
                        '3m10y': 'green',
                        '5y30y': 'purple'
                    }.get(spread_name, 'gray'))
                ),
                row=1, col=1
            )

        # Add zero line for spreads
        fig.add_hline(
            y=0, line_dash="dash", line_color=self.plot_config.ZERO_LINE_COLOR,
            row=1, col=1
        )

        # Plot stress indicators
        stress = metrics['stress_indicators']
        for col in stress.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index.strftime('%Y-%m-%d').tolist(),
                    y=self._clean_values(stress[col]),
                    name=col.replace('_', ' ').title(),
                    line=dict(width=1)
                ),
                row=2, col=1
            )

        # Add recession probability
        fig.add_trace(
            go.Scatter(
                x=df.index.strftime('%Y-%m-%d').tolist(),
                y=self._clean_values(metrics['recession_probability']),
                name='Recession Probability',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )

        # Plot curve shape metrics
        curve_shape = metrics['curve_shape']
        fig.add_trace(
            go.Scatter(
                x=df.index.strftime('%Y-%m-%d').tolist(),
                y=self._clean_values(curve_shape['smoothness']),
                name='Curve Smoothness',
                line=dict(color='blue')
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index.strftime('%Y-%m-%d').tolist(),
                y=self._clean_values(curve_shape['r_squared']),
                name='R-squared',
                line=dict(color='green')
            ),
            row=3, col=1
        )

        # Add rolling metrics
        rolling = metrics['rolling_metrics']
        fig.add_trace(
            go.Scatter(
                x=df.index.strftime('%Y-%m-%d').tolist(),
                y=self._clean_values(rolling['volatility']),
                name='Yield Volatility',
                line=dict(color='orange', dash='dash')
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text="Yield Curve Analysis",
            plot_bgcolor=self.plot_config.PLOT_BGCOLOR
        )

        # Update axes
        for i in range(1, 4):
            fig.update_xaxes(title_text="Date", row=i, col=1)
        
        fig.update_yaxes(title_text="Spread (%)", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=3, col=1)

        return self._convert_to_serializable(fig)

    def create_plots(self, df: pd.DataFrame, forecast_df: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create all plots and return them in a single dictionary."""
        try:
            # Create yield curve plot
            yield_curve_data = self.create_3d_yield_curve(
                pd.concat([df, forecast_df]), 
                len(forecast_df)
            )

            # Create spread plot
            spread_data = self.create_spread_plot(df, metrics)

            return {
                'yield_curve': yield_curve_data,
                'spread': spread_data
            }
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            raise 