from fredapi import Fred
import pandas as pd
import logging
from config import Config
from typing import Dict

logger = logging.getLogger(__name__)

class FredClient:
    """Client for interacting with the FRED API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.FRED_API_KEY
        self.fred = Fred(api_key=self.api_key)
    
    def fetch_yield_curve_data(self) -> pd.DataFrame:
        """Fetch all available yield curve data from FRED API."""
        logger.info("Fetching all historical data from FRED")
        try:
            df_dict: Dict[str, pd.Series] = {}
            for label, series_id in Config.SERIES_IDS.items():
                logger.debug(f"Fetching data for {label} (Series ID: {series_id})")
                data = self.fred.get_series(series_id, observation_start='1900-01-01')
                df_dict[label] = data

            df = pd.DataFrame(df_dict)
            df = df.dropna()  # Remove dates where any maturity is missing
            df = df.resample('M').last()  # Resample to monthly frequency
            
            logger.info(f"Fetched data from {df.index.min()} to {df.index.max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from FRED: {e}")
            raise 