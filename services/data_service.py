import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from config import Config
from services.fred_client import FredClient

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates treasury yield curve data."""
    
    REQUIRED_COLUMNS = ['1 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '5 Yr', '10 Yr', '30 Yr']
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """Validate the DataFrame structure and content."""
        if df is None or df.empty:
            return False
        return all(col in df.columns for col in DataValidator.REQUIRED_COLUMNS)

class DataService:
    """Service for managing Treasury yield curve data."""
    
    def __init__(self, fred_client: Optional[FredClient] = None):
        self.fred_client = fred_client or FredClient()
        self.data: Optional[pd.DataFrame] = None
        self._setup_data_directory()

    def _setup_data_directory(self) -> None:
        """Create data directory if it doesn't exist."""
        Config.DATA_DIR.mkdir(exist_ok=True)

    def load_data(self) -> pd.DataFrame:
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
                self.data = self._fetch_and_save_data()
            
            return self.data
        except Exception as e:
            logger.error(f"Error loading yield curve data: {str(e)}")
            raise

    def _fetch_and_save_data(self) -> pd.DataFrame:
        """Fetch fresh data from FRED and save to disk."""
        df = self.fred_client.fetch_yield_curve_data()
        df.to_parquet(Config.DATA_FILE, compression='snappy')
        logger.info("Data saved to disk")
        return df

    def filter_by_date_range(self, start_year: str, end_year: str) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        try:
            if self.data is None:
                logger.error("Data not loaded. Attempting to load now...")
                self.load_data()
                if self.data is None:
                    raise ValueError("Failed to load data")
            
            # Convert input years to dates
            start_year_float = float(start_year)
            end_year_float = float(end_year)
            
            start_year_int = int(start_year_float)
            start_month = int((start_year_float % 1) * 12) + 1
            start_date = datetime(start_year_int, start_month, 1)
            
            end_year_int = int(end_year_float)
            end_month = int((end_year_float % 1) * 12) + 1
            end_date = datetime(end_year_int, end_month, 1)
            
            logger.debug(f"Filtering data between {start_date} and {end_date}")
            
            filtered_df = self.data[(self.data.index >= start_date) & 
                                  (self.data.index <= end_date)].copy()
            
            return filtered_df
        except Exception as e:
            logger.error(f"Error in filter_by_date_range: {str(e)}")
            raise 