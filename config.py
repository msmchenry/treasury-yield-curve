from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration."""
    
    # API Configuration
    FRED_API_KEY = os.environ.get('FRED_API_KEY')
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY environment variable is not set")
    
    # Data Storage
    DATA_DIR = Path('data')
    DATA_FILE = DATA_DIR / 'treasury_yields.parquet'
    
    # Data Configuration
    START_DATE = datetime(2004, 1, 1)
    
    # FRED Series IDs
    SERIES_IDS = {
        '1 Mo': 'DGS1MO',
        '3 Mo': 'DGS3MO',
        '6 Mo': 'DGS6MO',
        '1 Yr': 'DGS1',
        '2 Yr': 'DGS2',
        '5 Yr': 'DGS5',
        '10 Yr': 'DGS10',
        '30 Yr': 'DGS30'
    }
    
    # Maturity mapping for visualization
    MATURITY_MAP = {
        '1 Mo': 1/12,
        '3 Mo': 3/12,
        '6 Mo': 6/12,
        '1 Yr': 1,
        '2 Yr': 2,
        '5 Yr': 5,
        '10 Yr': 10,
        '30 Yr': 30
    } 