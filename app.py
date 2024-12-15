from flask import Flask, render_template, request
from services.treasury_service import TreasuryService
import logging
from datetime import datetime
from config import Config
from typing import Tuple, Union
from http import HTTPStatus
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize treasury service
treasury_service = TreasuryService()

# Load initial data
logger.info("Loading initial yield curve data...")
try:
    treasury_service.load_data()
    if treasury_service.data is None:
        raise ValueError("Failed to initialize data")
    logger.info(f"Data loaded successfully. Shape: {treasury_service.data.shape}")
except Exception as e:
    logger.error(f"Error loading initial data: {str(e)}")
    raise

def validate_year_input(year: str) -> bool:
    """Validate that the year input is within acceptable range."""
    try:
        year_float = float(year)
        min_date = treasury_service.data.index.min()
        max_date = treasury_service.data.index.max()
        min_year = min_date.year + min_date.month/12
        max_year = max_date.year + max_date.month/12
        return min_year <= year_float <= max_year
    except ValueError:
        return False

@app.route('/')
def index() -> Union[str, Tuple[str, int]]:
    """Main route for displaying treasury yield curves."""
    logger.debug("Index route accessed")
    try:
        # Ensure data is loaded
        if treasury_service.data is None:
            logger.info("Data not loaded, attempting to reload...")
            treasury_service.load_data()
            
        if treasury_service.data is None:
            return "Unable to load Treasury data. Please try again later.", HTTPStatus.SERVICE_UNAVAILABLE
        
        # Get the full date range from the data
        min_date = treasury_service.data.index.min()
        max_date = treasury_service.data.index.max()
        
        # Convert dates to decimal years for the slider
        min_year = min_date.year + (min_date.month - 1)/12
        max_year = max_date.year + (max_date.month - 1)/12
        
        # Default to last 2 quarters if no dates specified
        current_date = datetime.now()
        current_year = current_date.year
        current_quarter = (current_date.month - 1) // 3 + 1
        
        default_end = min(current_year + current_quarter * 0.25, max_year)
        default_start = max(default_end - 0.5, min_year)
        
        # Get requested date range
        start_year = request.args.get('start_year', str(default_start))
        end_year = request.args.get('end_year', str(default_end))

        # Simple validation - just ensure the values are within bounds
        try:
            start_year_float = float(start_year)
            end_year_float = float(end_year)
            if start_year_float < min_year:
                start_year = str(min_year)
            if end_year_float > max_year:
                end_year = str(max_year)
        except ValueError:
            start_year = str(default_start)
            end_year = str(default_end)

        plot_data = treasury_service.get_yield_curve_plot(start_year, end_year)
        metrics_fig = treasury_service.plot_metrics()
        metrics_data = json.dumps(metrics_fig.to_dict())

        return render_template('index.html', 
                             plot_data=plot_data,
                             metrics_plot=metrics_data,
                             start_year=start_year,
                             end_year=end_year,
                             min_year=min_year,
                             max_year=max_year)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return f"An error occurred: {str(e)}", HTTPStatus.INTERNAL_SERVER_ERROR

@app.errorhandler(Exception)
def handle_error(error: Exception) -> Tuple[str, int]:
    """Global error handler."""
    logger.error(f"Unhandled error: {error}")
    return "An error occurred", HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    app.run(debug=True) 