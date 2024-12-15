from flask import Flask, render_template, request
from services.treasury_service import TreasuryService
import logging
from datetime import datetime
from config import Config
from typing import Tuple, Union
from http import HTTPStatus

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize treasury service
treasury_service = TreasuryService()

def validate_year_input(year: str) -> bool:
    """Validate that the year input is within acceptable range."""
    try:
        year_float = float(year)
        return 2004 <= year_float <= datetime.now().year + 1
    except ValueError:
        return False

@app.route('/')
def index() -> Union[str, Tuple[str, int]]:
    """Main route for displaying treasury yield curves."""
    logger.debug("Index route accessed")
    try:
        # Default to last 2 quarters if no dates specified
        current_date = datetime.now()
        current_year = current_date.year
        current_quarter = (current_date.month - 1) // 3 + 1
        
        default_end = current_year + current_quarter * 0.25
        default_start = current_year + (current_quarter - 2) * 0.25
        
        start_year = request.args.get('start_year', str(default_start))
        end_year = request.args.get('end_year', str(default_end))

        # Validate inputs
        if not all(validate_year_input(y) for y in [start_year, end_year]):
            return "Invalid year parameters", HTTPStatus.BAD_REQUEST

        plot_data = treasury_service.get_yield_curve_plot(start_year, end_year)
        return render_template('index.html', 
                             plot_data=plot_data,
                             start_year=start_year,
                             end_year=end_year)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return f"An error occurred: {str(e)}", HTTPStatus.INTERNAL_SERVER_ERROR

@app.errorhandler(Exception)
def handle_error(error: Exception) -> Tuple[str, int]:
    """Global error handler."""
    logger.error(f"Unhandled error: {error}")
    return "An error occurred", HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    # Load initial data
    logger.info("Loading initial yield curve data...")
    treasury_service.load_data()
    logger.info("Data loaded successfully")
    
    app.run(debug=True) 