#!/usr/bin/env python3
"""
DAN_G Refinery Forecasting Platform - Production Runner
"""

import os
import sys
import logging
from app import app, socketio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'uploads', 'models', 'static/uploads']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def main():
    """Main application runner"""
    try:
        # Create necessary directories
        create_directories()
        
        # Get configuration from environment
        debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        host = os.environ.get('FLASK_HOST', '0.0.0.0')
        port = int(os.environ.get('FLASK_PORT', 5000))
        
        logger.info(f"Starting DAN_G Refinery Forecasting Platform...")
        logger.info(f"Debug mode: {debug}")
        logger.info(f"Host: {host}")
        logger.info(f"Port: {port}")
        
        # Run the application
        socketio.run(
            app,
            debug=debug,
            host=host,
            port=port,
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()