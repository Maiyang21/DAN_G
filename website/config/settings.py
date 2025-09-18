"""
Configuration settings for the DAN_G Refinery Forecasting Platform
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Basic Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///refinery_forecasting.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API Configuration
    API_KEY = os.environ.get('API_KEY') or 'dev-api-key-change-in-production'
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = 'logs/app.log'
    
    # Forecasting model settings
    MODEL_UPDATE_INTERVAL = 3600  # 1 hour in seconds
    FORECAST_HORIZON_DAYS = 7
    CONFIDENCE_INTERVAL = 0.95
    
    # Real-time monitoring settings
    MONITORING_INTERVAL = 30  # seconds
    METRICS_RETENTION_DAYS = 30
    
    # Alert settings
    ALERT_THRESHOLDS = {
        'cpu_usage': 80,
        'memory_usage': 85,
        'forecast_accuracy': 0.7,
        'data_quality': 0.8
    }
    
    # Security settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'memory://')
    
    # Email settings (for alerts)
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # External API settings
    EIA_API_KEY = os.environ.get('EIA_API_KEY')
    WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')
    
    # Model training settings
    TRAINING_BATCH_SIZE = 32
    TRAINING_EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    
    # Performance settings
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///refinery_forecasting_dev.db'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://user:pass@localhost/refinery_forecasting'
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Logging
    LOG_LEVEL = 'WARNING'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
