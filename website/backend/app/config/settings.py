"""
Production Configuration for DAN_G Platform
Railway deployment with AWS integration
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Basic Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database configuration (Railway PostgreSQL)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///refinery_forecasting.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session Configuration
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # CORS Configuration
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
    AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET', 'dan-g-refinery-data')
    AWS_SAGEMAKER_ENDPOINT = os.environ.get('AWS_SAGEMAKER_ENDPOINT', 'dan-g-forecasting-endpoint')
    
    # Redis Configuration (Railway Redis)
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}
    
    # Session configuration
    SESSION_TYPE = 'redis'
    SESSION_REDIS = REDIS_URL
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_KEY_PREFIX = 'dan_g:'
    
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
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_STORAGE_URL = REDIS_URL
    
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
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = REDIS_URL
    CACHE_DEFAULT_TIMEOUT = 300

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///refinery_forecasting_dev.db'
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:5000']

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Logging
    LOG_LEVEL = 'WARNING'
    
    # Rate limiting
    RATE_LIMIT_ENABLED = True

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
    'default': ProductionConfig
}
