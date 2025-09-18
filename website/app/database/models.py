"""
Database models for the DAN_G Refinery Forecasting Platform
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication and user management"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    role = db.Column(db.String(20), default='user')  # user, admin, analyst
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    forecasts = db.relationship('ForecastHistory', backref='user', lazy='dynamic')
    alerts = db.relationship('Alert', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

class ForecastHistory(db.Model):
    """Model for storing forecast history and results"""
    __tablename__ = 'forecast_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    forecast_name = db.Column(db.String(200))
    forecast_type = db.Column(db.String(50))  # weekly, monthly, custom
    input_data_hash = db.Column(db.String(64))  # Hash of input data for deduplication
    forecast_data = db.Column(db.Text)  # JSON string of forecast results
    interpretations = db.Column(db.Text)  # JSON string of interpretations
    accuracy_score = db.Column(db.Float)
    confidence_score = db.Column(db.Float)
    model_used = db.Column(db.String(50))  # xgboost, ridge, ensemble
    parameters = db.Column(db.Text)  # JSON string of model parameters
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert forecast to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'forecast_name': self.forecast_name,
            'forecast_type': self.forecast_type,
            'accuracy_score': self.accuracy_score,
            'confidence_score': self.confidence_score,
            'model_used': self.model_used,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Alert(db.Model):
    """Model for system alerts and notifications"""
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    alert_type = db.Column(db.String(50), nullable=False)  # system, forecast, data_quality
    severity = db.Column(db.String(20), nullable=False)  # low, medium, high, critical
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_read = db.Column(db.Boolean, default=False)
    metadata = db.Column(db.Text)  # JSON string for additional data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime)
    
    def to_dict(self):
        """Convert alert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'title': self.title,
            'message': self.message,
            'is_active': self.is_active,
            'is_read': self.is_read,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

class SystemMetrics(db.Model):
    """Model for storing system performance metrics"""
    __tablename__ = 'system_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    cpu_usage = db.Column(db.Float)
    memory_usage = db.Column(db.Float)
    disk_usage = db.Column(db.Float)
    active_forecasts = db.Column(db.Integer, default=0)
    active_users = db.Column(db.Integer, default=0)
    data_quality_score = db.Column(db.Float)
    model_accuracy = db.Column(db.Float)
    response_time = db.Column(db.Float)  # Average response time in ms
    error_rate = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert metrics to dictionary"""
        return {
            'id': self.id,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'disk_usage': self.disk_usage,
            'active_forecasts': self.active_forecasts,
            'active_users': self.active_users,
            'data_quality_score': self.data_quality_score,
            'model_accuracy': self.model_accuracy,
            'response_time': self.response_time,
            'error_rate': self.error_rate,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class ModelPerformance(db.Model):
    """Model for tracking model performance over time"""
    __tablename__ = 'model_performance'
    
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50), nullable=False)
    model_version = db.Column(db.String(20))
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    mae = db.Column(db.Float)  # Mean Absolute Error
    mse = db.Column(db.Float)  # Mean Squared Error
    rmse = db.Column(db.Float)  # Root Mean Squared Error
    r2_score = db.Column(db.Float)
    training_time = db.Column(db.Float)  # Training time in seconds
    prediction_time = db.Column(db.Float)  # Average prediction time in ms
    data_size = db.Column(db.Integer)  # Size of training data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert model performance to dictionary"""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mae': self.mae,
            'mse': self.mse,
            'rmse': self.rmse,
            'r2_score': self.r2_score,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'data_size': self.data_size,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class DataQualityMetrics(db.Model):
    """Model for tracking data quality metrics"""
    __tablename__ = 'data_quality_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    dataset_name = db.Column(db.String(200))
    completeness_score = db.Column(db.Float)  # Percentage of non-null values
    consistency_score = db.Column(db.Float)  # Data consistency score
    accuracy_score = db.Column(db.Float)  # Data accuracy score
    validity_score = db.Column(db.Float)  # Data validity score
    uniqueness_score = db.Column(db.Float)  # Duplicate data score
    overall_score = db.Column(db.Float)  # Overall data quality score
    missing_values = db.Column(db.Integer)
    duplicate_rows = db.Column(db.Integer)
    outlier_count = db.Column(db.Integer)
    data_size = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert data quality metrics to dictionary"""
        return {
            'id': self.id,
            'dataset_name': self.dataset_name,
            'completeness_score': self.completeness_score,
            'consistency_score': self.consistency_score,
            'accuracy_score': self.accuracy_score,
            'validity_score': self.validity_score,
            'uniqueness_score': self.uniqueness_score,
            'overall_score': self.overall_score,
            'missing_values': self.missing_values,
            'duplicate_rows': self.duplicate_rows,
            'outlier_count': self.outlier_count,
            'data_size': self.data_size,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
