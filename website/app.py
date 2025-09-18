"""
DAN_G Refinery Forecasting Platform - Production Ready
A dynamic, real-time refinery forecasting and optimization platform
"""

import os
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import json
from functools import wraps
import traceback

# Import custom modules
from app.api.etl_processor import ETLProcessor
from app.api.forecasting_engine import ForecastingEngine
from app.api.interpretation_engine import InterpretationEngine
from app.api.real_time_monitor import RealTimeMonitor
from app.api.alert_system import AlertSystem
from app.database.models import db, User, ForecastHistory, Alert, SystemMetrics
from config.settings import Config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

# Initialize custom modules
etl_processor = ETLProcessor()
forecasting_engine = ForecastingEngine()
interpretation_engine = InterpretationEngine()
real_time_monitor = RealTimeMonitor()
alert_system = AlertSystem()

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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def require_api_key(f):
    """Decorator to require API key for certain endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != app.config['API_KEY']:
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with real-time data"""
    try:
        # Get recent forecasts
        recent_forecasts = ForecastHistory.query.filter_by(
            user_id=current_user.id
        ).order_by(ForecastHistory.created_at.desc()).limit(10).all()
        
        # Get system metrics
        metrics = SystemMetrics.query.order_by(
            SystemMetrics.timestamp.desc()
        ).limit(100).all()
        
        # Get active alerts
        active_alerts = Alert.query.filter_by(
            is_active=True
        ).order_by(Alert.created_at.desc()).limit(5).all()
        
        return render_template('dashboard.html', 
                             forecasts=recent_forecasts,
                             metrics=metrics,
                             alerts=active_alerts)
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        flash('Error loading dashboard data', 'error')
        return render_template('dashboard.html')

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_data():
    """Handle data upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Process the file
        processed_data = etl_processor.process_file(file)
        
        # Store processing result
        session['processed_data'] = processed_data.to_json()
        
        return jsonify({
            'success': True,
            'message': 'Data processed successfully',
            'data_shape': processed_data.shape,
            'columns': list(processed_data.columns)
        })
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/forecast', methods=['POST'])
@login_required
def generate_forecast():
    """Generate forecasting predictions"""
    try:
        data_json = session.get('processed_data')
        if not data_json:
            return jsonify({'error': 'No processed data available'}), 400
        
        data = pd.read_json(data_json)
        
        # Generate forecast
        forecast_results = forecasting_engine.generate_forecast(data)
        
        # Generate interpretations
        interpretations = interpretation_engine.analyze_forecast(
            data, forecast_results
        )
        
        # Store forecast in database
        forecast_record = ForecastHistory(
            user_id=current_user.id,
            forecast_data=forecast_results.to_json(),
            interpretations=json.dumps(interpretations),
            created_at=datetime.utcnow()
        )
        db.session.add(forecast_record)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'forecast': forecast_results.to_dict(),
            'interpretations': interpretations
        })
    
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        return jsonify({'error': f'Forecast generation failed: {str(e)}'}), 500

@app.route('/api/real-time-data')
@login_required
def get_real_time_data():
    """Get real-time monitoring data"""
    try:
        data = real_time_monitor.get_current_metrics()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Real-time data error: {str(e)}")
        return jsonify({'error': 'Failed to fetch real-time data'}), 500

@app.route('/api/alerts')
@login_required
def get_alerts():
    """Get system alerts"""
    try:
        alerts = Alert.query.filter_by(is_active=True).all()
        return jsonify([alert.to_dict() for alert in alerts])
    except Exception as e:
        logger.error(f"Alerts error: {str(e)}")
        return jsonify({'error': 'Failed to fetch alerts'}), 500

@app.route('/api/optimize', methods=['POST'])
@login_required
def optimize_parameters():
    """Optimize refinery parameters"""
    try:
        data = request.get_json()
        optimization_result = forecasting_engine.optimize_parameters(data)
        
        return jsonify({
            'success': True,
            'optimization': optimization_result
        })
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        return jsonify({'error': f'Optimization failed: {str(e)}'}), 500

# WebSocket events for real-time updates
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to DAN_G platform'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('request_update')
def handle_update_request():
    """Handle real-time update requests"""
    try:
        if current_user.is_authenticated:
            # Get latest metrics
            metrics = real_time_monitor.get_current_metrics()
            emit('metrics_update', metrics)
            
            # Get latest alerts
            alerts = Alert.query.filter_by(is_active=True).all()
            emit('alerts_update', [alert.to_dict() for alert in alerts])
    except Exception as e:
        logger.error(f"Update request error: {str(e)}")
        emit('error', {'message': 'Failed to fetch updates'})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('errors/500.html'), 500

# Background tasks
def background_monitoring():
    """Background task for continuous monitoring"""
    while True:
        try:
            # Update system metrics
            metrics = real_time_monitor.collect_metrics()
            system_metric = SystemMetrics(
                cpu_usage=metrics.get('cpu_usage', 0),
                memory_usage=metrics.get('memory_usage', 0),
                active_forecasts=metrics.get('active_forecasts', 0),
                timestamp=datetime.utcnow()
            )
            db.session.add(system_metric)
            db.session.commit()
            
            # Check for alerts
            alert_system.check_alerts(metrics)
            
            # Emit updates to connected clients
            socketio.emit('metrics_update', metrics)
            
        except Exception as e:
            logger.error(f"Background monitoring error: {str(e)}")
        
        socketio.sleep(30)  # Update every 30 seconds

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Start background monitoring
    socketio.start_background_task(background_monitoring)
    
    # Run the application
    socketio.run(app, debug=app.config['DEBUG'], host='0.0.0.0', port=5000)