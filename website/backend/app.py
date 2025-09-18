"""
DAN_G Refinery Forecasting Platform - Backend API
Production-ready Flask API for Railway deployment with AWS integration
"""

import os
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
# Removed JWT authentication - using simple session-based auth
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
import boto3
import pandas as pd
import numpy as np
import json
from functools import wraps
import traceback

# Import custom modules
from app.api.aws_client import AWSClient
from app.api.etl_processor import ETLProcessor
from app.api.forecasting_engine import ForecastingEngine
from app.api.interpretation_engine import InterpretationEngine
from app.api.simple_auth import SimpleAuth, require_auth, require_auth_optional
from app.database.models import db, User, ForecastHistory, Alert, SystemMetrics
from app.config.settings import Config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
CORS(app, origins=os.environ.get('CORS_ORIGINS', '*').split(','))
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize custom services
aws_client = AWSClient()
etl_processor = ETLProcessor()
forecasting_engine = ForecastingEngine()
interpretation_engine = InterpretationEngine()
auth_service = SimpleAuth()

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

# Authentication routes will use the imported require_auth decorator

# Authentication routes
@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login with simple authentication"""
    try:
        data = request.get_json()
        
        if not data or not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Username and password required'}), 400
        
        # Authenticate user
        auth_result = auth_service.authenticate_user(data['username'], data['password'])
        
        if not auth_result['success']:
            return jsonify({'error': auth_result['error']}), 401
        
        # Create session
        if auth_service.create_session(auth_result['user']):
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user': auth_result['user']
            })
        else:
            return jsonify({'error': 'Session creation failed'}), 500
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """User logout"""
    try:
        if auth_service.logout_user():
            return jsonify({
                'success': True,
                'message': 'Logout successful'
            })
        else:
            return jsonify({'error': 'Logout failed'}), 500
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({'error': 'Logout failed'}), 500

@app.route('/api/auth/me', methods=['GET'])
@require_auth
def get_current_user():
    """Get current user information"""
    try:
        user = auth_service.get_current_user()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'success': True,
            'user': user
        })
        
    except Exception as e:
        logger.error(f"Get user error: {str(e)}")
        return jsonify({'error': 'Failed to get user information'}), 500

# Data processing routes
@app.route('/api/upload', methods=['POST'])
@require_auth
def upload_data():
    """Handle data upload and processing"""
    try:
        # For demo purposes, use a default user ID
        user_id = 'demo-user'
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Process the file
        processed_data = etl_processor.process_file(file)
        
        # Upload to S3
        s3_key = f"processed_data/{user_id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        aws_client.upload_to_s3(processed_data, s3_key)
        
        # Store processing result in session
        session['processed_data_key'] = s3_key
        
        return jsonify({
            'success': True,
            'message': 'Data processed and uploaded to S3',
            's3_key': s3_key,
            'data_shape': processed_data.shape,
            'columns': list(processed_data.columns)
        })
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/forecast', methods=['POST'])
@require_auth
def generate_forecast():
    """Generate forecasting predictions using AWS SageMaker"""
    try:
        # For demo purposes, use a default user ID
        user_id = 'demo-user'
        s3_key = session.get('processed_data_key')
        
        if not s3_key:
            return jsonify({'error': 'No processed data available'}), 400
        
        # Download data from S3
        data = aws_client.download_from_s3(s3_key)
        
        # Generate forecast using AWS SageMaker
        forecast_results = aws_client.invoke_sagemaker_model(data)
        
        # Generate interpretations
        interpretations = interpretation_engine.analyze_forecast(data, forecast_results)
        
        # Store forecast in database
        forecast_record = ForecastHistory(
            user_id=user_id,
            forecast_name=f"Forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            forecast_type='weekly',
            input_data_hash=hash(str(data.values.tobytes())),
            forecast_data=forecast_results.to_json(),
            interpretations=json.dumps(interpretations),
            model_used='sagemaker',
            created_at=datetime.utcnow()
        )
        db.session.add(forecast_record)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'forecast': forecast_results.to_dict('records'),
            'interpretations': interpretations
        })
    
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        return jsonify({'error': f'Forecast generation failed: {str(e)}'}), 500

@app.route('/api/forecast/history', methods=['GET'])
@require_auth
def get_forecast_history():
    """Get user's forecast history"""
    try:
        # For demo purposes, use a default user ID
        user_id = 'demo-user'
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        forecasts = ForecastHistory.query.filter_by(user_id=user_id)\
            .order_by(ForecastHistory.created_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            'success': True,
            'forecasts': [forecast.to_dict() for forecast in forecasts.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': forecasts.total,
                'pages': forecasts.pages
            }
        })
        
    except Exception as e:
        logger.error(f"Forecast history error: {str(e)}")
        return jsonify({'error': 'Failed to fetch forecast history'}), 500

# System monitoring routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db.session.execute('SELECT 1')
        
        # Check AWS connection
        aws_status = aws_client.health_check()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'database': 'connected',
            'aws': aws_status,
            'version': '2.0.0'
        })
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/metrics', methods=['GET'])
@require_auth
def get_system_metrics():
    """Get system metrics"""
    try:
        metrics = SystemMetrics.query.order_by(
            SystemMetrics.timestamp.desc()
        ).limit(100).all()
        
        return jsonify({
            'success': True,
            'metrics': [metric.to_dict() for metric in metrics]
        })
        
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return jsonify({'error': 'Failed to fetch metrics'}), 500

# WebSocket events
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
@require_auth
def handle_update_request():
    """Handle real-time update requests"""
    try:
        # For demo purposes, use a default user ID
        user_id = 'demo-user'
        
        # Get latest metrics
        latest_metric = SystemMetrics.query.order_by(
            SystemMetrics.timestamp.desc()
        ).first()
        
        if latest_metric:
            emit('metrics_update', latest_metric.to_dict())
        
        # Get user's recent forecasts
        recent_forecasts = ForecastHistory.query.filter_by(user_id=user_id)\
            .order_by(ForecastHistory.created_at.desc())\
            .limit(5).all()
        
        emit('forecasts_update', [f.to_dict() for f in recent_forecasts])
        
    except Exception as e:
        logger.error(f"Update request error: {str(e)}")
        emit('error', {'message': 'Failed to fetch updates'})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

# Background tasks
def background_monitoring():
    """Background task for continuous monitoring"""
    while True:
        try:
            # Update system metrics
            metrics = {
                'cpu_usage': 0,  # Placeholder - implement actual monitoring
                'memory_usage': 0,
                'active_forecasts': ForecastHistory.query.count(),
                'active_users': User.query.filter_by(is_active=True).count(),
                'timestamp': datetime.utcnow()
            }
            
            system_metric = SystemMetrics(**metrics)
            db.session.add(system_metric)
            db.session.commit()
            
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
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, debug=False, host='0.0.0.0', port=port)
