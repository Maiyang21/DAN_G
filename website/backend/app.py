"""
DAN_G Refinery Forecasting Platform - Backend API
Production-ready Flask API for Railway deployment with AWS integration
"""

import os
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
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
from app.api.auth_service import AuthService
from app.database.models import db, User, ForecastHistory, Alert, SystemMetrics
from app.config.settings import Config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
jwt = JWTManager(app)
CORS(app, origins=os.environ.get('CORS_ORIGINS', '*').split(','))
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize custom services
aws_client = AWSClient()
etl_processor = ETLProcessor()
forecasting_engine = ForecastingEngine()
interpretation_engine = InterpretationEngine()
auth_service = AuthService()

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

# JWT error handlers
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token has expired'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({'error': 'Invalid token'}), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({'error': 'Authorization token required'}), 401

# Authentication routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if user exists
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 409
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 409
        
        # Create user
        user = User(
            username=data['username'],
            email=data['email'],
            first_name=data.get('first_name', ''),
            last_name=data.get('last_name', ''),
            role=data.get('role', 'user')
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        # Generate JWT token
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'access_token': access_token,
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        
        if not data or not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Username and password required'}), 400
        
        # Find user
        user = User.query.filter_by(username=data['username']).first()
        
        if not user or not user.check_password(data['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is disabled'}), 401
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Generate JWT token
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'access_token': access_token,
            'user': user.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current user information"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'success': True,
            'user': user.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Get user error: {str(e)}")
        return jsonify({'error': 'Failed to get user information'}), 500

# Data processing routes
@app.route('/api/upload', methods=['POST'])
@jwt_required()
def upload_data():
    """Handle data upload and processing"""
    try:
        user_id = get_jwt_identity()
        
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
@jwt_required()
def generate_forecast():
    """Generate forecasting predictions using AWS SageMaker"""
    try:
        user_id = get_jwt_identity()
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
@jwt_required()
def get_forecast_history():
    """Get user's forecast history"""
    try:
        user_id = get_jwt_identity()
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
@jwt_required()
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
@jwt_required()
def handle_update_request():
    """Handle real-time update requests"""
    try:
        user_id = get_jwt_identity()
        
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
