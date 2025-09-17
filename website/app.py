"""
DAN_G Forecasting Website
A web application for refinery yield forecasting using XGBoost and Ridge LR models.
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import the forecasting module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'orchestrator'))

from api.etl_processor import ETLProcessor
from api.forecasting_engine import ForecastingEngine
from api.interpretation_engine import InterpretationEngine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
etl_processor = ETLProcessor()
forecasting_engine = ForecastingEngine()
interpretation_engine = InterpretationEngine()

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for raw data processing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the uploaded file
            processed_data = etl_processor.process_file(filepath)
            
            return jsonify({
                'success': True,
                'message': 'File processed successfully',
                'data_info': {
                    'rows': len(processed_data),
                    'columns': list(processed_data.columns),
                    'date_range': {
                        'start': str(processed_data.index.min()),
                        'end': str(processed_data.index.max())
                    }
                }
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/forecast', methods=['POST'])
def generate_forecast():
    """Generate forecasting predictions."""
    try:
        data = request.get_json()
        
        # Get forecasting parameters
        horizon = data.get('horizon', 7)  # Default 7 days
        targets = data.get('targets', ['yield', 'quality'])
        model_type = data.get('model_type', 'ensemble')  # xgboost, ridge, ensemble
        
        # Generate forecast
        forecast_results = forecasting_engine.generate_forecast(
            horizon=horizon,
            targets=targets,
            model_type=model_type
        )
        
        # Generate interpretation
        interpretation = interpretation_engine.analyze_forecast(
            forecast_results,
            targets=targets
        )
        
        return jsonify({
            'success': True,
            'forecast': forecast_results,
            'interpretation': interpretation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Forecasting dashboard with visualizations."""
    return render_template('dashboard.html')

@app.route('/api/forecast/plot')
def forecast_plot():
    """Generate forecast plot data."""
    try:
        # Get parameters from query string
        horizon = int(request.args.get('horizon', 7))
        targets = request.args.get('targets', 'yield,quality').split(',')
        model_type = request.args.get('model_type', 'ensemble')
        
        # Generate forecast
        forecast_results = forecasting_engine.generate_forecast(
            horizon=horizon,
            targets=targets,
            model_type=model_type
        )
        
        # Create plot
        fig = create_forecast_plot(forecast_results, targets)
        
        # Convert to JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return graphJSON
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/interpretation')
def get_interpretation():
    """Get forecast interpretation."""
    try:
        targets = request.args.get('targets', 'yield,quality').split(',')
        
        # Generate sample interpretation (in real app, this would use actual data)
        interpretation = interpretation_engine.get_sample_interpretation(targets)
        
        return jsonify(interpretation)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_forecast_plot(forecast_results, targets):
    """Create forecast visualization plot."""
    fig = go.Figure()
    
    # Add historical data
    if 'historical' in forecast_results:
        for target in targets:
            if target in forecast_results['historical']:
                fig.add_trace(go.Scatter(
                    x=forecast_results['historical'][target]['dates'],
                    y=forecast_results['historical'][target]['values'],
                    mode='lines',
                    name=f'{target.title()} (Historical)',
                    line=dict(color='blue', width=2)
                ))
    
    # Add forecast data
    if 'forecast' in forecast_results:
        for target in targets:
            if target in forecast_results['forecast']:
                fig.add_trace(go.Scatter(
                    x=forecast_results['forecast'][target]['dates'],
                    y=forecast_results['forecast'][target]['values'],
                    mode='lines',
                    name=f'{target.title()} (Forecast)',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Add confidence intervals
                if 'confidence_upper' in forecast_results['forecast'][target]:
                    fig.add_trace(go.Scatter(
                        x=forecast_results['forecast'][target]['dates'],
                        y=forecast_results['forecast'][target]['confidence_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_results['forecast'][target]['dates'],
                        y=forecast_results['forecast'][target]['confidence_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.2)',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    fig.update_layout(
        title='Refinery Yield Forecasting - Next 7 Days',
        xaxis_title='Date',
        yaxis_title='Yield/Quality Metrics',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

