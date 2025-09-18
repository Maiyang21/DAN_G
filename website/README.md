# DAN_G Refinery Forecasting Platform

A dynamic, production-ready web application for refinery forecasting and optimization using advanced machine learning models.

## Features

### 🚀 Dynamic & Real-time
- **Real-time Monitoring**: Live system metrics, performance tracking, and health monitoring
- **WebSocket Integration**: Real-time updates and notifications
- **Interactive Dashboard**: Dynamic charts and visualizations with Plotly
- **Auto-refresh**: Automatic data updates and system status monitoring

### 🤖 AI-Powered Forecasting
- **Ensemble Models**: XGBoost, Ridge Regression, and Random Forest
- **Advanced ETL**: Comprehensive data processing with quality monitoring
- **Feature Engineering**: Automatic feature creation and optimization
- **Interpretation Engine**: Detailed analysis of crude composition and operating parameter effects

### 📊 Production-Ready Features
- **Database Integration**: SQLAlchemy with user management and audit trails
- **Alert System**: Comprehensive monitoring with email notifications
- **Security**: Authentication, input validation, and secure API endpoints
- **Scalability**: Modular architecture with background task processing
- **Responsive Design**: Modern UI that works on all devices

## Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Maiyang21/DAN_G.git
   cd DAN_G/website
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables** (optional)
   ```bash
   export FLASK_DEBUG=False
   export FLASK_HOST=0.0.0.0
   export FLASK_PORT=5000
   export SECRET_KEY=your-secret-key
   export DATABASE_URL=sqlite:///refinery_forecasting.db
   ```

4. **Run the application**
   ```bash
   python run_website.py
   ```

5. **Access the platform**
   - Open your browser to `http://localhost:5000`
   - Use demo credentials: `admin` / `admin123`

## Usage

### 1. Data Upload
- Upload CSV, Excel, or JSON files containing refinery data
- Automatic data validation and quality assessment
- Real-time processing status updates

### 2. Generate Forecasts
- Click "Generate Forecast" to create 7-day yield predictions
- View interactive charts with confidence intervals
- Analyze feature importance and driving factors

### 3. Monitor System
- Real-time system health monitoring
- Performance metrics and alerts
- Background task processing

### 4. Analyze Results
- Comprehensive interpretation of forecast results
- Crude composition and operating parameter analysis
- Actionable recommendations for optimization

## Architecture

### Backend Components
- **Flask Application**: Main web framework with SocketIO for real-time features
- **Database Models**: User management, forecast history, alerts, and metrics
- **API Modules**:
  - `etl_processor.py`: Advanced data processing and validation
  - `forecasting_engine.py`: ML models and ensemble methods
  - `interpretation_engine.py`: Analysis and insights generation
  - `real_time_monitor.py`: System monitoring and metrics
  - `alert_system.py`: Alerting and notification system

### Frontend Components
- **Responsive Dashboard**: Bootstrap-based UI with real-time updates
- **Interactive Charts**: Plotly visualizations for forecasts and metrics
- **WebSocket Client**: Real-time data streaming and notifications
- **Progressive Enhancement**: Works without JavaScript for basic functionality

## Configuration

### Environment Variables
```bash
# Flask Configuration
FLASK_DEBUG=False
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# Security
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# Database
DATABASE_URL=sqlite:///refinery_forecasting.db

# Email (for alerts)
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password

# External APIs
EIA_API_KEY=your-eia-api-key
WEATHER_API_KEY=your-weather-api-key
```

### Database Configuration
The application supports multiple database backends:
- **SQLite** (default): For development and small deployments
- **PostgreSQL**: For production deployments
- **MySQL**: Alternative production option

## API Endpoints

### Authentication
- `POST /login` - User authentication
- `POST /logout` - User logout

### Data Processing
- `POST /api/upload` - Upload and process data files
- `POST /api/forecast` - Generate forecasting predictions
- `POST /api/optimize` - Optimize refinery parameters

### Monitoring
- `GET /api/real-time-data` - Get current system metrics
- `GET /api/alerts` - Get system alerts
- `GET /api/health` - System health check

### WebSocket Events
- `connect` - Client connection
- `disconnect` - Client disconnection
- `request_update` - Request real-time updates
- `metrics_update` - System metrics update
- `alerts_update` - Alerts update

## Deployment

### Development
```bash
python run_website.py
```

### Production with Gunicorn
```bash
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "run_website.py"]
```

## Monitoring & Alerts

### System Metrics
- CPU and memory usage
- Disk space utilization
- Active forecasts and users
- Response times and error rates

### Alert Types
- **System Alerts**: High resource usage, disk space
- **Performance Alerts**: Slow response times, high error rates
- **Data Quality Alerts**: Low data quality scores
- **Forecast Alerts**: Low accuracy, model drift

### Notification Channels
- Email notifications
- Webhook integrations
- SMS alerts (configurable)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

## Changelog

### Version 2.0.0 (Current)
- Complete rewrite with dynamic, production-ready features
- Real-time monitoring and WebSocket integration
- Advanced ETL processing and data quality monitoring
- Comprehensive alert system and notifications
- Responsive UI with interactive visualizations
- Database integration and user management
- Enhanced security and API endpoints

### Version 1.0.0
- Initial release with basic forecasting functionality
- Simple web interface
- Basic ML models (XGBoost, Ridge Regression)