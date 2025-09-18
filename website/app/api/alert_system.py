"""
Alert System for DAN_G Platform
Handles system alerts, notifications, and automated responses
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Alert types"""
    SYSTEM = "system"
    FORECAST = "forecast"
    DATA_QUALITY = "data_quality"
    PERFORMANCE = "performance"
    SECURITY = "security"

class AlertSystem:
    """Comprehensive alert system for monitoring and notifications"""
    
    def __init__(self):
        self.alert_rules = self._initialize_alert_rules()
        self.alert_history = []
        self.notification_channels = {
            'email': True,
            'webhook': False,
            'sms': False
        }
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': None,
            'password': None
        }
        
    def _initialize_alert_rules(self) -> Dict:
        """Initialize alert rules and thresholds"""
        return {
            'cpu_usage': {
                'threshold': 80,
                'severity': AlertSeverity.HIGH,
                'type': AlertType.SYSTEM,
                'message_template': "High CPU usage detected: {value}% (threshold: {threshold}%)"
            },
            'memory_usage': {
                'threshold': 85,
                'severity': AlertSeverity.HIGH,
                'type': AlertType.SYSTEM,
                'message_template': "High memory usage detected: {value}% (threshold: {threshold}%)"
            },
            'disk_usage': {
                'threshold': 90,
                'severity': AlertSeverity.CRITICAL,
                'type': AlertType.SYSTEM,
                'message_template': "Critical disk usage: {value}% (threshold: {threshold}%)"
            },
            'response_time': {
                'threshold': 5000,  # 5 seconds
                'severity': AlertSeverity.MEDIUM,
                'type': AlertType.PERFORMANCE,
                'message_template': "Slow response time: {value}ms (threshold: {threshold}ms)"
            },
            'error_rate': {
                'threshold': 0.05,  # 5%
                'severity': AlertSeverity.HIGH,
                'type': AlertType.PERFORMANCE,
                'message_template': "High error rate: {value:.2%} (threshold: {threshold:.2%})"
            },
            'data_quality': {
                'threshold': 0.7,
                'severity': AlertSeverity.MEDIUM,
                'type': AlertType.DATA_QUALITY,
                'message_template': "Low data quality: {value:.3f} (threshold: {threshold:.3f})"
            },
            'forecast_accuracy': {
                'threshold': 0.7,
                'severity': AlertSeverity.MEDIUM,
                'type': AlertType.FORECAST,
                'message_template': "Low forecast accuracy: {value:.3f} (threshold: {threshold:.3f})"
            },
            'model_drift': {
                'threshold': 0.1,
                'severity': AlertSeverity.HIGH,
                'type': AlertType.FORECAST,
                'message_template': "Model drift detected: {value:.3f} (threshold: {threshold:.3f})"
            }
        }
    
    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check metrics against alert rules and generate alerts"""
        alerts = []
        
        try:
            for metric_name, value in metrics.items():
                if metric_name in self.alert_rules:
                    rule = self.alert_rules[metric_name]
                    
                    # Check if threshold is exceeded
                    if self._is_threshold_exceeded(value, rule['threshold']):
                        alert = self._create_alert(metric_name, value, rule)
                        alerts.append(alert)
                        
                        # Process the alert
                        self._process_alert(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
            return []
    
    def _is_threshold_exceeded(self, value: float, threshold: float) -> bool:
        """Check if a value exceeds the threshold"""
        return value > threshold
    
    def _create_alert(self, metric_name: str, value: float, rule: Dict) -> Dict:
        """Create an alert object"""
        alert = {
            'id': f"{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'metric_name': metric_name,
            'value': value,
            'threshold': rule['threshold'],
            'severity': rule['severity'].value,
            'type': rule['type'].value,
            'title': f"{rule['type'].value.title()} Alert: {metric_name}",
            'message': rule['message_template'].format(
                value=value,
                threshold=rule['threshold']
            ),
            'timestamp': datetime.now().isoformat(),
            'is_active': True,
            'is_acknowledged': False,
            'is_resolved': False
        }
        
        return alert
    
    def _process_alert(self, alert: Dict):
        """Process and handle an alert"""
        try:
            # Log the alert
            logger.warning(f"ALERT: {alert['message']}")
            
            # Store in alert history
            self.alert_history.append(alert)
            
            # Send notifications based on severity
            if alert['severity'] in ['high', 'critical']:
                self._send_notifications(alert)
            
            # Trigger automated responses
            self._trigger_automated_response(alert)
            
        except Exception as e:
            logger.error(f"Error processing alert: {str(e)}")
    
    def _send_notifications(self, alert: Dict):
        """Send notifications for critical alerts"""
        try:
            if self.notification_channels['email'] and self.email_config['username']:
                self._send_email_alert(alert)
            
            if self.notification_channels['webhook']:
                self._send_webhook_alert(alert)
            
            if self.notification_channels['sms']:
                self._send_sms_alert(alert)
                
        except Exception as e:
            logger.error(f"Error sending notifications: {str(e)}")
    
    def _send_email_alert(self, alert: Dict):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = "admin@refinery-platform.com"  # Configure recipient
            msg['Subject'] = f"DAN_G Alert: {alert['title']}"
            
            body = f"""
            Alert Details:
            - Type: {alert['type']}
            - Severity: {alert['severity']}
            - Metric: {alert['metric_name']}
            - Value: {alert['value']}
            - Threshold: {alert['threshold']}
            - Time: {alert['timestamp']}
            - Message: {alert['message']}
            
            Please check the system immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert['id']}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
    
    def _send_webhook_alert(self, alert: Dict):
        """Send webhook alert (placeholder)"""
        # Implement webhook notification
        logger.info(f"Webhook alert sent for {alert['id']}")
    
    def _send_sms_alert(self, alert: Dict):
        """Send SMS alert (placeholder)"""
        # Implement SMS notification
        logger.info(f"SMS alert sent for {alert['id']}")
    
    def _trigger_automated_response(self, alert: Dict):
        """Trigger automated responses based on alert type and severity"""
        try:
            if alert['type'] == 'system' and alert['severity'] == 'critical':
                # Trigger system scaling or restart
                self._trigger_system_scaling()
            
            elif alert['type'] == 'data_quality' and alert['severity'] in ['high', 'critical']:
                # Trigger data quality check
                self._trigger_data_quality_check()
            
            elif alert['type'] == 'forecast' and alert['severity'] in ['high', 'critical']:
                # Trigger model retraining
                self._trigger_model_retraining()
            
        except Exception as e:
            logger.error(f"Error triggering automated response: {str(e)}")
    
    def _trigger_system_scaling(self):
        """Trigger system scaling (placeholder)"""
        logger.info("Triggering system scaling...")
        # Implement auto-scaling logic
    
    def _trigger_data_quality_check(self):
        """Trigger data quality check (placeholder)"""
        logger.info("Triggering data quality check...")
        # Implement data quality validation
    
    def _trigger_model_retraining(self):
        """Trigger model retraining (placeholder)"""
        logger.info("Triggering model retraining...")
        # Implement model retraining logic
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        return [alert for alert in self.alert_history if alert['is_active'] and not alert['is_resolved']]
    
    def get_alerts_by_severity(self, severity: str) -> List[Dict]:
        """Get alerts by severity level"""
        return [alert for alert in self.alert_history if alert['severity'] == severity]
    
    def get_alerts_by_type(self, alert_type: str) -> List[Dict]:
        """Get alerts by type"""
        return [alert for alert in self.alert_history if alert['type'] == alert_type]
    
    def acknowledge_alert(self, alert_id: str, user_id: str = None) -> bool:
        """Acknowledge an alert"""
        try:
            for alert in self.alert_history:
                if alert['id'] == alert_id:
                    alert['is_acknowledged'] = True
                    alert['acknowledged_by'] = user_id
                    alert['acknowledged_at'] = datetime.now().isoformat()
                    logger.info(f"Alert {alert_id} acknowledged by {user_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error acknowledging alert: {str(e)}")
            return False
    
    def resolve_alert(self, alert_id: str, user_id: str = None, resolution_notes: str = None) -> bool:
        """Resolve an alert"""
        try:
            for alert in self.alert_history:
                if alert['id'] == alert_id:
                    alert['is_resolved'] = True
                    alert['resolved_by'] = user_id
                    alert['resolved_at'] = datetime.now().isoformat()
                    alert['resolution_notes'] = resolution_notes
                    logger.info(f"Alert {alert_id} resolved by {user_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error resolving alert: {str(e)}")
            return False
    
    def get_alert_statistics(self, hours: int = 24) -> Dict:
        """Get alert statistics for the specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_alerts = [
                alert for alert in self.alert_history
                if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
            ]
            
            stats = {
                'total_alerts': len(recent_alerts),
                'active_alerts': len([a for a in recent_alerts if a['is_active'] and not a['is_resolved']]),
                'resolved_alerts': len([a for a in recent_alerts if a['is_resolved']]),
                'acknowledged_alerts': len([a for a in recent_alerts if a['is_acknowledged']]),
                'by_severity': {},
                'by_type': {},
                'average_resolution_time': self._calculate_average_resolution_time(recent_alerts)
            }
            
            # Count by severity
            for severity in ['low', 'medium', 'high', 'critical']:
                stats['by_severity'][severity] = len([a for a in recent_alerts if a['severity'] == severity])
            
            # Count by type
            for alert_type in ['system', 'forecast', 'data_quality', 'performance', 'security']:
                stats['by_type'][alert_type] = len([a for a in recent_alerts if a['type'] == alert_type])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating alert statistics: {str(e)}")
            return {}
    
    def _calculate_average_resolution_time(self, alerts: List[Dict]) -> float:
        """Calculate average resolution time in minutes"""
        try:
            resolved_alerts = [a for a in alerts if a['is_resolved'] and 'resolved_at' in a]
            
            if not resolved_alerts:
                return 0.0
            
            total_time = 0
            for alert in resolved_alerts:
                created_time = datetime.fromisoformat(alert['timestamp'])
                resolved_time = datetime.fromisoformat(alert['resolved_at'])
                resolution_time = (resolved_time - created_time).total_seconds() / 60  # minutes
                total_time += resolution_time
            
            return round(total_time / len(resolved_alerts), 2)
            
        except Exception as e:
            logger.error(f"Error calculating average resolution time: {str(e)}")
            return 0.0
    
    def configure_notifications(self, email_config: Dict = None, webhook_url: str = None, sms_config: Dict = None):
        """Configure notification channels"""
        if email_config:
            self.email_config.update(email_config)
            self.notification_channels['email'] = True
        
        if webhook_url:
            self.notification_channels['webhook'] = True
            self.webhook_url = webhook_url
        
        if sms_config:
            self.notification_channels['sms'] = True
            self.sms_config = sms_config
    
    def update_alert_rules(self, new_rules: Dict):
        """Update alert rules and thresholds"""
        self.alert_rules.update(new_rules)
        logger.info("Alert rules updated")
    
    def clear_old_alerts(self, days: int = 30):
        """Clear alerts older than specified days"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            self.alert_history = [
                alert for alert in self.alert_history
                if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
            ]
            logger.info(f"Cleared alerts older than {days} days")
        except Exception as e:
            logger.error(f"Error clearing old alerts: {str(e)}")
