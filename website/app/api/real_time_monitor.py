"""
Real-time Monitoring System for DAN_G Platform
Monitors system performance, data quality, and forecasting accuracy
"""

import psutil
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from threading import Thread, Lock
import json

logger = logging.getLogger(__name__)

class RealTimeMonitor:
    """Real-time monitoring system for platform health and performance"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_history = []
        self.alert_thresholds = {
            'cpu_usage': 80,
            'memory_usage': 85,
            'disk_usage': 90,
            'response_time': 5000,  # 5 seconds
            'error_rate': 0.05,  # 5%
            'data_quality': 0.7
        }
        self.lock = Lock()
        self.active_forecasts = 0
        self.active_users = 0
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Store metrics with thread safety
                with self.lock:
                    self.metrics_history.append(metrics)
                    # Keep only last 1000 records
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Sleep for monitoring interval
                time.sleep(30)  # 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_metrics(self) -> Dict:
        """Collect comprehensive system metrics"""
        try:
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            process_cpu = process.cpu_percent()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Create metrics dictionary
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available': memory.available / 1024 / 1024 / 1024,  # GB
                'disk_usage': disk.percent,
                'disk_free': disk.free / 1024 / 1024 / 1024,  # GB
                'process_memory': process_memory,
                'process_cpu': process_cpu,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'active_forecasts': self.active_forecasts,
                'active_users': self.active_users,
                'uptime': time.time() - psutil.boot_time()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _check_alerts(self, metrics: Dict):
        """Check metrics against alert thresholds"""
        try:
            alerts = []
            
            # CPU usage alert
            if metrics.get('cpu_usage', 0) > self.alert_thresholds['cpu_usage']:
                alerts.append({
                    'type': 'cpu_usage',
                    'severity': 'high',
                    'message': f"High CPU usage: {metrics['cpu_usage']:.1f}%",
                    'value': metrics['cpu_usage'],
                    'threshold': self.alert_thresholds['cpu_usage']
                })
            
            # Memory usage alert
            if metrics.get('memory_usage', 0) > self.alert_thresholds['memory_usage']:
                alerts.append({
                    'type': 'memory_usage',
                    'severity': 'high',
                    'message': f"High memory usage: {metrics['memory_usage']:.1f}%",
                    'value': metrics['memory_usage'],
                    'threshold': self.alert_thresholds['memory_usage']
                })
            
            # Disk usage alert
            if metrics.get('disk_usage', 0) > self.alert_thresholds['disk_usage']:
                alerts.append({
                    'type': 'disk_usage',
                    'severity': 'critical',
                    'message': f"High disk usage: {metrics['disk_usage']:.1f}%",
                    'value': metrics['disk_usage'],
                    'threshold': self.alert_thresholds['disk_usage']
                })
            
            # Process alerts if any
            for alert in alerts:
                self._process_alert(alert)
                
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
    
    def _process_alert(self, alert: Dict):
        """Process and log alert"""
        logger.warning(f"ALERT: {alert['message']}")
        # In a real implementation, this would send notifications, emails, etc.
    
    def get_current_metrics(self) -> Dict:
        """Get current system metrics"""
        with self.lock:
            if self.metrics_history:
                return self.metrics_history[-1]
            return {}
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict]:
        """Get metrics history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            filtered_metrics = []
            for metric in self.metrics_history:
                try:
                    metric_time = datetime.fromisoformat(metric['timestamp'])
                    if metric_time >= cutoff_time:
                        filtered_metrics.append(metric)
                except:
                    continue
            
            return filtered_metrics
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for dashboard"""
        try:
            recent_metrics = self.get_metrics_history(hours=1)
            
            if not recent_metrics:
                return {'status': 'no_data'}
            
            # Calculate averages
            cpu_avg = np.mean([m.get('cpu_usage', 0) for m in recent_metrics])
            memory_avg = np.mean([m.get('memory_usage', 0) for m in recent_metrics])
            disk_avg = np.mean([m.get('disk_usage', 0) for m in recent_metrics])
            
            # Determine overall status
            status = 'healthy'
            if cpu_avg > 70 or memory_avg > 80 or disk_avg > 85:
                status = 'warning'
            if cpu_avg > 90 or memory_avg > 95 or disk_avg > 95:
                status = 'critical'
            
            return {
                'status': status,
                'cpu_usage': round(cpu_avg, 1),
                'memory_usage': round(memory_avg, 1),
                'disk_usage': round(disk_avg, 1),
                'active_forecasts': self.active_forecasts,
                'active_users': self.active_users,
                'uptime_hours': round((time.time() - psutil.boot_time()) / 3600, 1),
                'last_update': recent_metrics[-1]['timestamp'] if recent_metrics else None
            }
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def update_active_forecasts(self, count: int):
        """Update active forecasts count"""
        self.active_forecasts = max(0, count)
    
    def update_active_users(self, count: int):
        """Update active users count"""
        self.active_users = max(0, count)
    
    def get_system_health(self) -> Dict:
        """Get comprehensive system health status"""
        try:
            current_metrics = self.get_current_metrics()
            performance_summary = self.get_performance_summary()
            
            # Calculate health score (0-100)
            health_score = 100
            
            if current_metrics:
                # Deduct points for high resource usage
                cpu_usage = current_metrics.get('cpu_usage', 0)
                memory_usage = current_metrics.get('memory_usage', 0)
                disk_usage = current_metrics.get('disk_usage', 0)
                
                if cpu_usage > 80:
                    health_score -= (cpu_usage - 80) * 2
                if memory_usage > 80:
                    health_score -= (memory_usage - 80) * 2
                if disk_usage > 85:
                    health_score -= (disk_usage - 85) * 3
                
                health_score = max(0, min(100, health_score))
            
            return {
                'health_score': round(health_score, 1),
                'status': performance_summary.get('status', 'unknown'),
                'current_metrics': current_metrics,
                'performance_summary': performance_summary,
                'recommendations': self._get_health_recommendations(health_score, current_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {str(e)}")
            return {
                'health_score': 0,
                'status': 'error',
                'message': str(e)
            }
    
    def _get_health_recommendations(self, health_score: float, metrics: Dict) -> List[str]:
        """Get health recommendations based on current metrics"""
        recommendations = []
        
        if health_score < 50:
            recommendations.append("System health is critical. Consider scaling resources.")
        
        if metrics.get('cpu_usage', 0) > 80:
            recommendations.append("High CPU usage detected. Consider optimizing processes or scaling up.")
        
        if metrics.get('memory_usage', 0) > 80:
            recommendations.append("High memory usage detected. Consider memory optimization or scaling up.")
        
        if metrics.get('disk_usage', 0) > 85:
            recommendations.append("High disk usage detected. Consider cleaning up old data or scaling storage.")
        
        if self.active_forecasts > 100:
            recommendations.append("High number of active forecasts. Consider implementing queuing or load balancing.")
        
        if not recommendations:
            recommendations.append("System is running optimally.")
        
        return recommendations
    
    def collect_metrics(self) -> Dict:
        """Collect and return current metrics (for background tasks)"""
        return self._collect_system_metrics()


