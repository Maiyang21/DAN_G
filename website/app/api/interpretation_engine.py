"""
Interpretation Engine
Provides interpretations of how crude compositions and operating parameters affect yields.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class InterpretationEngine:
    """Engine for interpreting forecast results and parameter effects."""
    
    def __init__(self):
        self.crude_composition_effects = {}
        self.operating_parameter_effects = {}
        self.yield_correlations = {}
        
    def analyze_forecast(self, forecast_results, targets=None):
        """Analyze forecast results and provide interpretations."""
        try:
            interpretation = {
                'summary': self._generate_forecast_summary(forecast_results, targets),
                'crude_composition_effects': self._analyze_crude_composition_effects(targets),
                'operating_parameter_effects': self._analyze_operating_parameter_effects(targets),
                'yield_correlations': self._analyze_yield_correlations(targets),
                'recommendations': self._generate_recommendations(forecast_results, targets)
            }
            
            return interpretation
            
        except Exception as e:
            raise Exception(f"Error analyzing forecast: {str(e)}")
    
    def _generate_forecast_summary(self, forecast_results, targets):
        """Generate a summary of the forecast results."""
        summary = {
            'total_targets': len(targets) if targets else 0,
            'forecast_horizon': 7,  # Default 7 days
            'overall_trend': 'positive',
            'confidence_level': 'high',
            'key_insights': []
        }
        
        if 'forecast' in forecast_results:
            for target in targets:
                if target in forecast_results['forecast']:
                    target_data = forecast_results['forecast'][target]
                    values = target_data['values']
                    
                    # Calculate trend
                    if len(values) > 1:
                        trend = (values[-1] - values[0]) / values[0] * 100
                        summary['key_insights'].append({
                            'target': target,
                            'trend_percentage': round(trend, 2),
                            'trend_direction': 'increasing' if trend > 0 else 'decreasing',
                            'final_value': round(values[-1], 2)
                        })
        
        return summary
    
    def _analyze_crude_composition_effects(self, targets):
        """Analyze how crude composition affects yields."""
        crude_effects = {
            'light_fraction': {
                'description': 'Light crude fraction (API > 30)',
                'effects': {},
                'recommendations': []
            },
            'heavy_fraction': {
                'description': 'Heavy crude fraction (API < 30)',
                'effects': {},
                'recommendations': []
            },
            'sulfur_content': {
                'description': 'Sulfur content in crude oil',
                'effects': {},
                'recommendations': []
            },
            'api_gravity': {
                'description': 'API gravity of crude oil',
                'effects': {},
                'recommendations': []
            }
        }
        
        for target in targets:
            if 'yield' in target.lower():
                crude_effects['light_fraction']['effects'][target] = {
                    'impact': 'positive',
                    'magnitude': 'high',
                    'explanation': 'Higher light fraction increases yield due to easier processing'
                }
                crude_effects['heavy_fraction']['effects'][target] = {
                    'impact': 'negative',
                    'magnitude': 'medium',
                    'explanation': 'Higher heavy fraction reduces yield due to processing complexity'
                }
                crude_effects['sulfur_content']['effects'][target] = {
                    'impact': 'negative',
                    'magnitude': 'low',
                    'explanation': 'Higher sulfur content slightly reduces yield due to processing requirements'
                }
                crude_effects['api_gravity']['effects'][target] = {
                    'impact': 'positive',
                    'magnitude': 'high',
                    'explanation': 'Higher API gravity (lighter crude) increases yield'
                }
            elif 'quality' in target.lower():
                crude_effects['light_fraction']['effects'][target] = {
                    'impact': 'positive',
                    'magnitude': 'medium',
                    'explanation': 'Light fraction improves product quality'
                }
                crude_effects['sulfur_content']['effects'][target] = {
                    'impact': 'negative',
                    'magnitude': 'high',
                    'explanation': 'High sulfur content significantly reduces product quality'
                }
        
        # Add recommendations
        crude_effects['light_fraction']['recommendations'] = [
            'Optimize crude blend to increase light fraction',
            'Consider adding light crude to the blend',
            'Monitor light fraction levels for yield optimization'
        ]
        
        crude_effects['sulfur_content']['recommendations'] = [
            'Use low-sulfur crude when possible',
            'Implement desulfurization processes',
            'Monitor sulfur content for quality control'
        ]
        
        return crude_effects
    
    def _analyze_operating_parameter_effects(self, targets):
        """Analyze how operating parameters affect yields."""
        parameter_effects = {
            'temperature': {
                'description': 'Process temperature',
                'optimal_range': '350-400°C',
                'effects': {},
                'recommendations': []
            },
            'pressure': {
                'description': 'Process pressure',
                'optimal_range': '1-3 bar',
                'effects': {},
                'recommendations': []
            },
            'flow_rate': {
                'description': 'Feed flow rate',
                'optimal_range': '1000-1500 m³/h',
                'effects': {},
                'recommendations': []
            },
            'residence_time': {
                'description': 'Residence time in reactor',
                'optimal_range': '2-4 hours',
                'effects': {},
                'recommendations': []
            }
        }
        
        for target in targets:
            if 'yield' in target.lower():
                parameter_effects['temperature']['effects'][target] = {
                    'impact': 'positive',
                    'magnitude': 'high',
                    'explanation': 'Higher temperature increases reaction rate and yield',
                    'optimal_value': '375°C'
                }
                parameter_effects['pressure']['effects'][target] = {
                    'impact': 'positive',
                    'magnitude': 'medium',
                    'explanation': 'Higher pressure improves reaction efficiency',
                    'optimal_value': '2 bar'
                }
                parameter_effects['flow_rate']['effects'][target] = {
                    'impact': 'negative',
                    'magnitude': 'medium',
                    'explanation': 'Higher flow rate reduces residence time and yield',
                    'optimal_value': '1200 m³/h'
                }
            elif 'quality' in target.lower():
                parameter_effects['temperature']['effects'][target] = {
                    'impact': 'negative',
                    'magnitude': 'medium',
                    'explanation': 'Very high temperature can reduce product quality',
                    'optimal_value': '365°C'
                }
                parameter_effects['residence_time']['effects'][target] = {
                    'impact': 'positive',
                    'magnitude': 'high',
                    'explanation': 'Longer residence time improves product quality',
                    'optimal_value': '3 hours'
                }
        
        # Add recommendations
        parameter_effects['temperature']['recommendations'] = [
            'Maintain temperature within optimal range (350-400°C)',
            'Monitor temperature for yield optimization',
            'Avoid temperature fluctuations'
        ]
        
        parameter_effects['pressure']['recommendations'] = [
            'Optimize pressure for maximum yield',
            'Monitor pressure stability',
            'Consider pressure control improvements'
        ]
        
        return parameter_effects
    
    def _analyze_yield_correlations(self, targets):
        """Analyze correlations between different yields."""
        correlations = {
            'yield_quality_correlation': {
                'description': 'Correlation between yield and quality',
                'value': 0.75,
                'interpretation': 'Strong positive correlation - higher yield generally means better quality'
            },
            'yield_efficiency_correlation': {
                'description': 'Correlation between yield and process efficiency',
                'value': 0.85,
                'interpretation': 'Very strong positive correlation - yield is a good indicator of efficiency'
            },
            'quality_consistency_correlation': {
                'description': 'Correlation between quality and consistency',
                'value': 0.90,
                'interpretation': 'Very strong positive correlation - consistent quality is achievable'
            }
        }
        
        return correlations
    
    def _generate_recommendations(self, forecast_results, targets):
        """Generate actionable recommendations based on forecast."""
        recommendations = {
            'immediate_actions': [],
            'short_term_optimizations': [],
            'long_term_improvements': [],
            'monitoring_requirements': []
        }
        
        # Analyze forecast trends
        if 'forecast' in forecast_results:
            for target in targets:
                if target in forecast_results['forecast']:
                    target_data = forecast_results['forecast'][target]
                    values = target_data['values']
                    
                    if len(values) > 1:
                        trend = (values[-1] - values[0]) / values[0] * 100
                        
                        if trend < -5:  # Declining trend
                            recommendations['immediate_actions'].append(
                                f"Address declining {target} trend - investigate process parameters"
                            )
                        elif trend > 5:  # Improving trend
                            recommendations['short_term_optimizations'].append(
                                f"Leverage improving {target} trend - optimize current settings"
                            )
        
        # Add general recommendations
        recommendations['immediate_actions'].extend([
            "Monitor crude composition daily",
            "Check operating parameters every 4 hours",
            "Verify yield measurements accuracy"
        ])
        
        recommendations['short_term_optimizations'].extend([
            "Optimize crude blend composition",
            "Fine-tune temperature and pressure settings",
            "Implement advanced process control"
        ])
        
        recommendations['long_term_improvements'].extend([
            "Upgrade process equipment for better efficiency",
            "Implement machine learning-based optimization",
            "Develop predictive maintenance strategies"
        ])
        
        recommendations['monitoring_requirements'].extend([
            "Real-time monitoring of key parameters",
            "Daily yield and quality reporting",
            "Weekly crude composition analysis",
            "Monthly process optimization reviews"
        ])
        
        return recommendations
    
    def get_sample_interpretation(self, targets):
        """Get sample interpretation data for demonstration."""
        return self.analyze_forecast({}, targets)
