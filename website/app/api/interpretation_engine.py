"""
Interpretation Engine for DAN_G Platform
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

class InterpretationEngine:
    def __init__(self):
        pass
    
    def analyze_forecast(self, data: pd.DataFrame, forecast_results: pd.DataFrame) -> Dict:
        """Analyze forecast results and provide insights"""
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'data_summary': self._analyze_data_summary(data),
                'forecast_analysis': self._analyze_forecast_results(forecast_results),
                'feature_importance': self._calculate_feature_importance(data),
                'recommendations': self._generate_recommendations(data, forecast_results)
            }
            return analysis
        except Exception as e:
            logger.error(f"Error in forecast analysis: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_data_summary(self, data: pd.DataFrame) -> Dict:
        """Analyze input data summary"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        return {
            'total_records': len(data),
            'total_features': len(data.columns),
            'numeric_features': len(numeric_cols),
            'missing_values': data.isnull().sum().sum(),
            'data_quality_score': self._calculate_data_quality_score(data)
        }
    
    def _analyze_forecast_results(self, forecast_results: pd.DataFrame) -> Dict:
        """Analyze forecast results"""
        if forecast_results.empty:
            return {'error': 'No forecast results to analyze'}
        
        return {
            'forecast_period': len(forecast_results),
            'average_forecast_value': forecast_results['forecast_value'].mean(),
            'forecast_trend': self._calculate_forecast_trend(forecast_results)
        }
    
    def _calculate_feature_importance(self, data: pd.DataFrame) -> Dict:
        """Calculate feature importance"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return {'error': 'Insufficient features'}
            
            X = data[numeric_cols].fillna(0)
            y = X.iloc[:, -1]
            X = X.iloc[:, :-1]
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            importance_scores = rf.feature_importances_
            feature_names = X.columns.tolist()
            
            return {
                'method': 'Random Forest',
                'features': dict(zip(feature_names, importance_scores.tolist())),
                'top_features': self._get_top_features(feature_names, importance_scores, 5)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_recommendations(self, data: pd.DataFrame, forecast_results: pd.DataFrame) -> List[Dict]:
        """Generate recommendations"""
        recommendations = []
        
        data_quality_score = self._calculate_data_quality_score(data)
        if data_quality_score < 0.8:
            recommendations.append({
                'category': 'Data Quality',
                'priority': 'High',
                'recommendation': 'Improve data quality by addressing missing values',
                'expected_impact': 'Improved forecast accuracy by 10-15%'
            })
        
        recommendations.append({
            'category': 'Operations',
            'priority': 'Medium',
            'recommendation': 'Monitor key parameters for optimal yields',
            'expected_impact': 'Maintain consistent production levels'
        })
        
        return recommendations
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score"""
        try:
            completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
            consistency = 1 - (data.duplicated().sum() / len(data))
            return round((completeness * 0.7 + consistency * 0.3), 3)
        except:
            return 0.0
    
    def _calculate_forecast_trend(self, forecast_results: pd.DataFrame) -> str:
        """Calculate forecast trend"""
        try:
            if len(forecast_results) < 2:
                return 'insufficient_data'
            
            first_half = forecast_results['forecast_value'].iloc[:len(forecast_results)//2].mean()
            second_half = forecast_results['forecast_value'].iloc[len(forecast_results)//2:].mean()
            
            if second_half > first_half * 1.05:
                return 'increasing'
            elif second_half < first_half * 0.95:
                return 'decreasing'
            else:
                return 'stable'
        except:
            return 'unknown'
    
    def _get_top_features(self, feature_names: List[str], importance_scores: np.ndarray, top_n: int = 5) -> List[Dict]:
        """Get top features"""
        try:
            sorted_indices = np.argsort(importance_scores)[::-1]
            top_features = []
            
            for i in range(min(top_n, len(feature_names))):
                idx = sorted_indices[i]
                top_features.append({
                    'feature': feature_names[idx],
                    'importance': float(importance_scores[idx]),
                    'rank': i + 1
                })
            
            return top_features
        except:
            return []