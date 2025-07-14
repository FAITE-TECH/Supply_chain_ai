#!/usr/bin/env python3
"""
Model Deployment Service for Automated Procurement System
This service loads trained models and provides prediction capabilities for production use
"""

import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ProcurementModelService:
    """Production model service for procurement predictions"""
    
    def __init__(self, models_path: str = "../models"):
        self.models_path = Path(models_path)
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self.load_models()
        
    def load_models(self):
        """Load all trained models from disk"""
        try:
            logger.info("Loading trained models for production...")
            
            # Load training results
            with open(self.models_path / 'training_results.json', 'r') as f:
                training_results = json.load(f)
            
            for task_name, task_info in training_results['trained_models'].items():
                # Construct full path relative to models directory
                model_path = self.models_path / task_info['model_path'].replace('models\\', '').replace('models/', '')
                
                # Load model components
                self.models[task_name] = joblib.load(model_path / 'model.pkl')
                self.scalers[task_name] = joblib.load(model_path / 'scaler.pkl')
                
                with open(model_path / 'metadata.json', 'r') as f:
                    self.metadata[task_name] = json.load(f)
                
                logger.info(f"Loaded model for {task_name} - R²: {task_info['metrics']['r2']:.3f}")
            
            logger.info(f"Successfully loaded {len(self.models)} models for production")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict_demand_forecast(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict future demand for products
        
        Args:
            features: DataFrame with product features
            
        Returns:
            Dictionary with demand predictions and confidence
        """
        try:
            task_name = "demand_forecast"
            model = self.models[task_name]
            scaler = self.scalers[task_name]
            selected_features = self.metadata[task_name]['selected_features']
            
            # Prepare features
            X = features[selected_features].fillna(0)
            X_scaled = scaler.transform(X)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            # Calculate prediction intervals using model's estimators (for tree-based models)
            prediction_intervals = self._calculate_prediction_intervals(
                model, X_scaled, predictions, confidence=0.95
            )
            
            results = {
                'predictions': predictions.tolist(),
                'prediction_intervals': prediction_intervals,
                'confidence_score': self._calculate_confidence_score(model, X_scaled),
                'task': task_name,
                'timestamp': datetime.now().isoformat(),
                'model_performance': {
                    'r2_score': self.metadata[task_name]['metrics']['r2'],
                    'rmse': self.metadata[task_name]['metrics']['rmse']
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in demand forecasting: {e}")
            raise
    
    def predict_stockout_risk(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict stockout risk for products
        
        Args:
            features: DataFrame with product features
            
        Returns:
            Dictionary with stockout risk predictions
        """
        try:
            task_name = "stockout_risk"
            model = self.models[task_name]
            scaler = self.scalers[task_name]
            selected_features = self.metadata[task_name]['selected_features']
            
            # Prepare features
            X = features[selected_features].fillna(0)
            X_scaled = scaler.transform(X)
            
            # Make predictions
            risk_scores = model.predict(X_scaled)
            
            # Convert to risk categories
            risk_categories = self._categorize_stockout_risk(risk_scores)
            
            results = {
                'risk_scores': risk_scores.tolist(),
                'risk_categories': risk_categories,
                'high_risk_products': self._identify_high_risk_products(features, risk_scores),
                'recommended_actions': self._generate_stockout_recommendations(risk_scores),
                'task': task_name,
                'timestamp': datetime.now().isoformat(),
                'model_performance': {
                    'r2_score': self.metadata[task_name]['metrics']['r2'],
                    'rmse': self.metadata[task_name]['metrics']['rmse']
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in stockout risk prediction: {e}")
            raise
    
    def predict_delivery_performance(self, supplier_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict supplier delivery performance
        
        Args:
            supplier_features: DataFrame with supplier features
            
        Returns:
            Dictionary with delivery performance predictions
        """
        try:
            task_name = "delivery_performance"
            model = self.models[task_name]
            scaler = self.scalers[task_name]
            selected_features = self.metadata[task_name]['selected_features']
            
            # Prepare features
            X = supplier_features[selected_features].fillna(0)
            X_scaled = scaler.transform(X)
            
            # Make predictions
            performance_scores = model.predict(X_scaled)
            
            # Rank suppliers
            supplier_rankings = self._rank_suppliers(supplier_features, performance_scores)
            
            results = {
                'performance_scores': performance_scores.tolist(),
                'supplier_rankings': supplier_rankings,
                'recommended_suppliers': self._get_top_suppliers(supplier_rankings, top_n=5),
                'delivery_insights': self._generate_delivery_insights(performance_scores),
                'task': task_name,
                'timestamp': datetime.now().isoformat(),
                'model_performance': {
                    'r2_score': self.metadata[task_name]['metrics']['r2'],
                    'rmse': self.metadata[task_name]['metrics']['rmse']
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in delivery performance prediction: {e}")
            raise
    
    def predict_procurement_priority(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict procurement priority scores
        
        Args:
            features: DataFrame with product/procurement features
            
        Returns:
            Dictionary with priority predictions
        """
        try:
            task_name = "procurement_priority"
            model = self.models[task_name]
            scaler = self.scalers[task_name]
            selected_features = self.metadata[task_name]['selected_features']
            
            # Prepare features
            X = features[selected_features].fillna(0)
            X_scaled = scaler.transform(X)
            
            # Make predictions
            priority_scores = model.predict(X_scaled)
            
            # Create priority rankings
            priority_rankings = self._create_priority_rankings(features, priority_scores)
            
            results = {
                'priority_scores': priority_scores.tolist(),
                'priority_rankings': priority_rankings,
                'urgent_items': self._identify_urgent_procurement(priority_rankings),
                'procurement_schedule': self._generate_procurement_schedule(priority_rankings),
                'task': task_name,
                'timestamp': datetime.now().isoformat(),
                'model_performance': {
                    'r2_score': self.metadata[task_name]['metrics']['r2'],
                    'rmse': self.metadata[task_name]['metrics']['rmse']
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in procurement priority prediction: {e}")
            raise
    
    def comprehensive_procurement_analysis(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive analysis using all models
        
        Args:
            features: DataFrame with all required features
            
        Returns:
            Comprehensive procurement insights
        """
        try:
            logger.info("Running comprehensive procurement analysis...")
            
            # Run all predictions
            demand_results = self.predict_demand_forecast(features)
            stockout_results = self.predict_stockout_risk(features)
            priority_results = self.predict_procurement_priority(features)
            
            # Combine insights
            comprehensive_results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'demand_forecast': demand_results,
                'stockout_risk': stockout_results,
                'procurement_priority': priority_results,
                'integrated_insights': self._generate_integrated_insights(
                    demand_results, stockout_results, priority_results, features
                ),
                'action_plan': self._create_action_plan(
                    demand_results, stockout_results, priority_results
                ),
                'performance_summary': {
                    'models_used': len(self.models),
                    'prediction_confidence': self._calculate_overall_confidence(),
                    'analysis_scope': len(features)
                }
            }
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            raise
    
    def _calculate_prediction_intervals(self, model, X, predictions, confidence=0.95):
        """Calculate prediction intervals for uncertainty quantification"""
        try:
            if hasattr(model, 'estimators_'):
                # For ensemble models, use individual estimator predictions
                estimator_predictions = np.array([
                    estimator.predict(X) for estimator in model.estimators_
                ])
                
                alpha = 1 - confidence
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bound = np.percentile(estimator_predictions, lower_percentile, axis=0)
                upper_bound = np.percentile(estimator_predictions, upper_percentile, axis=0)
                
                return {
                    'lower_bound': lower_bound.tolist(),
                    'upper_bound': upper_bound.tolist(),
                    'confidence_level': confidence
                }
            else:
                # For non-ensemble models, use simple approximation
                prediction_std = np.std(predictions) if len(predictions) > 1 else 0.1
                margin = 1.96 * prediction_std  # 95% confidence interval
                
                return {
                    'lower_bound': (predictions - margin).tolist(),
                    'upper_bound': (predictions + margin).tolist(),
                    'confidence_level': confidence
                }
        except:
            return {'lower_bound': [], 'upper_bound': [], 'confidence_level': confidence}
    
    def _calculate_confidence_score(self, model, X):
        """Calculate overall confidence score for predictions"""
        try:
            if hasattr(model, 'estimators_'):
                # Use prediction variance for ensemble models
                estimator_predictions = np.array([
                    estimator.predict(X) for estimator in model.estimators_
                ])
                prediction_variance = np.mean(np.var(estimator_predictions, axis=0))
                confidence = max(0, min(1, 1 - prediction_variance))
                return float(confidence)
            else:
                return 0.85  # Default confidence for non-ensemble models
        except:
            return 0.80
    
    def _categorize_stockout_risk(self, risk_scores):
        """Categorize stockout risk scores into risk levels"""
        categories = []
        for score in risk_scores:
            if score > 0.7:
                categories.append("High Risk")
            elif score > 0.4:
                categories.append("Medium Risk")
            elif score > 0.2:
                categories.append("Low Risk")
            else:
                categories.append("Very Low Risk")
        return categories
    
    def _identify_high_risk_products(self, features, risk_scores):
        """Identify products with high stockout risk"""
        high_risk_threshold = 0.7
        high_risk_indices = np.where(risk_scores > high_risk_threshold)[0]
        
        if 'Product Name' in features.columns:
            high_risk_products = [
                {
                    'product': features.iloc[i]['Product Name'],
                    'risk_score': float(risk_scores[i]),
                    'index': int(i)
                }
                for i in high_risk_indices
            ]
        else:
            high_risk_products = [
                {
                    'product': f"Product_{i}",
                    'risk_score': float(risk_scores[i]),
                    'index': int(i)
                }
                for i in high_risk_indices
            ]
        
        return sorted(high_risk_products, key=lambda x: x['risk_score'], reverse=True)
    
    def _generate_stockout_recommendations(self, risk_scores):
        """Generate recommendations based on stockout risk"""
        high_risk_count = np.sum(risk_scores > 0.7)
        medium_risk_count = np.sum((risk_scores > 0.4) & (risk_scores <= 0.7))
        
        recommendations = []
        
        if high_risk_count > 0:
            recommendations.append({
                'priority': 'Critical',
                'action': f'Immediate reordering required for {high_risk_count} high-risk products',
                'timeline': 'Within 24 hours'
            })
        
        if medium_risk_count > 0:
            recommendations.append({
                'priority': 'Medium',
                'action': f'Schedule reordering for {medium_risk_count} medium-risk products',
                'timeline': 'Within 1 week'
            })
        
        recommendations.append({
            'priority': 'Low',
            'action': 'Monitor inventory levels and update forecasts',
            'timeline': 'Ongoing'
        })
        
        return recommendations
    
    def _rank_suppliers(self, supplier_features, performance_scores):
        """Rank suppliers based on performance scores"""
        rankings = []
        
        for i, score in enumerate(performance_scores):
            supplier_name = f"Supplier_{i}"
            if 'Supplier Name' in supplier_features.columns:
                supplier_name = supplier_features.iloc[i].get('Supplier Name', supplier_name)
            elif 'Product Name' in supplier_features.columns:
                supplier_name = supplier_features.iloc[i].get('Product Name', supplier_name)
            
            rankings.append({
                'supplier': supplier_name,
                'performance_score': float(score),
                'rank': 0,  # Will be set after sorting
                'index': i
            })
        
        # Sort by performance score (descending)
        rankings.sort(key=lambda x: x['performance_score'], reverse=True)
        
        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def _get_top_suppliers(self, supplier_rankings, top_n=5):
        """Get top N suppliers based on rankings"""
        return supplier_rankings[:top_n]
    
    def _generate_delivery_insights(self, performance_scores):
        """Generate insights about delivery performance"""
        avg_performance = np.mean(performance_scores)
        excellent_suppliers = np.sum(performance_scores > 0.9)
        poor_suppliers = np.sum(performance_scores < 0.5)
        
        insights = {
            'average_performance': float(avg_performance),
            'excellent_suppliers_count': int(excellent_suppliers),
            'poor_suppliers_count': int(poor_suppliers),
            'performance_distribution': {
                'excellent': int(excellent_suppliers),
                'good': int(np.sum((performance_scores >= 0.7) & (performance_scores <= 0.9))),
                'average': int(np.sum((performance_scores >= 0.5) & (performance_scores < 0.7))),
                'poor': int(poor_suppliers)
            },
            'recommendation': self._get_performance_recommendation(avg_performance)
        }
        
        return insights
    
    def _get_performance_recommendation(self, avg_performance):
        """Get recommendation based on average performance"""
        if avg_performance > 0.85:
            return "Supplier portfolio performing excellently. Continue current relationships."
        elif avg_performance > 0.70:
            return "Good supplier performance. Consider optimizing underperforming suppliers."
        elif avg_performance > 0.50:
            return "Average performance. Review supplier contracts and consider alternatives."
        else:
            return "Poor supplier performance. Urgent review and replacement needed."
    
    def _create_priority_rankings(self, features, priority_scores):
        """Create priority rankings for procurement items"""
        rankings = []
        
        for i, score in enumerate(priority_scores):
            product_name = f"Item_{i}"
            if 'Product Name' in features.columns:
                product_name = features.iloc[i].get('Product Name', product_name)
            
            rankings.append({
                'product': product_name,
                'priority_score': float(score),
                'priority_level': self._get_priority_level(score),
                'rank': 0,  # Will be set after sorting
                'index': i
            })
        
        # Sort by priority score (descending)
        rankings.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def _get_priority_level(self, score):
        """Convert priority score to priority level"""
        if score > 80:
            return "Critical"
        elif score > 60:
            return "High"
        elif score > 40:
            return "Medium"
        elif score > 20:
            return "Low"
        else:
            return "Minimal"
    
    def _identify_urgent_procurement(self, priority_rankings):
        """Identify urgent procurement items"""
        urgent_items = [
            item for item in priority_rankings 
            if item['priority_level'] in ['Critical', 'High']
        ]
        return urgent_items[:10]  # Top 10 urgent items
    
    def _generate_procurement_schedule(self, priority_rankings):
        """Generate procurement schedule based on priorities"""
        schedule = {
            'immediate': [],
            'this_week': [],
            'next_week': [],
            'this_month': []
        }
        
        for item in priority_rankings:
            if item['priority_level'] == 'Critical':
                schedule['immediate'].append(item)
            elif item['priority_level'] == 'High':
                schedule['this_week'].append(item)
            elif item['priority_level'] == 'Medium':
                schedule['next_week'].append(item)
            else:
                schedule['this_month'].append(item)
        
        return schedule
    
    def _generate_integrated_insights(self, demand_results, stockout_results, priority_results, features):
        """Generate integrated insights from all model predictions"""
        insights = {
            'critical_actions': [],
            'optimization_opportunities': [],
            'risk_assessment': {},
            'cost_savings_potential': {}
        }
        
        # Identify critical actions
        high_demand_items = [i for i, pred in enumerate(demand_results['predictions']) if pred > np.mean(demand_results['predictions']) * 1.5]
        high_risk_items = [item['index'] for item in stockout_results['high_risk_products']]
        urgent_items = [item['index'] for item in priority_results['urgent_items']]
        
        critical_items = set(high_demand_items) & set(high_risk_items) & set(urgent_items)
        
        if critical_items:
            insights['critical_actions'].append({
                'action': 'Immediate procurement required',
                'items_count': len(critical_items),
                'items': list(critical_items),
                'reason': 'High demand + High stockout risk + Urgent priority'
            })
        
        # Risk assessment
        insights['risk_assessment'] = {
            'overall_risk_level': self._calculate_overall_risk(stockout_results['risk_scores']),
            'high_risk_products_count': len(stockout_results['high_risk_products']),
            'average_stockout_risk': float(np.mean(stockout_results['risk_scores']))
        }
        
        return insights
    
    def _calculate_overall_risk(self, risk_scores):
        """Calculate overall risk level"""
        avg_risk = np.mean(risk_scores)
        if avg_risk > 0.7:
            return "High"
        elif avg_risk > 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _create_action_plan(self, demand_results, stockout_results, priority_results):
        """Create comprehensive action plan"""
        action_plan = {
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': []
        }
        
        # Immediate actions
        if stockout_results['high_risk_products']:
            action_plan['immediate_actions'].append({
                'action': 'Emergency procurement for high-risk products',
                'products': [p['product'] for p in stockout_results['high_risk_products'][:5]],
                'timeline': 'Within 24 hours'
            })
        
        # Short-term actions
        if priority_results['urgent_items']:
            action_plan['short_term_actions'].append({
                'action': 'Process urgent procurement requests',
                'count': len(priority_results['urgent_items']),
                'timeline': 'Within 1 week'
            })
        
        # Long-term actions
        action_plan['long_term_actions'].append({
            'action': 'Optimize supplier relationships and contracts',
            'timeline': 'Within 1 month'
        })
        
        return action_plan
    
    def _calculate_overall_confidence(self):
        """Calculate overall confidence across all models"""
        r2_scores = [self.metadata[task]['metrics']['r2'] for task in self.models.keys()]
        return float(np.mean(r2_scores))