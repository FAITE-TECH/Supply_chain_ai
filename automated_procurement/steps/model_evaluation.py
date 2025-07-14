#!/usr/bin/env python3
"""
Model Evaluation Step for Automated Procurement Model
This script implements comprehensive model evaluation, validation, and performance assessment
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, learning_curve
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class ProcurementModelEvaluation:
    """Class for evaluating trained procurement ML models"""
    
    def __init__(self, models_path: str = "models", processed_data_path: str = "data/processed", 
                 artifacts_path: str = "artifacts"):
        self.models_path = Path(models_path)
        self.processed_data_path = Path(processed_data_path)
        self.artifacts_path = Path(artifacts_path)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized model evaluation with models path: {self.models_path}")
    
    def load_trained_models(self) -> Dict[str, Any]:
        """
        Load all trained models and their metadata
        
        Returns:
            Dictionary with loaded models and metadata
        """
        try:
            logger.info("Loading trained models...")
            
            # Load training results
            results_file = self.models_path / 'training_results.json'
            if not results_file.exists():
                raise FileNotFoundError("Training results file not found. Run model training first.")
            
            with open(results_file, 'r') as f:
                training_results = json.load(f)
            
            loaded_models = {}
            
            for task_name, task_info in training_results['trained_models'].items():
                task_path = Path(task_info['model_path'])
                
                # Load model
                model_file = task_path / 'model.pkl'
                scaler_file = task_path / 'scaler.pkl'
                metadata_file = task_path / 'metadata.json'
                
                if all(file.exists() for file in [model_file, scaler_file, metadata_file]):
                    model = joblib.load(model_file)
                    scaler = joblib.load(scaler_file)
                    
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    loaded_models[task_name] = {
                        'model': model,
                        'scaler': scaler,
                        'metadata': metadata,
                        'task_info': task_info
                    }
                    
                    logger.info(f"Loaded model for {task_name}")
                else:
                    logger.warning(f"Missing files for task {task_name}")
            
            logger.info(f"Successfully loaded {len(loaded_models)} models")
            return loaded_models
            
        except Exception as e:
            logger.error(f"Error loading trained models: {e}")
            raise
    
    def load_evaluation_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data for model evaluation
        
        Returns:
            Tuple of (procurement_df, features_df)
        """
        try:
            procurement_df = pd.read_csv(self.processed_data_path / "procurement_features.csv")
            
            # Create features dataframe if not exists
            ml_features_path = self.processed_data_path / "ml_features.csv"
            if ml_features_path.exists():
                features_df = pd.read_csv(ml_features_path)
            else:
                # Create from procurement features
                feature_columns = [col for col in procurement_df.columns 
                                 if col not in ['Product Name'] and 
                                 procurement_df[col].dtype in ['float64', 'int64']]
                features_df = procurement_df[['Product Name'] + feature_columns].copy()
                features_df = features_df.fillna(0)
            
            logger.info(f"Loaded evaluation data: {procurement_df.shape}, {features_df.shape}")
            return procurement_df, features_df
            
        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}")
            raise
    
    def evaluate_model_performance(self, model: Any, scaler: Any, selected_features: List[str],
                                 X: pd.DataFrame, y: np.ndarray, task_name: str) -> Dict[str, float]:
        """
        Evaluate model performance on the full dataset
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            selected_features: List of selected feature names
            X: Feature matrix
            y: Target variable
            task_name: Name of the task
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Select and scale features
            X_selected = X[selected_features].values
            X_scaled = scaler.transform(X_selected)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            
            # Calculate metrics
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
            
            # Calculate additional metrics
            metrics['mean_target'] = np.mean(y)
            metrics['std_target'] = np.std(y)
            metrics['mean_prediction'] = np.mean(y_pred)
            metrics['std_prediction'] = np.std(y_pred)
            metrics['prediction_range'] = (np.min(y_pred), np.max(y_pred))
            metrics['target_range'] = (np.min(y), np.max(y))
            
            # Cross-validation score
            try:
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                metrics['cv_r2_mean'] = np.mean(cv_scores)
                metrics['cv_r2_std'] = np.std(cv_scores)
            except:
                metrics['cv_r2_mean'] = np.nan
                metrics['cv_r2_std'] = np.nan
            
            logger.info(f"Model evaluation completed for {task_name}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model for {task_name}: {e}")
            return {}
    
    def generate_feature_importance_analysis(self, loaded_models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feature importance analysis for all models
        
        Args:
            loaded_models: Dictionary of loaded models
            
        Returns:
            Dictionary with feature importance results
        """
        feature_importance_results = {}
        
        for task_name, model_info in loaded_models.items():
            try:
                model = model_info['model']
                selected_features = model_info['metadata']['selected_features']
                
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance = list(zip(selected_features, importances))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    
                    feature_importance_results[task_name] = {
                        'type': 'tree_based',
                        'importances': feature_importance,
                        'top_10': feature_importance[:10]
                    }
                    
                elif hasattr(model, 'coef_'):
                    coefficients = np.abs(model.coef_)
                    feature_importance = list(zip(selected_features, coefficients))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    
                    feature_importance_results[task_name] = {
                        'type': 'linear',
                        'importances': feature_importance,
                        'top_10': feature_importance[:10]
                    }
                else:
                    feature_importance_results[task_name] = {
                        'type': 'not_available',
                        'importances': [],
                        'top_10': []
                    }
                
                logger.info(f"Feature importance analyzed for {task_name}")
                
            except Exception as e:
                logger.error(f"Error analyzing feature importance for {task_name}: {e}")
                feature_importance_results[task_name] = {'type': 'error', 'importances': [], 'top_10': []}
        
        return feature_importance_results
    
    def assess_business_impact(self, evaluation_results: Dict[str, Any], 
                             procurement_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess potential business impact of the models
        
        Args:
            evaluation_results: Model evaluation results
            procurement_df: Procurement dataframe
            
        Returns:
            Dictionary with business impact assessment
        """
        logger.info("Assessing business impact...")
        
        # Calculate baseline business metrics
        total_revenue = procurement_df['Gross Sales_sum'].sum()
        total_products = len(procurement_df)
        stockout_products = len(procurement_df[procurement_df['Stock_Status'] == 'Stockout'])
        
        business_impact = {
            'baseline_metrics': {
                'total_revenue': total_revenue,
                'total_products': total_products,
                'stockout_products': stockout_products,
                'stockout_rate': stockout_products / total_products
            },
            'model_impact': {}
        }
        
        for task_name, metrics in evaluation_results.items():
            r2_score = metrics.get('r2', 0)
            
            impact_assessment = {
                'model_quality': self._assess_model_quality(r2_score),
                'expected_improvements': self._calculate_expected_improvements(task_name, r2_score, procurement_df),
                'implementation_readiness': self._assess_implementation_readiness(metrics)
            }
            
            business_impact['model_impact'][task_name] = impact_assessment
        
        return business_impact
    
    def _assess_model_quality(self, r2_score: float) -> Dict[str, Any]:
        """Assess model quality based on R² score"""
        if r2_score >= 0.8:
            return {'level': 'Excellent', 'description': 'Ready for production deployment', 'confidence': 'High'}
        elif r2_score >= 0.6:
            return {'level': 'Good', 'description': 'Suitable for pilot implementation', 'confidence': 'Medium-High'}
        elif r2_score >= 0.4:
            return {'level': 'Fair', 'description': 'Requires monitoring and improvement', 'confidence': 'Medium'}
        else:
            return {'level': 'Poor', 'description': 'Needs significant improvement', 'confidence': 'Low'}
    
    def _calculate_expected_improvements(self, task_name: str, r2_score: float, 
                                       procurement_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate expected business improvements"""
        total_revenue = procurement_df['Gross Sales_sum'].sum()
        
        if task_name == 'demand_forecast':
            if r2_score > 0.7:
                inventory_cost_reduction = 0.20  # 20% reduction
                demand_accuracy_improvement = 0.85  # 85% accuracy
            elif r2_score > 0.5:
                inventory_cost_reduction = 0.10  # 10% reduction
                demand_accuracy_improvement = 0.70  # 70% accuracy
            else:
                inventory_cost_reduction = 0.05  # 5% reduction
                demand_accuracy_improvement = 0.55  # 55% accuracy
            
            return {
                'inventory_cost_reduction_percent': inventory_cost_reduction * 100,
                'demand_accuracy_improvement_percent': demand_accuracy_improvement * 100,
                'estimated_annual_savings': total_revenue * 0.02 * inventory_cost_reduction
            }
        
        elif task_name == 'stockout_risk':
            if r2_score > 0.6:
                stockout_prevention_rate = 0.80  # Prevent 80% of stockouts
            elif r2_score > 0.4:
                stockout_prevention_rate = 0.60  # Prevent 60% of stockouts
            else:
                stockout_prevention_rate = 0.40  # Prevent 40% of stockouts
            
            return {
                'stockout_prevention_rate_percent': stockout_prevention_rate * 100,
                'revenue_protection': 'High' if stockout_prevention_rate > 0.7 else 'Medium'
            }
        
        elif task_name == 'delivery_performance':
            if r2_score > 0.5:
                supplier_optimization = 'Significant'
                lead_time_improvement = 0.15  # 15% improvement
            else:
                supplier_optimization = 'Moderate'
                lead_time_improvement = 0.08  # 8% improvement
            
            return {
                'supplier_optimization_level': supplier_optimization,
                'lead_time_improvement_percent': lead_time_improvement * 100
            }
        
        elif task_name == 'procurement_priority':
            if r2_score > 0.6:
                efficiency_gain = 0.25  # 25% efficiency improvement
            else:
                efficiency_gain = 0.15  # 15% efficiency improvement
            
            return {
                'process_efficiency_gain_percent': efficiency_gain * 100,
                'automation_potential': 'High' if efficiency_gain > 0.2 else 'Medium'
            }
        
        return {'general_improvement': 'Model-specific improvements not defined'}
    
    def _assess_implementation_readiness(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess readiness for implementation"""
        r2 = metrics.get('r2', 0)
        rmse = metrics.get('rmse', float('inf'))
        
        if r2 > 0.7 and rmse < 1.0:
            return {
                'readiness_level': 'Ready',
                'deployment_recommendation': 'Proceed with production deployment',
                'monitoring_requirements': 'Standard monitoring',
                'risk_level': 'Low'
            }
        elif r2 > 0.5:
            return {
                'readiness_level': 'Pilot Ready',
                'deployment_recommendation': 'Start with pilot implementation',
                'monitoring_requirements': 'Enhanced monitoring and validation',
                'risk_level': 'Medium'
            }
        else:
            return {
                'readiness_level': 'Not Ready',
                'deployment_recommendation': 'Improve model before deployment',
                'monitoring_requirements': 'Extensive monitoring if deployed',
                'risk_level': 'High'
            }
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any],
                                 feature_importance: Dict[str, Any],
                                 business_impact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            evaluation_results: Model performance metrics
            feature_importance: Feature importance analysis
            business_impact: Business impact assessment
            
        Returns:
            Comprehensive evaluation report
        """
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'executive_summary': self._create_executive_summary(evaluation_results, business_impact),
            'model_performance': evaluation_results,
            'feature_importance': feature_importance,
            'business_impact': business_impact,
            'recommendations': self._generate_recommendations(evaluation_results, business_impact),
            'next_steps': self._define_next_steps(evaluation_results)
        }
        
        return report
    
    def _create_executive_summary(self, evaluation_results: Dict[str, Any], 
                                business_impact: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of evaluation results"""
        
        # Calculate average model performance
        r2_scores = [metrics.get('r2', 0) for metrics in evaluation_results.values()]
        avg_r2 = np.mean(r2_scores) if r2_scores else 0
        
        # Count models by readiness level
        ready_models = sum(1 for task, impact in business_impact['model_impact'].items() 
                          if impact['implementation_readiness']['readiness_level'] == 'Ready')
        
        total_models = len(evaluation_results)
        
        return {
            'overall_model_quality': self._assess_model_quality(avg_r2)['level'],
            'average_r2_score': avg_r2,
            'production_ready_models': ready_models,
            'total_models_evaluated': total_models,
            'deployment_readiness_percentage': (ready_models / total_models * 100) if total_models > 0 else 0,
            'key_insights': [
                f"Average model performance (R²): {avg_r2:.3f}",
                f"{ready_models}/{total_models} models ready for production",
                "Procurement automation feasibility: " + ("High" if avg_r2 > 0.6 else "Medium" if avg_r2 > 0.4 else "Low")
            ]
        }
    
    def _generate_recommendations(self, evaluation_results: Dict[str, Any], 
                                business_impact: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        for task_name, metrics in evaluation_results.items():
            r2_score = metrics.get('r2', 0)
            readiness = business_impact['model_impact'][task_name]['implementation_readiness']
            
            if readiness['readiness_level'] == 'Ready':
                recommendations.append({
                    'priority': 'High',
                    'task': task_name,
                    'action': 'Deploy to production',
                    'timeline': 'Immediate (1-2 weeks)',
                    'impact': 'High business value expected'
                })
            elif readiness['readiness_level'] == 'Pilot Ready':
                recommendations.append({
                    'priority': 'Medium',
                    'task': task_name,
                    'action': 'Start pilot implementation',
                    'timeline': 'Short-term (1-2 months)',
                    'impact': 'Moderate business value with monitoring'
                })
            else:
                recommendations.append({
                    'priority': 'Low',
                    'task': task_name,
                    'action': 'Improve model performance',
                    'timeline': 'Long-term (3-6 months)',
                    'impact': 'Requires additional development'
                })
        
        return recommendations
    
    def _define_next_steps(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Define next steps based on evaluation results"""
        avg_r2 = np.mean([metrics.get('r2', 0) for metrics in evaluation_results.values()])
        
        next_steps = [
            "Review evaluation results with stakeholders",
            "Prioritize models for deployment based on business impact"
        ]
        
        if avg_r2 > 0.6:
            next_steps.extend([
                "Proceed with production deployment planning",
                "Set up model monitoring and alerting",
                "Develop user training materials"
            ])
        else:
            next_steps.extend([
                "Investigate model performance improvements",
                "Consider additional feature engineering",
                "Evaluate alternative algorithms"
            ])
        
        next_steps.extend([
            "Establish model retraining schedule",
            "Plan integration with existing systems",
            "Define success metrics and KPIs"
        ])
        
        return next_steps
    
    def save_evaluation_results(self, evaluation_report: Dict[str, Any]) -> str:
        """
        Save evaluation results to files
        
        Args:
            evaluation_report: Comprehensive evaluation report
            
        Returns:
            Path to saved evaluation report
        """
        try:
            # Save comprehensive report
            report_file = self.artifacts_path / f"model_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert numpy types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_types(item) for item in obj)
                return obj
            
            json_report = convert_numpy_types(evaluation_report)
            
            with open(report_file, 'w') as f:
                json.dump(json_report, f, indent=2)
            
            # Save summary CSV
            summary_data = []
            for task_name, metrics in evaluation_report['model_performance'].items():
                summary_data.append({
                    'Task': task_name,
                    'R²': metrics.get('r2', 0),
                    'RMSE': metrics.get('rmse', 0),
                    'MAE': metrics.get('mae', 0),
                    'Readiness': evaluation_report['business_impact']['model_impact'][task_name]['implementation_readiness']['readiness_level']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = self.artifacts_path / f"model_performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            summary_df.to_csv(summary_file, index=False)
            
            logger.info(f"Evaluation results saved to {report_file}")
            logger.info(f"Performance summary saved to {summary_file}")
            
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            raise
    
    def run_model_evaluation_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete model evaluation pipeline
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting model evaluation pipeline...")
        
        try:
            # Load trained models
            loaded_models = self.load_trained_models()
            
            if not loaded_models:
                raise ValueError("No trained models found. Run model training first.")
            
            # Load evaluation data
            procurement_df, features_df = self.load_evaluation_data()
            
            # Evaluate each model
            evaluation_results = {}
            
            for task_name, model_info in loaded_models.items():
                logger.info(f"Evaluating model for {task_name}")
                
                # Get target variable
                target_col = model_info['metadata']['config']['target_col']
                
                if target_col in procurement_df.columns:
                    y = procurement_df[target_col].values
                    valid_mask = ~np.isnan(y)
                    
                    # Prepare features
                    feature_columns = [col for col in features_df.columns if col != 'Product Name']
                    X = features_df[feature_columns].fillna(0)[valid_mask]
                    y_clean = y[valid_mask]
                    
                    # Evaluate model
                    metrics = self.evaluate_model_performance(
                        model_info['model'],
                        model_info['scaler'],
                        model_info['metadata']['selected_features'],
                        X,
                        y_clean,
                        task_name
                    )
                    
                    evaluation_results[task_name] = metrics
                
                else:
                    logger.warning(f"Target column {target_col} not found for {task_name}")
            
            # Generate feature importance analysis
            feature_importance = self.generate_feature_importance_analysis(loaded_models)
            
            # Assess business impact
            business_impact = self.assess_business_impact(evaluation_results, procurement_df)
            
            # Generate comprehensive report
            evaluation_report = self.generate_evaluation_report(
                evaluation_results, feature_importance, business_impact
            )
            
            # Save results
            report_path = self.save_evaluation_results(evaluation_report)
            
            logger.info("Model evaluation pipeline completed successfully")
            
            return {
                'status': 'success',
                'evaluated_models': len(evaluation_results),
                'report_path': report_path,
                'evaluation_report': evaluation_report
            }
            
        except Exception as e:
            logger.error(f"Model evaluation pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }


def main():
    """Main function to run the model evaluation pipeline"""
    try:
        # Initialize model evaluation
        evaluator = ProcurementModelEvaluation()
        
        # Run evaluation pipeline
        results = evaluator.run_model_evaluation_pipeline()
        
        if results['status'] == 'success':
            print(f" Model evaluation completed successfully!")
            print(f" Evaluated models: {results['evaluated_models']}")
            print(f" Report saved to: {results['report_path']}")
            
            # Print executive summary
            exec_summary = results['evaluation_report']['executive_summary']
            print(f"\n EXECUTIVE SUMMARY:")
            print(f"   Overall Quality: {exec_summary['overall_model_quality']}")
            print(f"   Average R² Score: {exec_summary['average_r2_score']:.3f}")
            print(f"   Production Ready: {exec_summary['production_ready_models']}/{exec_summary['total_models_evaluated']} models")
            print(f"   Deployment Readiness: {exec_summary['deployment_readiness_percentage']:.1f}%")
        else:
            print(f" Model evaluation failed: {results['error']}")
            
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f" Fatal error: {e}")


if __name__ == "__main__":
    main()