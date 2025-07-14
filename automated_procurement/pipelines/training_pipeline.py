#!/usr/bin/env python3
"""
Machine Learning Pipeline for Automated Procurement Model
This script orchestrates the complete ML pipeline including training, evaluation, and model selection
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import pipeline steps
from steps.model_training import ProcurementModelTraining
from steps.model_evaluation import ProcurementModelEvaluation

# Configure logging
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / 'ml_pipeline.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcurementMLPipeline:
    """Main orchestrator for the procurement ML pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ML pipeline
        
        Args:
            config: Configuration dictionary for the pipeline
        """
        self.config = config or self._load_default_config()
        self.pipeline_id = f"ml_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {}
        
        # Ensure required directories exist
        self._setup_directories()
        
        logger.info(f"Initialized ML pipeline: {self.pipeline_id}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default ML pipeline configuration"""
        return {
            'processed_data_path': 'data/processed',
            'models_path': 'models',
            'artifacts_path': 'artifacts',
            'enable_hyperparameter_optimization': True,
            'enable_cross_validation': True,
            'save_intermediate_results': True,
            'ml_settings': {
                'test_size': 0.2,
                'random_state': 42,
                'n_features_select': 15,
                'cv_folds': 5,
                'optimization_models': ['random_forest', 'xgboost', 'gradient_boosting']
            },
            'evaluation_settings': {
                'generate_feature_importance': True,
                'assess_business_impact': True,
                'create_visualizations': True
            }
        }
    
    def _setup_directories(self):
        """Setup required directories for the pipeline"""
        directories = [
            'models',
            'artifacts',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _log_pipeline_step(self, step_name: str, status: str, details: Dict = None):
        """Log pipeline step execution"""
        log_entry = {
            'pipeline_id': self.pipeline_id,
            'step': step_name,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.results[step_name] = log_entry
        
        if status == 'success':
            logger.info(f"SUCCESS: {step_name} completed successfully")
        elif status == 'failed':
            logger.error(f"FAILED: {step_name} failed")
        elif status == 'started':
            logger.info(f"STARTED: {step_name} started")
    
    def run_model_training(self) -> bool:
        """
        Run model training step
        
        Returns:
            bool: True if successful, False otherwise
        """
        step_name = "model_training"
        self._log_pipeline_step(step_name, "started")
        
        try:
            trainer = ProcurementModelTraining(
                processed_data_path=self.config['processed_data_path'],
                models_path=self.config['models_path']
            )
            
            results = trainer.run_model_training_pipeline()
            
            if results['status'] == 'success':
                self._log_pipeline_step(step_name, "success", results)
                return True
            else:
                self._log_pipeline_step(step_name, "failed", results)
                return False
                
        except Exception as e:
            error_details = {'error': str(e), 'type': type(e).__name__}
            self._log_pipeline_step(step_name, "failed", error_details)
            logger.exception(f"Error in {step_name}")
            return False
    
    def run_model_evaluation(self) -> bool:
        """
        Run model evaluation step
        
        Returns:
            bool: True if successful, False otherwise
        """
        step_name = "model_evaluation"
        self._log_pipeline_step(step_name, "started")
        
        try:
            evaluator = ProcurementModelEvaluation(
                models_path=self.config['models_path'],
                processed_data_path=self.config['processed_data_path'],
                artifacts_path=self.config['artifacts_path']
            )
            
            results = evaluator.run_model_evaluation_pipeline()
            
            if results['status'] == 'success':
                self._log_pipeline_step(step_name, "success", results)
                return True
            else:
                self._log_pipeline_step(step_name, "failed", results)
                return False
                
        except Exception as e:
            error_details = {'error': str(e), 'type': type(e).__name__}
            self._log_pipeline_step(step_name, "failed", error_details)
            logger.exception(f"Error in {step_name}")
            return False
    
    def validate_pipeline_outputs(self) -> bool:
        """
        Validate that all expected ML pipeline outputs exist and are valid
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        step_name = "ml_pipeline_validation"
        self._log_pipeline_step(step_name, "started")
        
        try:
            models_path = Path(self.config['models_path'])
            artifacts_path = Path(self.config['artifacts_path'])
            
            # Check required model files exist
            required_model_files = [
                'training_results.json'
            ]
            
            missing_files = []
            for file in required_model_files:
                if not (models_path / file).exists():
                    missing_files.append(str(models_path / file))
            
            # Check for trained model directories
            model_dirs = [d for d in models_path.iterdir() if d.is_dir()]
            
            if len(model_dirs) == 0:
                missing_files.append("No trained model directories found")
            
            # Check evaluation artifacts
            evaluation_files = list(artifacts_path.glob("model_evaluation_report_*.json"))
            
            if len(evaluation_files) == 0:
                missing_files.append("No evaluation reports found")
            
            if missing_files:
                error_details = {'missing_files': missing_files}
                self._log_pipeline_step(step_name, "failed", error_details)
                return False
            
            # Validate model quality
            try:
                with open(models_path / 'training_results.json', 'r') as f:
                    training_results = json.load(f)
                
                trained_models = training_results.get('trained_models', {})
                
                if len(trained_models) == 0:
                    error_details = {'error': 'No models were successfully trained'}
                    self._log_pipeline_step(step_name, "failed", error_details)
                    return False
                
                # Check model performance
                avg_r2_scores = []
                for task_name, task_info in trained_models.items():
                    r2_score = task_info.get('metrics', {}).get('r2', 0)
                    avg_r2_scores.append(r2_score)
                
                avg_r2 = sum(avg_r2_scores) / len(avg_r2_scores) if avg_r2_scores else 0
                
                validation_results = {
                    'trained_models_count': len(trained_models),
                    'average_r2_score': avg_r2,
                    'model_quality': 'Good' if avg_r2 > 0.5 else 'Fair' if avg_r2 > 0.3 else 'Poor',
                    'evaluation_reports_count': len(evaluation_files)
                }
                
                self._log_pipeline_step(step_name, "success", validation_results)
                return True
                
            except Exception as e:
                error_details = {'error': f'Error validating model results: {str(e)}'}
                self._log_pipeline_step(step_name, "failed", error_details)
                return False
            
        except Exception as e:
            error_details = {'error': str(e), 'type': type(e).__name__}
            self._log_pipeline_step(step_name, "failed", error_details)
            logger.exception(f"Error in {step_name}")
            return False
    
    def select_best_models(self) -> Dict[str, Any]:
        """
        Select the best models for deployment based on evaluation results
        
        Returns:
            Dictionary with selected models information
        """
        step_name = "model_selection"
        self._log_pipeline_step(step_name, "started")
        
        try:
            models_path = Path(self.config['models_path'])
            artifacts_path = Path(self.config['artifacts_path'])
            
            # Load training results
            with open(models_path / 'training_results.json', 'r') as f:
                training_results = json.load(f)
            
            # Load latest evaluation report
            evaluation_files = sorted(artifacts_path.glob("model_evaluation_report_*.json"))
            if not evaluation_files:
                raise FileNotFoundError("No evaluation reports found")
            
            with open(evaluation_files[-1], 'r') as f:
                evaluation_report = json.load(f)
            
            selected_models = {}
            deployment_recommendations = []
            
            # Analyze each model for deployment readiness
            for task_name, task_info in training_results['trained_models'].items():
                model_metrics = task_info['metrics']
                
                # Get business impact assessment
                business_impact = evaluation_report['business_impact']['model_impact'].get(task_name, {})
                readiness = business_impact.get('implementation_readiness', {})
                
                selection_criteria = {
                    'r2_score': model_metrics.get('r2', 0),
                    'rmse': model_metrics.get('rmse', float('inf')),
                    'readiness_level': readiness.get('readiness_level', 'Not Ready'),
                    'risk_level': readiness.get('risk_level', 'High')
                }
                
                # Determine deployment recommendation
                if (selection_criteria['r2_score'] > 0.6 and 
                    selection_criteria['readiness_level'] in ['Ready', 'Pilot Ready']):
                    
                    recommendation = 'Deploy'
                    priority = 'High' if selection_criteria['readiness_level'] == 'Ready' else 'Medium'
                    
                elif selection_criteria['r2_score'] > 0.4:
                    recommendation = 'Pilot'
                    priority = 'Medium'
                else:
                    recommendation = 'Improve'
                    priority = 'Low'
                
                selected_models[task_name] = {
                    'model_info': task_info,
                    'selection_criteria': selection_criteria,
                    'recommendation': recommendation,
                    'priority': priority,
                    'business_impact': business_impact
                }
                
                deployment_recommendations.append({
                    'task': task_name,
                    'recommendation': recommendation,
                    'priority': priority,
                    'r2_score': selection_criteria['r2_score']
                })
            
            # Create deployment plan
            deployment_plan = {
                'immediate_deployment': [r for r in deployment_recommendations if r['recommendation'] == 'Deploy' and r['priority'] == 'High'],
                'pilot_deployment': [r for r in deployment_recommendations if r['recommendation'] in ['Deploy', 'Pilot'] and r['priority'] == 'Medium'],
                'future_development': [r for r in deployment_recommendations if r['recommendation'] == 'Improve']
            }
            
            selection_results = {
                'selected_models': selected_models,
                'deployment_recommendations': deployment_recommendations,
                'deployment_plan': deployment_plan,
                'selection_timestamp': datetime.now().isoformat()
            }
            
            # Save model selection results
            selection_file = artifacts_path / f"model_selection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(selection_file, 'w') as f:
                json.dump(selection_results, f, indent=2)
            
            self._log_pipeline_step(step_name, "success", {
                'selected_models_count': len(selected_models),
                'immediate_deployment_count': len(deployment_plan['immediate_deployment']),
                'pilot_deployment_count': len(deployment_plan['pilot_deployment']),
                'selection_file': str(selection_file)
            })
            
            return selection_results
            
        except Exception as e:
            error_details = {'error': str(e), 'type': type(e).__name__}
            self._log_pipeline_step(step_name, "failed", error_details)
            logger.exception(f"Error in {step_name}")
            return {}
    
    def generate_pipeline_report(self) -> Dict:
        """
        Generate a comprehensive ML pipeline execution report
        
        Returns:
            Dict: ML pipeline execution report
        """
        logger.info("Generating ML pipeline execution report...")
        
        # Calculate overall pipeline status
        failed_steps = [step for step, result in self.results.items() if result['status'] == 'failed']
        overall_status = 'failed' if failed_steps else 'success'
        
        # Load model selection results if available
        model_selection_results = {}
        try:
            artifacts_path = Path(self.config['artifacts_path'])
            selection_files = sorted(artifacts_path.glob("model_selection_results_*.json"))
            if selection_files:
                with open(selection_files[-1], 'r') as f:
                    model_selection_results = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load model selection results: {e}")
        
        # Create comprehensive report
        report = {
            'pipeline_id': self.pipeline_id,
            'execution_timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'config': self.config,
            'step_results': self.results,
            'failed_steps': failed_steps,
            'model_selection_results': model_selection_results,
            'summary': {
                'total_steps': len(self.results),
                'successful_steps': len([r for r in self.results.values() if r['status'] == 'success']),
                'failed_steps': len(failed_steps)
            }
        }
        
        # Add model summary if available
        try:
            models_path = Path(self.config['models_path'])
            training_results_file = models_path / 'training_results.json'
            if training_results_file.exists():
                with open(training_results_file, 'r') as f:
                    training_results = json.load(f)
                
                trained_models = training_results.get('trained_models', {})
                
                # Calculate average performance
                r2_scores = []
                for task_info in trained_models.values():
                    r2_scores.append(task_info.get('metrics', {}).get('r2', 0))
                
                avg_r2 = sum(r2_scores) / len(r2_scores) if r2_scores else 0
                
                report['model_summary'] = {
                    'total_models_trained': len(trained_models),
                    'average_r2_score': avg_r2,
                    'model_quality_assessment': 'Excellent' if avg_r2 > 0.8 else 'Good' if avg_r2 > 0.6 else 'Fair' if avg_r2 > 0.4 else 'Poor',
                    'models_ready_for_deployment': len(model_selection_results.get('deployment_plan', {}).get('immediate_deployment', [])),
                    'models_ready_for_pilot': len(model_selection_results.get('deployment_plan', {}).get('pilot_deployment', []))
                }
        except Exception as e:
            logger.warning(f"Could not generate model summary: {e}")
        
        return report
    
    def save_pipeline_report(self, report: Dict) -> str:
        """
        Save ML pipeline report to file
        
        Args:
            report: ML pipeline execution report
            
        Returns:
            str: Path to saved report
        """
        report_path = Path('artifacts') / f'ml_pipeline_report_{self.pipeline_id}.json'
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"ML pipeline report saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error saving ML pipeline report: {e}")
            raise
    
    def run_complete_ml_pipeline(self) -> Dict:
        """
        Run the complete ML pipeline from start to finish
        
        Returns:
            Dict: Complete ML pipeline execution report
        """
        logger.info(f"STARTING: Complete ML pipeline: {self.pipeline_id}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Model Training
            if not self.run_model_training():
                logger.error("Model training failed. Stopping pipeline.")
                report = self.generate_pipeline_report()
                self.save_pipeline_report(report)
                return report
            
            # Step 2: Model Evaluation
            if not self.run_model_evaluation():
                logger.error("Model evaluation failed. Stopping pipeline.")
                report = self.generate_pipeline_report()
                self.save_pipeline_report(report)
                return report
            
            # Step 3: Pipeline Validation
            if not self.validate_pipeline_outputs():
                logger.error("ML pipeline validation failed.")
                report = self.generate_pipeline_report()
                self.save_pipeline_report(report)
                return report
            
            # Step 4: Model Selection
            selection_results = self.select_best_models()
            if not selection_results:
                logger.error("Model selection failed.")
                report = self.generate_pipeline_report()
                self.save_pipeline_report(report)
                return report
            
            # Pipeline completed successfully
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.info(f"SUCCESS: Complete ML pipeline executed successfully in {execution_time:.2f} seconds")
            
            # Generate and save final report
            report = self.generate_pipeline_report()
            report['execution_time_seconds'] = execution_time
            report_path = self.save_pipeline_report(report)
            
            # Print summary
            self._print_pipeline_summary(report)
            
            return report
            
        except Exception as e:
            logger.exception("Fatal error in ML pipeline execution")
            report = self.generate_pipeline_report()
            report['fatal_error'] = str(e)
            self.save_pipeline_report(report)
            return report
    
    def _print_pipeline_summary(self, report: Dict):
        """Print a user-friendly ML pipeline summary"""
        print("\n" + "="*70)
        print("AUTOMATED PROCUREMENT ML PIPELINE SUMMARY")
        print("="*70)
        
        print(f"Pipeline ID: {report['pipeline_id']}")
        print(f"Status: {report['overall_status'].upper()}")
        print(f"Execution Time: {report.get('execution_time_seconds', 0):.2f} seconds")
        
        if 'model_summary' in report:
            model_summary = report['model_summary']
            print(f"\nMODEL TRAINING SUMMARY:")
            print(f"   - Total Models Trained: {model_summary['total_models_trained']}")
            print(f"   - Average R² Score: {model_summary['average_r2_score']:.3f}")
            print(f"   - Model Quality: {model_summary['model_quality_assessment']}")
            print(f"   - Ready for Deployment: {model_summary['models_ready_for_deployment']}")
            print(f"   - Ready for Pilot: {model_summary['models_ready_for_pilot']}")
        
        if 'model_selection_results' in report and report['model_selection_results']:
            selection_results = report['model_selection_results']
            deployment_plan = selection_results.get('deployment_plan', {})
            
            print(f"\nDEPLOYMENT RECOMMENDATIONS:")
            print(f"   - Immediate Deployment: {len(deployment_plan.get('immediate_deployment', []))} models")
            print(f"   - Pilot Deployment: {len(deployment_plan.get('pilot_deployment', []))} models")
            print(f"   - Future Development: {len(deployment_plan.get('future_development', []))} models")
        
        print(f"\nOutput Locations:")
        print(f"   - Models: {self.config['models_path']}")
        print(f"   - Artifacts: {self.config['artifacts_path']}")
        
        print("="*70)


def main():
    """Main function to run the ML pipeline"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run Automated Procurement ML Pipeline')
    parser.add_argument('--processed-data-path', default='data/processed', help='Path to processed data')
    parser.add_argument('--models-path', default='models', help='Path for model output')
    parser.add_argument('--artifacts-path', default='artifacts', help='Path for artifacts output')
    parser.add_argument('--config-file', help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config_file and Path(args.config_file).exists():
        try:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from: {args.config_file}")
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
    
    # Override with command line arguments
    if not config:
        config = {}
    
    config['processed_data_path'] = args.processed_data_path
    config['models_path'] = args.models_path
    config['artifacts_path'] = args.artifacts_path
    
    try:
        # Initialize and run ML pipeline
        pipeline = ProcurementMLPipeline(config)
        report = pipeline.run_complete_ml_pipeline()
        
        # Exit with appropriate code
        if report['overall_status'] == 'success':
            print("\nSUCCESS: ML Pipeline completed successfully!")
            print("Models are trained, evaluated, and ready for deployment!")
            sys.exit(0)
        else:
            print("\nFAILED: ML Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("ML Pipeline interrupted by user")
        print("\nML Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Fatal error in main")
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()